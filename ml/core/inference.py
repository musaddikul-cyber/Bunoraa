"""
Inference Engine for Production ML Model Serving

Provides:
- Batch and real-time inference
- Model caching
- Request batching
- Fallback handling
- Monitoring and logging
"""

import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from threading import Lock
import numpy as np

from .registry import ModelRegistry, get_registry, ModelStatus

logger = logging.getLogger("bunoraa.ml.inference")


@dataclass
class PredictionRequest:
    """Container for prediction requests."""
    request_id: str
    features: np.ndarray
    model_name: str
    version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class PredictionResponse:
    """Container for prediction responses."""
    request_id: str
    predictions: np.ndarray
    model_id: str
    model_version: str
    latency_ms: float
    timestamp: str = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class ModelCache:
    """LRU cache for loaded models."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = Lock()
    
    def get(self, model_id: str) -> Optional[Any]:
        """Get model from cache."""
        with self._lock:
            if model_id in self._cache:
                self._access_times[model_id] = time.time()
                return self._cache[model_id]
        return None
    
    def put(self, model_id: str, model: Any):
        """Add model to cache."""
        with self._lock:
            # Evict if full
            if len(self._cache) >= self.max_size:
                oldest = min(self._access_times, key=self._access_times.get)
                del self._cache[oldest]
                del self._access_times[oldest]
            
            self._cache[model_id] = model
            self._access_times[model_id] = time.time()
    
    def invalidate(self, model_id: str):
        """Remove model from cache."""
        with self._lock:
            self._cache.pop(model_id, None)
            self._access_times.pop(model_id, None)
    
    def clear(self):
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()


class InferenceMetrics:
    """Collect inference metrics for monitoring."""
    
    def __init__(self):
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._lock = Lock()
    
    def record_request(self, model_id: str, latency_ms: float, error: bool = False):
        """Record a prediction request."""
        with self._lock:
            self._latencies[model_id].append(latency_ms)
            self._request_counts[model_id] += 1
            if error:
                self._error_counts[model_id] += 1
            
            # Keep only last 1000 latencies
            if len(self._latencies[model_id]) > 1000:
                self._latencies[model_id] = self._latencies[model_id][-1000:]
    
    def get_stats(self, model_id: str) -> Dict[str, Any]:
        """Get statistics for a model."""
        with self._lock:
            latencies = self._latencies.get(model_id, [])
            if not latencies:
                return {
                    "model_id": model_id,
                    "request_count": 0,
                    "error_count": 0,
                    "avg_latency_ms": 0,
                    "p50_latency_ms": 0,
                    "p95_latency_ms": 0,
                    "p99_latency_ms": 0,
                }
            
            latencies_sorted = sorted(latencies)
            return {
                "model_id": model_id,
                "request_count": self._request_counts[model_id],
                "error_count": self._error_counts[model_id],
                "error_rate": self._error_counts[model_id] / max(self._request_counts[model_id], 1),
                "avg_latency_ms": np.mean(latencies),
                "p50_latency_ms": latencies_sorted[len(latencies) // 2],
                "p95_latency_ms": latencies_sorted[int(len(latencies) * 0.95)],
                "p99_latency_ms": latencies_sorted[int(len(latencies) * 0.99)],
            }
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all models."""
        with self._lock:
            return [self.get_stats(model_id) for model_id in self._request_counts.keys()]


class InferenceEngine:
    """
    Production inference engine for ML models.
    
    Features:
    - Automatic model loading and caching
    - Request batching for efficiency
    - Fallback handling
    - A/B testing support
    - Comprehensive monitoring
    """
    
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        cache_size: int = 10,
        default_timeout_seconds: float = 30.0
    ):
        self.registry = registry or get_registry()
        self.cache = ModelCache(max_size=cache_size)
        self.metrics = InferenceMetrics()
        self.default_timeout = default_timeout_seconds
        
        # Fallback models
        self._fallback_handlers: Dict[str, Callable] = {}
        
        # Request batching
        self._batch_queues: Dict[str, List] = defaultdict(list)
        self._batch_locks: Dict[str, Lock] = defaultdict(Lock)
    
    def register_fallback(self, model_name: str, handler: Callable):
        """
        Register a fallback handler for when model inference fails.
        
        Args:
            model_name: Name of the model
            handler: Function that takes features and returns predictions
        """
        self._fallback_handlers[model_name] = handler
    
    def _load_model(self, model_id: str) -> Any:
        """Load model, using cache if available."""
        model = self.cache.get(model_id)
        if model is not None:
            return model
        
        model = self.registry.load_model(model_id)
        self.cache.put(model_id, model)
        return model
    
    def _get_production_model_id(self, model_name: str) -> Optional[str]:
        """Get the production model ID for a model name."""
        entry = self.registry.get_production_model(model_name)
        return entry.model_id if entry else None
    
    def predict(
        self,
        request: PredictionRequest
    ) -> PredictionResponse:
        """
        Make a prediction using the appropriate model.
        
        Args:
            request: PredictionRequest with features and model info
        
        Returns:
            PredictionResponse with predictions
        """
        start_time = time.time()
        
        try:
            # Get model
            model_id = self._get_production_model_id(request.model_name)
            if not model_id:
                raise ValueError(f"No production model found for {request.model_name}")
            
            entry = self.registry.get(model_id)
            model = self._load_model(model_id)
            
            # Make prediction
            if hasattr(model, 'predict'):
                predictions = model.predict(request.features)
            elif hasattr(model, '__call__'):
                predictions = model(request.features)
            else:
                raise ValueError(f"Model {model_id} does not have predict method")
            
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_request(model_id, latency_ms)
            
            return PredictionResponse(
                request_id=request.request_id,
                predictions=predictions,
                model_id=model_id,
                model_version=entry.version,
                latency_ms=latency_ms,
                metadata={"success": True}
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Prediction failed for {request.model_name}: {e}")
            
            # Try fallback
            if request.model_name in self._fallback_handlers:
                try:
                    predictions = self._fallback_handlers[request.model_name](request.features)
                    return PredictionResponse(
                        request_id=request.request_id,
                        predictions=predictions,
                        model_id="fallback",
                        model_version="0.0.0",
                        latency_ms=latency_ms,
                        metadata={"success": True, "fallback": True}
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            # Record error
            model_id = self._get_production_model_id(request.model_name) or "unknown"
            self.metrics.record_request(model_id, latency_ms, error=True)
            
            raise
    
    def predict_batch(
        self,
        requests: List[PredictionRequest]
    ) -> List[PredictionResponse]:
        """
        Make predictions for a batch of requests.
        
        Args:
            requests: List of PredictionRequest objects
        
        Returns:
            List of PredictionResponse objects
        """
        # Group by model
        by_model: Dict[str, List[Tuple[int, PredictionRequest]]] = defaultdict(list)
        for i, req in enumerate(requests):
            by_model[req.model_name].append((i, req))
        
        responses = [None] * len(requests)
        
        for model_name, indexed_requests in by_model.items():
            indices, reqs = zip(*indexed_requests)
            
            # Stack features
            features = np.vstack([r.features for r in reqs])
            
            # Create single batch request
            batch_request = PredictionRequest(
                request_id=f"batch_{hashlib.md5(str(indices).encode()).hexdigest()[:8]}",
                features=features,
                model_name=model_name
            )
            
            # Get batch prediction
            batch_response = self.predict(batch_request)
            
            # Split back to individual responses
            for i, idx in enumerate(indices):
                responses[idx] = PredictionResponse(
                    request_id=reqs[i].request_id,
                    predictions=batch_response.predictions[i:i+1],
                    model_id=batch_response.model_id,
                    model_version=batch_response.model_version,
                    latency_ms=batch_response.latency_ms,
                    metadata=batch_response.metadata
                )
        
        return responses
    
    def predict_with_ab_test(
        self,
        request: PredictionRequest,
        experiment_id: str
    ) -> PredictionResponse:
        """
        Make prediction with A/B testing support.
        
        Routes request to appropriate model version based on
        traffic percentages configured in the registry.
        
        Args:
            request: PredictionRequest
            experiment_id: Unique ID for the experiment
        
        Returns:
            PredictionResponse from selected model
        """
        import random
        
        # Get all production models for this name
        models = self.registry.get_by_name(
            request.model_name,
            status=ModelStatus.PRODUCTION
        )
        
        if not models:
            raise ValueError(f"No production models for {request.model_name}")
        
        # Calculate traffic allocation
        total_traffic = sum(m.traffic_percentage for m in models if not m.is_shadow)
        if total_traffic == 0:
            # Fallback to first model
            selected = models[0]
        else:
            # Select based on traffic percentage
            roll = random.random() * total_traffic
            cumulative = 0.0
            selected = models[0]
            
            for model in models:
                if model.is_shadow:
                    continue
                cumulative += model.traffic_percentage
                if roll <= cumulative:
                    selected = model
                    break
        
        # Make prediction with selected model
        model = self._load_model(selected.model_id)
        
        start_time = time.time()
        
        if hasattr(model, 'predict'):
            predictions = model.predict(request.features)
        else:
            predictions = model(request.features)
        
        latency_ms = (time.time() - start_time) * 1000
        self.metrics.record_request(selected.model_id, latency_ms)
        
        # Log for shadow models
        for model_entry in models:
            if model_entry.is_shadow:
                try:
                    shadow_model = self._load_model(model_entry.model_id)
                    shadow_start = time.time()
                    if hasattr(shadow_model, 'predict'):
                        shadow_predictions = shadow_model.predict(request.features)
                    else:
                        shadow_predictions = shadow_model(request.features)
                    shadow_latency = (time.time() - shadow_start) * 1000
                    
                    logger.info(f"Shadow prediction from {model_entry.model_id}: "
                               f"latency={shadow_latency:.2f}ms")
                    self.metrics.record_request(model_entry.model_id, shadow_latency)
                except Exception as e:
                    logger.warning(f"Shadow model {model_entry.model_id} failed: {e}")
        
        return PredictionResponse(
            request_id=request.request_id,
            predictions=predictions,
            model_id=selected.model_id,
            model_version=selected.version,
            latency_ms=latency_ms,
            metadata={
                "experiment_id": experiment_id,
                "selected_model": selected.model_id
            }
        )
    
    def get_model_stats(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get inference statistics for models."""
        if model_name:
            model_id = self._get_production_model_id(model_name)
            if model_id:
                return [self.metrics.get_stats(model_id)]
            return []
        return self.metrics.get_all_stats()
    
    def reload_model(self, model_name: str):
        """Reload model from registry (invalidate cache)."""
        model_id = self._get_production_model_id(model_name)
        if model_id:
            self.cache.invalidate(model_id)
            self._load_model(model_id)
            logger.info(f"Reloaded model: {model_id}")
    
    def warm_cache(self, model_names: List[str]):
        """Pre-load models into cache."""
        for name in model_names:
            model_id = self._get_production_model_id(name)
            if model_id:
                self._load_model(model_id)
                logger.info(f"Warmed cache for: {model_id}")


# Singleton inference engine
_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get or create the global inference engine."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
