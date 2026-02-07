"""
Feature Store for ML Feature Management

Provides centralized feature engineering and storage:
- Feature computation and caching
- Feature versioning
- Real-time feature serving
- Batch feature extraction
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger("bunoraa.ml.features")


@dataclass
class FeatureDefinition:
    """Definition of a feature."""
    name: str
    dtype: str  # float, int, string, embedding
    description: str = ""
    entity: str = ""  # user, product, order, etc.
    computation_fn: Optional[str] = None  # Name of registered computation function
    dependencies: List[str] = field(default_factory=list)
    ttl_seconds: int = 3600
    is_realtime: bool = False
    default_value: Any = None


class FeatureBackend(ABC):
    """Abstract backend for feature storage."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        pass
    
    @abstractmethod
    def delete(self, key: str):
        pass
    
    @abstractmethod
    def get_batch(self, keys: List[str]) -> Dict[str, Any]:
        pass


class MemoryBackend(FeatureBackend):
    """In-memory feature storage (for development/testing)."""
    
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._expiry:
            if datetime.utcnow() > self._expiry[key]:
                del self._store[key]
                del self._expiry[key]
                return None
        return self._store.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self._store[key] = value
        if ttl:
            self._expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)
    
    def delete(self, key: str):
        self._store.pop(key, None)
        self._expiry.pop(key, None)
    
    def get_batch(self, keys: List[str]) -> Dict[str, Any]:
        return {k: self.get(k) for k in keys if self.get(k) is not None}


class RedisBackend(FeatureBackend):
    """Redis-based feature storage for production."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "bunoraa:features:"):
        self.prefix = prefix
        self._client = None
        self._redis_url = redis_url
    
    @property
    def client(self):
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self._redis_url)
            except ImportError:
                raise ImportError("redis package required. Install with: pip install redis")
        return self._client
    
    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        try:
            value = self.client.get(self._key(key))
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            self.client.set(
                self._key(key),
                json.dumps(value),
                ex=ttl
            )
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
    
    def delete(self, key: str):
        try:
            self.client.delete(self._key(key))
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
    
    def get_batch(self, keys: List[str]) -> Dict[str, Any]:
        try:
            values = self.client.mget([self._key(k) for k in keys])
            result = {}
            for k, v in zip(keys, values):
                if v:
                    result[k] = json.loads(v)
            return result
        except Exception as e:
            logger.warning(f"Redis mget failed: {e}")
            return {}


class FeatureStore:
    """
    Central feature store for ML features.
    
    Provides:
    - Feature registration and versioning
    - Feature computation with caching
    - Batch feature extraction
    - Real-time feature serving
    """
    
    def __init__(
        self,
        backend: Optional[FeatureBackend] = None,
        cache_ttl: int = 3600
    ):
        self.backend = backend or MemoryBackend()
        self.cache_ttl = cache_ttl
        
        # Feature definitions
        self._features: Dict[str, FeatureDefinition] = {}
        
        # Computation functions
        self._computations: Dict[str, Callable] = {}
        
        # Feature groups
        self._groups: Dict[str, List[str]] = {}
    
    def register_feature(
        self,
        name: str,
        dtype: str,
        entity: str,
        description: str = "",
        computation_fn: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
        ttl_seconds: int = 3600,
        is_realtime: bool = False,
        default_value: Any = None
    ) -> FeatureDefinition:
        """
        Register a new feature.
        
        Args:
            name: Unique feature name
            dtype: Data type (float, int, string, embedding)
            entity: Entity type (user, product, etc.)
            description: Feature description
            computation_fn: Function to compute the feature
            dependencies: List of dependent feature names
            ttl_seconds: Cache TTL
            is_realtime: Whether feature needs real-time computation
            default_value: Default value when feature is missing
        
        Returns:
            FeatureDefinition
        """
        feature = FeatureDefinition(
            name=name,
            dtype=dtype,
            entity=entity,
            description=description,
            computation_fn=name if computation_fn else None,
            dependencies=dependencies or [],
            ttl_seconds=ttl_seconds,
            is_realtime=is_realtime,
            default_value=default_value
        )
        
        self._features[name] = feature
        
        if computation_fn:
            self._computations[name] = computation_fn
        
        logger.info(f"Registered feature: {name}")
        return feature
    
    def register_group(self, group_name: str, feature_names: List[str]):
        """Register a group of features for batch retrieval."""
        self._groups[group_name] = feature_names
    
    def _feature_key(self, feature_name: str, entity_id: str) -> str:
        """Generate cache key for a feature."""
        return f"{feature_name}:{entity_id}"
    
    def get_feature(
        self,
        feature_name: str,
        entity_id: str,
        compute_if_missing: bool = True
    ) -> Optional[Any]:
        """
        Get a single feature value.
        
        Args:
            feature_name: Name of the feature
            entity_id: Entity ID (user_id, product_id, etc.)
            compute_if_missing: Whether to compute if not cached
        
        Returns:
            Feature value or None
        """
        feature = self._features.get(feature_name)
        if not feature:
            logger.warning(f"Feature not found: {feature_name}")
            return None
        
        key = self._feature_key(feature_name, entity_id)
        
        # Check cache
        value = self.backend.get(key)
        if value is not None:
            return value
        
        # Compute if needed
        if compute_if_missing and feature.computation_fn:
            computation = self._computations.get(feature.computation_fn)
            if computation:
                try:
                    # Get dependencies first
                    deps = {}
                    for dep_name in feature.dependencies:
                        deps[dep_name] = self.get_feature(dep_name, entity_id)
                    
                    value = computation(entity_id, deps)
                    
                    # Cache the result
                    self.backend.set(key, value, feature.ttl_seconds)
                    return value
                except Exception as e:
                    logger.error(f"Feature computation failed for {feature_name}: {e}")
        
        return feature.default_value
    
    def get_features(
        self,
        feature_names: List[str],
        entity_id: str,
        compute_if_missing: bool = True
    ) -> Dict[str, Any]:
        """
        Get multiple features for an entity.
        
        Args:
            feature_names: List of feature names
            entity_id: Entity ID
            compute_if_missing: Whether to compute missing features
        
        Returns:
            Dictionary of feature name to value
        """
        keys = [self._feature_key(name, entity_id) for name in feature_names]
        cached = self.backend.get_batch(keys)
        
        result = {}
        for name in feature_names:
            key = self._feature_key(name, entity_id)
            if key in cached:
                result[name] = cached[key]
            else:
                result[name] = self.get_feature(
                    name, entity_id, compute_if_missing
                )
        
        return result
    
    def get_feature_group(
        self,
        group_name: str,
        entity_id: str,
        compute_if_missing: bool = True
    ) -> Dict[str, Any]:
        """Get all features in a group."""
        if group_name not in self._groups:
            logger.warning(f"Feature group not found: {group_name}")
            return {}
        
        return self.get_features(
            self._groups[group_name],
            entity_id,
            compute_if_missing
        )
    
    def get_batch_features(
        self,
        feature_names: List[str],
        entity_ids: List[str],
        compute_if_missing: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get features for multiple entities.
        
        Args:
            feature_names: List of feature names
            entity_ids: List of entity IDs
            compute_if_missing: Whether to compute missing features
        
        Returns:
            Dictionary of entity_id to feature dict
        """
        result = {}
        for entity_id in entity_ids:
            result[entity_id] = self.get_features(
                feature_names, entity_id, compute_if_missing
            )
        return result
    
    def set_feature(
        self,
        feature_name: str,
        entity_id: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set a feature value directly."""
        feature = self._features.get(feature_name)
        if not feature:
            logger.warning(f"Setting unregistered feature: {feature_name}")
        
        key = self._feature_key(feature_name, entity_id)
        ttl = ttl or (feature.ttl_seconds if feature else self.cache_ttl)
        self.backend.set(key, value, ttl)
    
    def invalidate_feature(self, feature_name: str, entity_id: str):
        """Invalidate a cached feature."""
        key = self._feature_key(feature_name, entity_id)
        self.backend.delete(key)
    
    def invalidate_entity(self, entity_id: str):
        """Invalidate all features for an entity."""
        for feature_name in self._features:
            self.invalidate_feature(feature_name, entity_id)
    
    def get_feature_vector(
        self,
        feature_names: List[str],
        entity_id: str,
        as_numpy: bool = True
    ) -> Union[List[float], np.ndarray]:
        """
        Get features as a vector for ML model input.
        
        Args:
            feature_names: Ordered list of feature names
            entity_id: Entity ID
            as_numpy: Whether to return numpy array
        
        Returns:
            Feature vector
        """
        features = self.get_features(feature_names, entity_id)
        
        vector = []
        for name in feature_names:
            value = features.get(name)
            feature_def = self._features.get(name)
            
            if value is None:
                value = feature_def.default_value if feature_def else 0.0
            
            if isinstance(value, (list, np.ndarray)):
                vector.extend(value)
            else:
                vector.append(float(value) if value is not None else 0.0)
        
        if as_numpy:
            return np.array(vector, dtype=np.float32)
        return vector
    
    def get_batch_feature_vectors(
        self,
        feature_names: List[str],
        entity_ids: List[str]
    ) -> np.ndarray:
        """
        Get feature vectors for multiple entities.
        
        Args:
            feature_names: Ordered list of feature names
            entity_ids: List of entity IDs
        
        Returns:
            2D numpy array of shape (len(entity_ids), num_features)
        """
        vectors = [
            self.get_feature_vector(feature_names, eid)
            for eid in entity_ids
        ]
        return np.vstack(vectors)
    
    def list_features(self, entity: Optional[str] = None) -> List[FeatureDefinition]:
        """List all registered features."""
        features = list(self._features.values())
        if entity:
            features = [f for f in features if f.entity == entity]
        return features


# Singleton feature store
_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get or create the global feature store."""
    global _store
    if _store is None:
        # Try to use Redis in production
        try:
            from django.conf import settings
            redis_url = getattr(settings, 'REDIS_URL', None)
            if redis_url:
                _store = FeatureStore(backend=RedisBackend(redis_url))
            else:
                _store = FeatureStore()
        except Exception:
            _store = FeatureStore()
    return _store
