"""
Model Registry for Version Control and Deployment

Provides:
- Model versioning and tracking
- Model deployment management
- A/B testing support
- Model comparison
- Rollback capabilities
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("bunoraa.ml.registry")


class ModelStatus(Enum):
    """Model lifecycle status."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelEntry:
    """Registry entry for a model."""
    model_id: str
    model_name: str
    version: str
    model_type: str
    framework: str
    status: str
    
    # Paths
    artifact_path: str
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    deployed_at: Optional[str] = None
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Training info
    training_data_hash: str = ""
    num_training_samples: int = 0
    training_time_seconds: float = 0.0
    
    # Tags and notes
    tags: List[str] = field(default_factory=list)
    description: str = ""
    notes: str = ""
    
    # A/B testing
    traffic_percentage: float = 0.0
    is_shadow: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        return cls(**data)


class ModelRegistry:
    """
    Central registry for ML model management.
    
    Features:
    - Track all trained models
    - Manage model versions
    - Deploy/undeploy models
    - Support A/B testing
    - Enable rollbacks
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path(__file__).parent.parent / "registry"
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        self.registry_file = self.registry_path / "registry.json"
        
        self._entries: Dict[str, ModelEntry] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    data = json.load(f)
                    self._entries = {
                        k: ModelEntry.from_dict(v) 
                        for k, v in data.get("models", {}).items()
                    }
                logger.info(f"Loaded {len(self._entries)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self._entries = {}
        else:
            self._entries = {}
    
    def _save_registry(self):
        """Save registry to disk."""
        data = {
            "models": {k: v.to_dict() for k, v in self._entries.items()},
            "updated_at": datetime.utcnow().isoformat()
        }
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_model_id(self, model_name: str, version: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{model_name}-{version}-{timestamp}"
    
    def register(
        self,
        model: Any,
        model_name: str,
        version: str,
        model_type: str = "unknown",
        framework: str = "unknown",
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        **kwargs
    ) -> ModelEntry:
        """
        Register a new model in the registry.
        
        Args:
            model: The model object to register
            model_name: Name of the model
            version: Version string
            model_type: Type of model (recommendation, classification, etc.)
            framework: ML framework used
            metrics: Model performance metrics
            tags: Optional tags for organization
            description: Model description
        
        Returns:
            ModelEntry for the registered model
        """
        model_id = self._generate_model_id(model_name, version)
        
        # Save model artifacts
        artifact_path = self.models_path / model_id
        artifact_path.mkdir(parents=True, exist_ok=True)
        
        # Save model using its save method if available
        if hasattr(model, 'save'):
            model.save(artifact_path)
        else:
            # Fallback to pickle
            import pickle
            with open(artifact_path / "model.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        # Create entry
        entry = ModelEntry(
            model_id=model_id,
            model_name=model_name,
            version=version,
            model_type=model_type,
            framework=framework,
            status=ModelStatus.DEVELOPMENT.value,
            artifact_path=str(artifact_path),
            metrics=metrics or {},
            tags=tags or [],
            description=description,
            **kwargs
        )
        
        self._entries[model_id] = entry
        self._save_registry()
        
        logger.info(f"Registered model: {model_id}")
        return entry
    
    def get(self, model_id: str) -> Optional[ModelEntry]:
        """Get model entry by ID."""
        return self._entries.get(model_id)
    
    def get_by_name(
        self,
        model_name: str,
        version: Optional[str] = None,
        status: Optional[ModelStatus] = None
    ) -> List[ModelEntry]:
        """
        Get all models with the given name.
        
        Args:
            model_name: Name to search for
            version: Optional version filter
            status: Optional status filter
        
        Returns:
            List of matching ModelEntry objects
        """
        results = []
        for entry in self._entries.values():
            if entry.model_name != model_name:
                continue
            if version and entry.version != version:
                continue
            if status and entry.status != status.value:
                continue
            results.append(entry)
        
        # Sort by created_at descending
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results
    
    def get_production_model(self, model_name: str) -> Optional[ModelEntry]:
        """Get the current production model for a given name."""
        models = self.get_by_name(model_name, status=ModelStatus.PRODUCTION)
        return models[0] if models else None
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelEntry]:
        """
        List all models with optional filters.
        
        Args:
            model_type: Filter by model type
            status: Filter by status
            tags: Filter by tags (models must have all specified tags)
        
        Returns:
            List of matching ModelEntry objects
        """
        results = []
        for entry in self._entries.values():
            if model_type and entry.model_type != model_type:
                continue
            if status and entry.status != status.value:
                continue
            if tags and not all(t in entry.tags for t in tags):
                continue
            results.append(entry)
        
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results
    
    def promote_to_staging(self, model_id: str) -> ModelEntry:
        """Promote model to staging."""
        entry = self._entries.get(model_id)
        if not entry:
            raise ValueError(f"Model not found: {model_id}")
        
        entry.status = ModelStatus.STAGING.value
        entry.updated_at = datetime.utcnow().isoformat()
        self._save_registry()
        
        logger.info(f"Promoted {model_id} to staging")
        return entry
    
    def deploy_to_production(
        self,
        model_id: str,
        traffic_percentage: float = 100.0,
        is_shadow: bool = False
    ) -> ModelEntry:
        """
        Deploy model to production.
        
        Args:
            model_id: Model to deploy
            traffic_percentage: Percentage of traffic to route to this model
            is_shadow: If True, run in shadow mode (log but don't serve)
        
        Returns:
            Updated ModelEntry
        """
        entry = self._entries.get(model_id)
        if not entry:
            raise ValueError(f"Model not found: {model_id}")
        
        # If not shadow and full traffic, demote other production models
        if not is_shadow and traffic_percentage == 100.0:
            for other_entry in self._entries.values():
                if (other_entry.model_name == entry.model_name and 
                    other_entry.model_id != model_id and
                    other_entry.status == ModelStatus.PRODUCTION.value):
                    other_entry.status = ModelStatus.ARCHIVED.value
                    other_entry.traffic_percentage = 0.0
                    other_entry.updated_at = datetime.utcnow().isoformat()
        
        entry.status = ModelStatus.PRODUCTION.value
        entry.deployed_at = datetime.utcnow().isoformat()
        entry.traffic_percentage = traffic_percentage
        entry.is_shadow = is_shadow
        entry.updated_at = datetime.utcnow().isoformat()
        
        self._save_registry()
        
        logger.info(f"Deployed {model_id} to production (traffic: {traffic_percentage}%, shadow: {is_shadow})")
        return entry
    
    def rollback(self, model_name: str) -> Optional[ModelEntry]:
        """
        Rollback to the previous production version.
        
        Args:
            model_name: Name of model to rollback
        
        Returns:
            The newly active ModelEntry, or None if no previous version
        """
        # Get archived models sorted by deployed_at
        archived = [
            e for e in self._entries.values()
            if e.model_name == model_name and e.status == ModelStatus.ARCHIVED.value
        ]
        archived.sort(key=lambda x: x.deployed_at or "", reverse=True)
        
        if not archived:
            logger.warning(f"No previous version to rollback to for {model_name}")
            return None
        
        # Demote current production
        current_prod = self.get_production_model(model_name)
        if current_prod:
            current_prod.status = ModelStatus.ARCHIVED.value
            current_prod.traffic_percentage = 0.0
            current_prod.updated_at = datetime.utcnow().isoformat()
        
        # Promote previous version
        previous = archived[0]
        previous.status = ModelStatus.PRODUCTION.value
        previous.traffic_percentage = 100.0
        previous.deployed_at = datetime.utcnow().isoformat()
        previous.updated_at = datetime.utcnow().isoformat()
        
        self._save_registry()
        
        logger.info(f"Rolled back {model_name} to {previous.model_id}")
        return previous
    
    def archive(self, model_id: str) -> ModelEntry:
        """Archive a model."""
        entry = self._entries.get(model_id)
        if not entry:
            raise ValueError(f"Model not found: {model_id}")
        
        entry.status = ModelStatus.ARCHIVED.value
        entry.traffic_percentage = 0.0
        entry.updated_at = datetime.utcnow().isoformat()
        self._save_registry()
        
        logger.info(f"Archived model: {model_id}")
        return entry
    
    def delete(self, model_id: str, force: bool = False):
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model to delete
            force: If True, delete even if in production
        """
        entry = self._entries.get(model_id)
        if not entry:
            raise ValueError(f"Model not found: {model_id}")
        
        if entry.status == ModelStatus.PRODUCTION.value and not force:
            raise ValueError("Cannot delete production model. Use force=True or demote first.")
        
        # Delete artifacts
        artifact_path = Path(entry.artifact_path)
        if artifact_path.exists():
            shutil.rmtree(artifact_path)
        
        del self._entries[model_id]
        self._save_registry()
        
        logger.info(f"Deleted model: {model_id}")
    
    def load_model(self, model_id: str, model_class: Optional[Type] = None) -> Any:
        """
        Load a model from the registry.
        
        Args:
            model_id: Model to load
            model_class: Optional class to use for loading
        
        Returns:
            Loaded model object
        """
        entry = self._entries.get(model_id)
        if not entry:
            raise ValueError(f"Model not found: {model_id}")
        
        artifact_path = Path(entry.artifact_path)
        
        if model_class and hasattr(model_class, 'load'):
            return model_class.load(artifact_path)
        
        # Try common loading patterns
        if (artifact_path / "model.pt").exists():
            import torch
            return torch.load(artifact_path / "model.pt")
        elif (artifact_path / "model.pkl").exists():
            import pickle
            with open(artifact_path / "model.pkl", 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Cannot determine how to load model from {artifact_path}")
    
    def compare_models(
        self,
        model_ids: List[str],
        metric: str = "accuracy"
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple models by metrics.
        
        Args:
            model_ids: List of model IDs to compare
            metric: Primary metric for comparison
        
        Returns:
            List of comparison dicts sorted by metric
        """
        comparisons = []
        for model_id in model_ids:
            entry = self._entries.get(model_id)
            if entry:
                comparisons.append({
                    "model_id": model_id,
                    "model_name": entry.model_name,
                    "version": entry.version,
                    "status": entry.status,
                    metric: entry.metrics.get(metric, 0),
                    "all_metrics": entry.metrics
                })
        
        comparisons.sort(key=lambda x: x[metric], reverse=True)
        return comparisons


# Singleton registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create the global registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
