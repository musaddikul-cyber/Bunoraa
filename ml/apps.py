"""
ML Django App Configuration

This app provides comprehensive ML/NN capabilities for the Bunoraa e-commerce platform.
"""

from django.apps import AppConfig


class MLConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml'
    verbose_name = 'Machine Learning'
    
    def ready(self):
        """
        Initialize ML components when Django starts.
        
        This method:
        1. Registers signal handlers for data collection
        2. Initializes the model registry
        3. Sets up auto-training if in production mode
        """
        # Import signals to register handlers
        try:
            from . import signals  # noqa: F401
        except ImportError:
            pass
        
        # Initialize model registry (lazy loading)
        self._init_model_registry()
        
        # Log startup (debug level to avoid noise)
        import logging
        logger = logging.getLogger("bunoraa.ml")
        logger.debug("ML app initialized")
    
    def _init_model_registry(self):
        """Initialize the model registry for lazy loading."""
        # This is deferred to avoid circular imports
        pass
