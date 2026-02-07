"""
ML Auto-Train Command

Django management command to manage automatic model training.

Usage:
    python manage.py ml_auto_train                    # Run auto-training check
    python manage.py ml_auto_train --force           # Force training for all models
    python manage.py ml_auto_train --schedule        # Show training schedule
    python manage.py ml_auto_train --history         # Show training history
"""

import logging
import json
from datetime import datetime

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger("bunoraa.ml.commands")


class Command(BaseCommand):
    help = "Manage automatic ML model training"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--check',
            action='store_true',
            help='Check which models need training without triggering',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force training for all models',
        )
        parser.add_argument(
            '--model',
            type=str,
            default=None,
            help='Force training for specific model',
        )
        parser.add_argument(
            '--schedule',
            action='store_true',
            help='Show auto-training schedule',
        )
        parser.add_argument(
            '--history',
            type=int,
            default=0,
            metavar='N',
            help='Show last N training history entries',
        )
        parser.add_argument(
            '--config',
            action='store_true',
            help='Show current auto-training configuration',
        )
        parser.add_argument(
            '--enable',
            action='store_true',
            help='Enable auto-training',
        )
        parser.add_argument(
            '--disable',
            action='store_true',
            help='Disable auto-training',
        )
        parser.add_argument(
            '--async',
            action='store_true',
            dest='run_async',
            help='Run training asynchronously via Celery',
        )
    
    def handle(self, *args, **options):
        # Handle config display
        if options['config']:
            self._show_config()
            return
        
        # Handle schedule display
        if options['schedule']:
            self._show_schedule()
            return
        
        # Handle history display
        if options['history']:
            self._show_history(options['history'])
            return
        
        # Handle enable/disable
        if options['enable'] or options['disable']:
            self._toggle_auto_training(options['enable'])
            return
        
        # Import auto-training manager
        try:
            from ml.auto_training import AutoTrainingManager, TrainingConfig
        except ImportError as e:
            raise CommandError(f"Failed to import auto-training module: {e}")
        
        # Create manager
        config = TrainingConfig.from_settings()
        manager = AutoTrainingManager(config)
        
        # Check only mode
        if options['check']:
            self._check_training_needed(manager)
            return
        
        # Force training for specific model
        if options['model']:
            self._force_train_model(options['model'], options['run_async'])
            return
        
        # Force training for all models
        if options['force']:
            self._force_train_all(options['run_async'])
            return
        
        # Run auto-training check
        self._run_auto_training(manager, options['run_async'])
    
    def _show_config(self):
        """Display current auto-training configuration."""
        from ml.auto_training import TrainingConfig
        
        config = TrainingConfig.from_settings()
        is_production = getattr(settings, 'PRODUCTION', False)
        
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write("AUTO-TRAINING CONFIGURATION")
        self.stdout.write("=" * 50)
        
        self.stdout.write(f"\nEnvironment: {'PRODUCTION' if is_production else 'DEVELOPMENT'}")
        
        if config.enabled:
            self.stdout.write(self.style.SUCCESS("Status: ENABLED"))
        else:
            self.stdout.write(self.style.WARNING("Status: DISABLED"))
        
        self.stdout.write("\nData Requirements:")
        self.stdout.write(f"  Min Interactions: {config.min_interactions}")
        self.stdout.write(f"  Min Users: {config.min_users}")
        self.stdout.write(f"  Min Products: {config.min_products}")
        
        self.stdout.write("\nTraining Triggers:")
        self.stdout.write(f"  New Interactions Threshold: {config.new_interactions_threshold}")
        self.stdout.write(f"  New Users Threshold: {config.new_users_threshold}")
        self.stdout.write(f"  New Products Threshold: {config.new_products_threshold}")
        self.stdout.write(f"  Max Days Between Training: {config.max_days_between_training}")
        
        self.stdout.write("\nPerformance Thresholds:")
        self.stdout.write(f"  Min Performance Score: {config.min_performance_score}")
        self.stdout.write(f"  Max Performance Drop: {config.max_performance_drop}")
        
        self.stdout.write("\nDrift Detection:")
        self.stdout.write(f"  Enabled: {config.enable_drift_detection}")
        self.stdout.write(f"  Threshold: {config.drift_threshold}")
        
        self.stdout.write("\nResource Limits:")
        self.stdout.write(f"  Max Concurrent Trainings: {config.max_concurrent_trainings}")
        self.stdout.write(f"  Training Timeout: {config.training_timeout_minutes} minutes")
        
        self.stdout.write("\nModels:")
        if config.models_to_train:
            self.stdout.write(f"  Include: {', '.join(config.models_to_train)}")
        else:
            self.stdout.write("  Include: ALL")
        if config.models_to_exclude:
            self.stdout.write(f"  Exclude: {', '.join(config.models_to_exclude)}")
        
        self.stdout.write("\n" + "=" * 50 + "\n")
    
    def _show_schedule(self):
        """Display Celery Beat schedule for ML tasks."""
        from ml.tasks import get_celery_beat_schedule
        
        schedule = get_celery_beat_schedule()
        
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write("ML CELERY BEAT SCHEDULE")
        self.stdout.write("=" * 50 + "\n")
        
        for task_name, task_config in schedule.items():
            self.stdout.write(f"\n{task_name}:")
            self.stdout.write(f"  Task: {task_config['task']}")
            schedule_value = task_config['schedule']
            
            if isinstance(schedule_value, int):
                self.stdout.write(f"  Interval: Every {schedule_value} seconds")
            else:
                self.stdout.write(f"  Schedule: {schedule_value}")
        
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write("\nAdd to your Celery settings:")
        self.stdout.write("  from ml.tasks import get_celery_beat_schedule")
        self.stdout.write("  CELERY_BEAT_SCHEDULE.update(get_celery_beat_schedule())")
        self.stdout.write("\n")
    
    def _show_history(self, limit: int):
        """Display training history."""
        try:
            import redis
            
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            r = redis.from_url(redis_url)
            
            entries = r.lrange('ml:training_log', 0, limit - 1)
            
            self.stdout.write("\n" + "=" * 60)
            self.stdout.write("TRAINING HISTORY")
            self.stdout.write("=" * 60 + "\n")
            
            if not entries:
                self.stdout.write("No training history found.\n")
                return
            
            for entry in entries:
                try:
                    data = json.loads(entry)
                    timestamp = data.get('timestamp', 'Unknown')
                    model = data.get('model', 'Unknown')
                    trigger = data.get('trigger', 'Unknown')
                    reason = data.get('reason', '')
                    
                    self.stdout.write(f"\n{timestamp}")
                    self.stdout.write(f"  Model: {model}")
                    self.stdout.write(f"  Trigger: {trigger}")
                    if reason:
                        self.stdout.write(f"  Reason: {reason}")
                except json.JSONDecodeError:
                    continue
            
            self.stdout.write("\n" + "=" * 60 + "\n")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error fetching history: {e}"))
    
    def _toggle_auto_training(self, enable: bool):
        """Enable or disable auto-training."""
        import redis
        
        try:
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            r = redis.from_url(redis_url)
            
            if enable:
                r.delete('ml:auto_training_disabled')
                self.stdout.write(self.style.SUCCESS("Auto-training ENABLED"))
            else:
                r.set('ml:auto_training_disabled', '1')
                self.stdout.write(self.style.WARNING("Auto-training DISABLED"))
            
            self.stdout.write("\nNote: This is a runtime override. For permanent changes,")
            self.stdout.write("update ML_AUTO_TRAINING_CONFIG in your settings.")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {e}"))
    
    def _check_training_needed(self, manager):
        """Check which models need training."""
        self.stdout.write("\nChecking training requirements...\n")
        
        decisions = manager.check_training_needed()
        
        if not decisions:
            self.stdout.write(self.style.SUCCESS("No models need training at this time."))
            return
        
        self.stdout.write(f"Found {len(decisions)} model(s) that need training:\n")
        
        for decision in decisions:
            trigger = decision.trigger.value if decision.trigger else 'unknown'
            
            if decision.priority >= 80:
                style = self.style.ERROR
                priority_label = "HIGH"
            elif decision.priority >= 50:
                style = self.style.WARNING
                priority_label = "MEDIUM"
            else:
                style = self.style.NOTICE
                priority_label = "LOW"
            
            self.stdout.write(style(f"\n  {decision.model_name}"))
            self.stdout.write(f"    Trigger: {trigger}")
            self.stdout.write(f"    Priority: {priority_label} ({decision.priority})")
            self.stdout.write(f"    Reason: {decision.reason}")
    
    def _force_train_model(self, model_name: str, async_mode: bool):
        """Force training for a specific model."""
        valid_models = [
            'ncf', 'deepfm', 'two_tower', 'sequence',
            'embeddings', 'demand', 'fraud', 'churn', 'search', 'image'
        ]
        
        if model_name not in valid_models:
            raise CommandError(
                f"Invalid model: {model_name}. "
                f"Valid options: {', '.join(valid_models)}"
            )
        
        self.stdout.write(f"\nForce training model: {model_name}")
        
        if async_mode:
            from ml.tasks import train_model_task
            
            task = train_model_task.delay(model_name, {}, 'manual')
            self.stdout.write(self.style.SUCCESS(f"Training queued. Task ID: {task.id}"))
        else:
            from ml.services import MLService
            
            self.stdout.write("Training synchronously...")
            service = MLService()
            result = service.train_model(model_name)
            self.stdout.write(self.style.SUCCESS(f"Training completed: {result}"))
    
    def _force_train_all(self, async_mode: bool):
        """Force training for all models."""
        self.stdout.write("\nForce training ALL models...")
        
        if async_mode:
            from ml.tasks import train_all_models_task
            
            result = train_all_models_task.delay({}, 'manual')
            self.stdout.write(self.style.SUCCESS(f"All models queued for training."))
        else:
            from ml.services import MLService
            
            models = [
                'embeddings', 'ncf', 'deepfm', 'two_tower', 'sequence',
                'demand', 'fraud', 'churn', 'search', 'image'
            ]
            
            service = MLService()
            
            for model in models:
                self.stdout.write(f"  Training {model}...")
                try:
                    service.train_model(model)
                    self.stdout.write(self.style.SUCCESS(f"    ✓ {model} completed"))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"    ✗ {model} failed: {e}"))
            
            self.stdout.write(self.style.SUCCESS("\nAll training completed!"))
    
    def _run_auto_training(self, manager, async_mode: bool):
        """Run the auto-training check and trigger training if needed."""
        is_production = getattr(settings, 'PRODUCTION', False)
        
        if not is_production:
            self.stdout.write(self.style.WARNING(
                "\nWarning: Not in production mode. "
                "Auto-training will check but not trigger.\n"
                "Set PRODUCTION = True in settings or use --force to override.\n"
            ))
        
        self.stdout.write("Running auto-training check...\n")
        
        result = manager.run_auto_training_check()
        
        status = result.get('status', 'unknown')
        
        if status == 'disabled':
            self.stdout.write(self.style.WARNING("Auto-training is disabled."))
        elif status == 'skipped':
            self.stdout.write(self.style.WARNING(
                f"Auto-training skipped: {result.get('reason', 'unknown')}"
            ))
        elif status == 'ok':
            self.stdout.write(self.style.SUCCESS(
                f"No training needed. Checked {result.get('models_checked', 0)} models."
            ))
        elif status == 'training_triggered':
            triggered = result.get('triggered', [])
            self.stdout.write(self.style.SUCCESS(
                f"Training triggered for {len(triggered)} model(s)!"
            ))
            
            for decision in result.get('decisions', []):
                model = decision.get('model_name', 'Unknown')
                trigger = decision.get('trigger', 'unknown')
                self.stdout.write(f"  - {model}: {trigger}")
        elif status == 'error':
            self.stdout.write(self.style.ERROR(
                f"Auto-training check failed: {result.get('error', 'Unknown error')}"
            ))
