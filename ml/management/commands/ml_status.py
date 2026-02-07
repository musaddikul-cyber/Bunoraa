"""
ML Status Command

Django management command to check ML system status.

Usage:
    python manage.py ml_status                  # Show all status
    python manage.py ml_status --health        # Health check only
    python manage.py ml_status --data          # Data collection status
"""

import logging
import os
from pathlib import Path

from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger("bunoraa.ml.commands")


class Command(BaseCommand):
    help = "Check ML system status"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--health',
            action='store_true',
            help='Show model health status only',
        )
        parser.add_argument(
            '--data',
            action='store_true',
            help='Show data collection status only',
        )
        parser.add_argument(
            '--queue',
            action='store_true',
            help='Show Redis queue status only',
        )
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output as JSON',
        )
    
    def handle(self, *args, **options):
        show_health = options['health']
        show_data = options['data']
        show_queue = options['queue']
        output_json = options['json']
        
        # If no specific option, show all
        show_all = not (show_health or show_data or show_queue)
        
        status = {
            'timestamp': timezone.now().isoformat(),
            'production': getattr(settings, 'PRODUCTION', False),
            'auto_training': getattr(settings, 'ML_AUTO_TRAINING', False),
        }
        
        if show_all or show_health:
            status['models'] = self._get_model_status()
        
        if show_all or show_data:
            status['data'] = self._get_data_status()
        
        if show_all or show_queue:
            status['queue'] = self._get_queue_status()
        
        if show_all:
            status['system'] = self._get_system_status()
        
        if output_json:
            import json
            self.stdout.write(json.dumps(status, indent=2, default=str))
        else:
            self._display_status(status)
    
    def _get_model_status(self) -> dict:
        """Get status of all ML models."""
        models_status = {}
        
        try:
            from ml.core.registry import ModelRegistry
            registry = ModelRegistry()
            
            model_names = [
                'ncf', 'deepfm', 'two_tower', 'sequence',
                'embeddings', 'demand', 'fraud', 'churn', 'search', 'image'
            ]
            
            for model_name in model_names:
                try:
                    model_info = registry.get_model_info(model_name)
                    
                    if model_info:
                        models_status[model_name] = {
                            'status': 'loaded',
                            'version': model_info.get('version', 'unknown'),
                            'last_trained': model_info.get('last_trained'),
                            'metrics': model_info.get('metrics', {}),
                        }
                    else:
                        models_status[model_name] = {
                            'status': 'not_found',
                        }
                except Exception as e:
                    models_status[model_name] = {
                        'status': 'error',
                        'error': str(e),
                    }
        except ImportError:
            models_status['error'] = 'ModelRegistry not available'
        
        return models_status
    
    def _get_data_status(self) -> dict:
        """Get data collection status."""
        data_status = {
            'interactions': 0,
            'user_profiles': 0,
            'product_features': 0,
            'training_ready': False,
            'last_collection': None,
        }
        
        try:
            from apps.catalog.models import Product
            from apps.accounts.models import UserProfile
            
            # Count products and users
            data_status['total_products'] = Product.objects.count()
            data_status['total_users'] = UserProfile.objects.count()
            
            # Check training data directory
            training_data_path = Path(settings.BASE_DIR) / 'ml' / 'training_data'
            
            if training_data_path.exists():
                csv_files = list(training_data_path.glob('*.csv'))
                parquet_files = list(training_data_path.glob('*.parquet'))
                
                data_status['training_files'] = {
                    'csv': len(csv_files),
                    'parquet': len(parquet_files),
                }
                
                # Check file sizes
                total_size = sum(f.stat().st_size for f in csv_files + parquet_files)
                data_status['total_size_mb'] = round(total_size / (1024 * 1024), 2)
                
                # Get latest modification time
                if csv_files or parquet_files:
                    all_files = csv_files + parquet_files
                    latest_file = max(all_files, key=lambda f: f.stat().st_mtime)
                    data_status['last_collection'] = timezone.datetime.fromtimestamp(
                        latest_file.stat().st_mtime
                    ).isoformat()
                
                # Check if we have enough data for training
                interactions_file = training_data_path / 'interactions.csv'
                if interactions_file.exists():
                    # Count lines (approximate interaction count)
                    with open(interactions_file, 'r') as f:
                        data_status['interactions'] = sum(1 for _ in f) - 1  # Minus header
                
                profiles_file = training_data_path / 'user_profiles.csv'
                if profiles_file.exists():
                    with open(profiles_file, 'r') as f:
                        data_status['user_profiles'] = sum(1 for _ in f) - 1
                
                products_file = training_data_path / 'product_features.csv'
                if products_file.exists():
                    with open(products_file, 'r') as f:
                        data_status['product_features'] = sum(1 for _ in f) - 1
                
                # Training ready if we have sufficient data
                data_status['training_ready'] = (
                    data_status['interactions'] >= 1000 and
                    data_status['product_features'] >= 100
                )
        except Exception as e:
            data_status['error'] = str(e)
        
        return data_status
    
    def _get_queue_status(self) -> dict:
        """Get Redis queue status."""
        queue_status = {
            'redis_available': False,
            'queues': {},
        }
        
        try:
            import redis
            
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            r = redis.from_url(redis_url)
            
            # Test connection
            r.ping()
            queue_status['redis_available'] = True
            
            # Check ML queues
            queue_names = [
                'ml:interactions',
                'ml:events',
                'ml:user_profiles',
                'ml:product_features',
                'ml:training_queue',
            ]
            
            for queue_name in queue_names:
                queue_status['queues'][queue_name] = {
                    'length': r.llen(queue_name),
                    'type': r.type(queue_name).decode() if r.exists(queue_name) else 'none',
                }
            
            # Check sets and sorted sets
            set_names = [
                'ml:active_users',
                'ml:popular_products',
            ]
            
            for set_name in set_names:
                if r.exists(set_name):
                    set_type = r.type(set_name).decode()
                    if set_type == 'set':
                        queue_status['queues'][set_name] = {
                            'length': r.scard(set_name),
                            'type': set_type,
                        }
                    elif set_type == 'zset':
                        queue_status['queues'][set_name] = {
                            'length': r.zcard(set_name),
                            'type': set_type,
                        }
            
            # Get memory usage
            info = r.info('memory')
            queue_status['memory'] = {
                'used_mb': round(info.get('used_memory', 0) / (1024 * 1024), 2),
                'peak_mb': round(info.get('used_memory_peak', 0) / (1024 * 1024), 2),
            }
            
        except Exception as e:
            queue_status['error'] = str(e)
        
        return queue_status
    
    def _get_system_status(self) -> dict:
        """Get system resource status."""
        system_status = {
            'gpu_available': False,
            'cuda_version': None,
            'pytorch_version': None,
        }
        
        try:
            import torch
            system_status['pytorch_version'] = torch.__version__
            system_status['gpu_available'] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                system_status['cuda_version'] = torch.version.cuda
                system_status['gpu_count'] = torch.cuda.device_count()
                system_status['gpu_name'] = torch.cuda.get_device_name(0)
                
                # GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                system_status['gpu_memory_gb'] = round(gpu_memory / (1024**3), 2)
        except ImportError:
            system_status['pytorch_version'] = 'not installed'
        
        # CPU and memory
        try:
            import psutil
            
            system_status['cpu_percent'] = psutil.cpu_percent()
            system_status['cpu_count'] = psutil.cpu_count()
            
            memory = psutil.virtual_memory()
            system_status['memory_total_gb'] = round(memory.total / (1024**3), 2)
            system_status['memory_available_gb'] = round(memory.available / (1024**3), 2)
            system_status['memory_percent'] = memory.percent
            
            disk = psutil.disk_usage('/')
            system_status['disk_total_gb'] = round(disk.total / (1024**3), 2)
            system_status['disk_free_gb'] = round(disk.free / (1024**3), 2)
        except ImportError:
            pass
        
        # Celery status
        try:
            from celery import current_app
            
            inspect = current_app.control.inspect()
            active = inspect.active()
            
            system_status['celery_available'] = active is not None
            system_status['celery_workers'] = len(active) if active else 0
        except Exception:
            system_status['celery_available'] = False
        
        return system_status
    
    def _display_status(self, status: dict):
        """Display status in formatted output."""
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("ML SYSTEM STATUS")
        self.stdout.write("=" * 60 + "\n")
        
        # Production status
        prod_status = "PRODUCTION" if status['production'] else "DEVELOPMENT"
        auto_train = "ENABLED" if status['auto_training'] else "DISABLED"
        
        self.stdout.write(f"Environment: {prod_status}")
        self.stdout.write(f"Auto-Training: {auto_train}")
        self.stdout.write(f"Timestamp: {status['timestamp']}\n")
        
        # Models
        if 'models' in status:
            self.stdout.write("-" * 40)
            self.stdout.write("MODEL STATUS")
            self.stdout.write("-" * 40)
            
            for model_name, model_info in status['models'].items():
                if model_name == 'error':
                    continue
                
                model_status = model_info.get('status', 'unknown')
                
                if model_status == 'loaded':
                    self.stdout.write(
                        self.style.SUCCESS(f"  ✓ {model_name}: {model_status}")
                    )
                    self.stdout.write(f"      Version: {model_info.get('version', 'unknown')}")
                elif model_status == 'not_found':
                    self.stdout.write(
                        self.style.WARNING(f"  ○ {model_name}: not trained")
                    )
                else:
                    self.stdout.write(
                        self.style.ERROR(f"  ✗ {model_name}: {model_status}")
                    )
            
            self.stdout.write("")
        
        # Data
        if 'data' in status:
            self.stdout.write("-" * 40)
            self.stdout.write("DATA STATUS")
            self.stdout.write("-" * 40)
            
            data = status['data']
            self.stdout.write(f"  Products: {data.get('total_products', 0)}")
            self.stdout.write(f"  Users: {data.get('total_users', 0)}")
            self.stdout.write(f"  Interactions: {data.get('interactions', 0)}")
            self.stdout.write(f"  User Profiles: {data.get('user_profiles', 0)}")
            self.stdout.write(f"  Product Features: {data.get('product_features', 0)}")
            
            if data.get('training_files'):
                files = data['training_files']
                self.stdout.write(f"  Training Files: {files.get('csv', 0)} CSV, {files.get('parquet', 0)} Parquet")
            
            if data.get('total_size_mb'):
                self.stdout.write(f"  Total Size: {data['total_size_mb']} MB")
            
            if data.get('training_ready'):
                self.stdout.write(self.style.SUCCESS("  ✓ Training Ready"))
            else:
                self.stdout.write(self.style.WARNING("  ○ Insufficient data for training"))
            
            self.stdout.write("")
        
        # Queue
        if 'queue' in status:
            self.stdout.write("-" * 40)
            self.stdout.write("QUEUE STATUS")
            self.stdout.write("-" * 40)
            
            queue = status['queue']
            
            if queue.get('redis_available'):
                self.stdout.write(self.style.SUCCESS("  ✓ Redis Connected"))
                
                for queue_name, queue_info in queue.get('queues', {}).items():
                    length = queue_info.get('length', 0)
                    self.stdout.write(f"    {queue_name}: {length} items")
                
                if queue.get('memory'):
                    mem = queue['memory']
                    self.stdout.write(f"  Memory: {mem.get('used_mb', 0)} MB / {mem.get('peak_mb', 0)} MB peak")
            else:
                self.stdout.write(self.style.ERROR("  ✗ Redis Not Available"))
            
            self.stdout.write("")
        
        # System
        if 'system' in status:
            self.stdout.write("-" * 40)
            self.stdout.write("SYSTEM STATUS")
            self.stdout.write("-" * 40)
            
            system = status['system']
            
            self.stdout.write(f"  PyTorch: {system.get('pytorch_version', 'N/A')}")
            
            if system.get('gpu_available'):
                self.stdout.write(self.style.SUCCESS(f"  ✓ GPU: {system.get('gpu_name', 'Unknown')}"))
                self.stdout.write(f"    CUDA: {system.get('cuda_version', 'N/A')}")
                self.stdout.write(f"    Memory: {system.get('gpu_memory_gb', 0)} GB")
            else:
                self.stdout.write(self.style.WARNING("  ○ GPU: Not available"))
            
            if 'cpu_count' in system:
                self.stdout.write(f"  CPU: {system.get('cpu_count', 0)} cores ({system.get('cpu_percent', 0)}% used)")
            
            if 'memory_total_gb' in system:
                self.stdout.write(
                    f"  Memory: {system.get('memory_available_gb', 0)} / "
                    f"{system.get('memory_total_gb', 0)} GB available"
                )
            
            if system.get('celery_available'):
                self.stdout.write(self.style.SUCCESS(f"  ✓ Celery: {system.get('celery_workers', 0)} workers"))
            else:
                self.stdout.write(self.style.WARNING("  ○ Celery: Not available"))
            
            self.stdout.write("")
        
        self.stdout.write("=" * 60 + "\n")
