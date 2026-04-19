"""
Cache Warming Management Command
=================================

Usage:
    python manage.py warm_cache --all          # Warm all caches
    python manage.py warm_cache --products     # Warm product caches
    python manage.py warm_cache --categories   # Warm category caches
    python manage.py warm_cache --stats        # Show cache statistics
"""
import time
from django.core.management.base import BaseCommand, CommandError
from django.core.cache import cache

from core.cache_manager import CacheWarmer, CacheManager
from core.monitoring import HealthChecker, MetricsCollector


class Command(BaseCommand):
    help = 'Warm caches to improve application performance'

    def add_arguments(self, parser):
        parser.add_argument(
            '--all',
            action='store_true',
            help='Warm all caches (products and categories)',
        )
        parser.add_argument(
            '--products',
            action='store_true',
            help='Warm product caches',
        )
        parser.add_argument(
            '--categories',
            action='store_true',
            help='Warm category caches',
        )
        parser.add_argument(
            '--stats',
            action='store_true',
            help='Show cache statistics',
        )
        parser.add_argument(
            '--health',
            action='store_true',
            help='Run health checks',
        )

    def handle(self, *args, **options):
        start_time = time.time()
        
        # Show stats only
        if options['stats']:
            self.show_stats()
            return
        
        # Run health checks
        if options['health']:
            self.run_health_checks()
            return
        
        # Determine what to warm
        warm_all = options['all'] or not (options['products'] or options['categories'])
        
        results = {}
        
        # Warm products
        if warm_all or options['products']:
            self.stdout.write(self.style.HTTP_INFO('Warming product caches...'))
            try:
                product_results = CacheWarmer.warm_products()
                results['products'] = product_results
                for key, count in product_results.items():
                    self.stdout.write(f"  ✓ {key}: {count} items cached")
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  ✗ Error warming products: {e}"))
        
        # Warm categories
        if warm_all or options['categories']:
            self.stdout.write(self.style.HTTP_INFO('Warming category caches...'))
            try:
                category_count = CacheWarmer.warm_categories()
                results['categories'] = category_count
                self.stdout.write(f"  ✓ {category_count} categories cached")
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  ✗ Error warming categories: {e}"))
        
        # Summary
        duration = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(f"\n✓ Cache warming completed in {duration:.2f}s"))
        
        # Show system metrics
        self.show_quick_metrics()
    
    def show_stats(self):
        """Show cache statistics."""
        self.stdout.write(self.style.HTTP_INFO('\nCache Statistics'))
        self.stdout.write('=' * 50)
        
        try:
            from django_redis import get_redis_connection
            redis_conn = get_redis_connection('default')
            
            # Memory info
            info = redis_conn.info('memory')
            used_mb = info.get('used_memory', 0) / (1024 * 1024)
            peak_mb = info.get('used_memory_peak', 0) / (1024 * 1024)
            
            self.stdout.write(f"\nRedis Memory Usage:")
            self.stdout.write(f"  Current: {used_mb:.2f} MB")
            self.stdout.write(f"  Peak: {peak_mb:.2f} MB")
            
            # Key statistics
            db_size = redis_conn.dbsize()
            self.stdout.write(f"\nTotal Keys: {db_size}")
            
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"Could not get Redis stats: {e}"))
    
    def run_health_checks(self):
        """Run and display health checks."""
        self.stdout.write(self.style.HTTP_INFO('\nHealth Checks'))
        self.stdout.write('=' * 50)
        
        results = HealthChecker.run_all()
        
        for check_name, check_result in results.get('checks', {}).items():
            status = check_result.get('status', 'unknown')
            message = check_result.get('message', 'No message')
            
            if status == 'healthy':
                self.stdout.write(self.style.SUCCESS(f"  ✓ {check_name}: {message}"))
            elif status == 'warning':
                self.stdout.write(self.style.WARNING(f"  ⚠ {check_name}: {message}"))
            else:
                self.stdout.write(self.style.ERROR(f"  ✗ {check_name}: {message}"))
    
    def show_quick_metrics(self):
        """Show quick system metrics."""
        self.stdout.write(self.style.HTTP_INFO('\nSystem Status'))
        self.stdout.write('-' * 50)
        
        try:
            metrics = MetricsCollector.collect_system_metrics()
            self.stdout.write(f"  CPU: {metrics.cpu_percent:.1f}%")
            self.stdout.write(f"  Memory: {metrics.memory_percent:.1f}% ({metrics.memory_used_mb:.1f} MB)")
            self.stdout.write(f"  Disk: {metrics.disk_usage_percent:.1f}%")
        except Exception:
            self.stdout.write(self.style.WARNING("  Metrics collection unavailable"))
