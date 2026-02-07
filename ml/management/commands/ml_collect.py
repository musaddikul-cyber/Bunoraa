"""
ML Collect Data Command

Django management command to collect and process ML training data.

Usage:
    python manage.py ml_collect                   # Process all queued data
    python manage.py ml_collect --profiles       # Collect user profiles
    python manage.py ml_collect --products       # Collect product features
    python manage.py ml_collect --export         # Export training data
"""

import logging
import os
from datetime import datetime

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger("bunoraa.ml.commands")


class Command(BaseCommand):
    help = "Collect and process ML training data"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--process',
            action='store_true',
            help='Process queued interaction data',
        )
        parser.add_argument(
            '--profiles',
            action='store_true',
            help='Collect user behavior profiles',
        )
        parser.add_argument(
            '--products',
            action='store_true',
            help='Collect product ML features',
        )
        parser.add_argument(
            '--export',
            action='store_true',
            help='Export training data to files',
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default=None,
            help='Output directory for exports (default: ml/training_data)',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1000,
            help='Batch size for processing (default: 1000)',
        )
        parser.add_argument(
            '--days',
            type=int,
            default=90,
            help='Number of days of data to include (default: 90)',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            dest='run_all',
            help='Run all collection tasks',
        )
    
    def handle(self, *args, **options):
        run_all = options['run_all']
        batch_size = options['batch_size']
        days = options['days']
        output_dir = options['output_dir'] or str(settings.ML_TRAINING_DATA_DIR)
        
        # If no specific option, run all
        if not any([
            options['process'], options['profiles'],
            options['products'], options['export'], run_all
        ]):
            run_all = True
        
        start_time = timezone.now()
        
        try:
            if run_all or options['process']:
                self._process_queued_data(batch_size)
            
            if run_all or options['profiles']:
                self._collect_profiles(batch_size)
            
            if run_all or options['products']:
                self._collect_products(batch_size)
            
            if run_all or options['export']:
                self._export_data(output_dir, days)
            
            elapsed = (timezone.now() - start_time).total_seconds()
            self.stdout.write(
                self.style.SUCCESS(f"\nData collection completed in {elapsed:.1f}s")
            )
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Collection failed: {e}"))
            logger.exception("Collection error")
            raise CommandError(str(e))
    
    def _process_queued_data(self, batch_size: int):
        """Process queued interaction data from Redis."""
        self.stdout.write("Processing queued data...")
        
        from ml.data_collection.collector import DataCollector
        from ml.data_collection.events import EventTracker
        
        collector = DataCollector()
        tracker = EventTracker()
        
        # Get queue stats before
        stats = collector.get_queue_stats()
        self.stdout.write(f"  Queue sizes: {stats}")
        
        # Process queues
        results = collector.process_queued_data(batch_size)
        events = tracker.process_event_queue(batch_size)
        
        self.stdout.write(f"  Processed interactions: {results.get('interactions', 0)}")
        self.stdout.write(f"  Processed product interactions: {results.get('product_interactions', 0)}")
        self.stdout.write(f"  Processed conversions: {results.get('conversions', 0)}")
        self.stdout.write(f"  Processed events: {events}")
    
    def _collect_profiles(self, batch_size: int):
        """Collect user behavior profiles."""
        self.stdout.write("Collecting user profiles...")
        
        from ml.data_collection.user_profile import UserProfileCollector
        
        collector = UserProfileCollector()
        count = collector.collect_all_profiles(batch_size)
        
        self.stdout.write(f"  Collected {count} user profiles")
    
    def _collect_products(self, batch_size: int):
        """Collect product ML features."""
        self.stdout.write("Collecting product features...")
        
        from ml.data_collection.product_analytics import ProductAnalyticsCollector
        
        collector = ProductAnalyticsCollector()
        count = collector.collect_all_features(batch_size)
        
        self.stdout.write(f"  Collected features for {count} products")
    
    def _export_data(self, output_dir: str, days: int):
        """Export training data to files."""
        self.stdout.write(f"Exporting data to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export product features
        self._export_product_features(output_dir, timestamp)
        
        # Export user interactions
        self._export_interactions(output_dir, timestamp, days)
        
        # Export user profiles
        self._export_profiles(output_dir, timestamp)
        
        # Export co-occurrence matrix
        self._export_cooccurrence(output_dir, timestamp, days)
    
    def _export_product_features(self, output_dir: str, timestamp: str):
        """Export product features as CSV."""
        from ml.data_collection.product_analytics import ProductAnalyticsCollector
        
        collector = ProductAnalyticsCollector()
        path = os.path.join(output_dir, f'product_features_{timestamp}.csv')
        count = collector.export_training_data(path)
        
        self.stdout.write(f"  Exported {count} product features to {path}")
    
    def _export_interactions(self, output_dir: str, timestamp: str, days: int):
        """Export user-product interactions."""
        import csv
        from datetime import timedelta
        from apps.analytics.models import ProductView
        from django.db.models import Count
        
        path = os.path.join(output_dir, f'interactions_{timestamp}.csv')
        cutoff = timezone.now() - timedelta(days=days)
        
        # Get interactions
        interactions = ProductView.objects.filter(
            created_at__gte=cutoff,
            user_id__isnull=False
        ).values('user_id', 'product_id').annotate(
            count=Count('id')
        )
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['user_id', 'product_id', 'count'])
            writer.writeheader()
            
            for row in interactions:
                writer.writerow(row)
        
        self.stdout.write(f"  Exported interactions to {path}")
    
    def _export_profiles(self, output_dir: str, timestamp: str):
        """Export user profiles."""
        import csv
        import json
        
        path = os.path.join(output_dir, f'user_profiles_{timestamp}.csv')
        
        from ml.data_collection.user_profile import UserProfileCollector
        from apps.accounts.models import User
        from dataclasses import asdict
        
        collector = UserProfileCollector()
        
        # Get active users
        users = User.objects.filter(is_active=True).values_list('id', flat=True)[:10000]
        
        profiles = []
        for user_id in users:
            try:
                profile = collector.get_cached_profile(user_id)
                if profile:
                    profiles.append(asdict(profile))
            except Exception:
                pass
        
        if profiles:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                # Flatten nested dicts
                flat_profiles = []
                for p in profiles:
                    flat = {}
                    for k, v in p.items():
                        if isinstance(v, dict):
                            flat[k] = json.dumps(v)
                        else:
                            flat[k] = v
                    flat_profiles.append(flat)
                
                writer = csv.DictWriter(f, fieldnames=flat_profiles[0].keys())
                writer.writeheader()
                writer.writerows(flat_profiles)
        
        self.stdout.write(f"  Exported {len(profiles)} user profiles to {path}")
    
    def _export_cooccurrence(self, output_dir: str, timestamp: str, days: int):
        """Export product co-occurrence for collaborative filtering."""
        import csv
        from datetime import timedelta
        from collections import defaultdict
        from apps.orders.models import OrderItem
        
        path = os.path.join(output_dir, f'cooccurrence_{timestamp}.csv')
        cutoff = timezone.now() - timedelta(days=days)
        
        # Get orders with multiple items
        order_items = OrderItem.objects.filter(
            order__created_at__gte=cutoff,
            order__status__in=['completed', 'delivered']
        ).values('order_id', 'product_id')
        
        # Group by order
        orders = defaultdict(list)
        for item in order_items:
            orders[item['order_id']].append(item['product_id'])
        
        # Count co-occurrences
        cooccurrence = defaultdict(int)
        for order_id, products in orders.items():
            if len(products) >= 2:
                for i, p1 in enumerate(products):
                    for p2 in products[i+1:]:
                        key = tuple(sorted([p1, p2]))
                        cooccurrence[key] += 1
        
        # Write to CSV
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['product_1', 'product_2', 'count'])
            
            for (p1, p2), count in cooccurrence.items():
                if count >= 2:  # Minimum threshold
                    writer.writerow([p1, p2, count])
        
        self.stdout.write(f"  Exported co-occurrence data to {path}")
