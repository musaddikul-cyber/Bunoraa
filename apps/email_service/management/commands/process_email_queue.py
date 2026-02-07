"""
Process Email Queue Command
============================

Process the email queue and send pending emails.

Usage:
    python manage.py process_email_queue --batch-size 100
"""

from django.core.management.base import BaseCommand

from apps.email_service.engine import QueueManager


class Command(BaseCommand):
    help = 'Process the email queue and send pending emails'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Number of emails to process per batch'
        )
        parser.add_argument(
            '--continuous',
            action='store_true',
            help='Run continuously (use Ctrl+C to stop)'
        )
        parser.add_argument(
            '--interval',
            type=int,
            default=5,
            help='Seconds between batches in continuous mode'
        )
    
    def handle(self, *args, **options):
        batch_size = options['batch_size']
        continuous = options['continuous']
        interval = options['interval']
        
        self.stdout.write(f'Processing email queue (batch size: {batch_size})...')
        
        if continuous:
            self.stdout.write('Running in continuous mode. Press Ctrl+C to stop.')
            import time
            try:
                while True:
                    processed = QueueManager.process_queue(batch_size=batch_size)
                    if processed > 0:
                        self.stdout.write(f'Processed {processed} emails')
                    time.sleep(interval)
            except KeyboardInterrupt:
                self.stdout.write('\nStopping queue processor...')
        else:
            processed = QueueManager.process_queue(batch_size=batch_size)
            self.stdout.write(self.style.SUCCESS(f'\n✅ Processed {processed} emails'))
