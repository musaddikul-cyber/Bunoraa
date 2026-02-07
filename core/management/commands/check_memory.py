"""
Management command to check system and Django process memory usage.

Usage:
  python manage.py check_memory                    # Show memory stats
  python manage.py check_memory --threshold=400    # Warn if over 400MB
  python manage.py check_memory --watch            # Monitor memory every 5 seconds
"""

import os
import sys
import time
import psutil
from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings


class Command(BaseCommand):
    help = 'Check memory usage of current Django process and system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--threshold',
            type=int,
            default=400,
            help='Warning threshold in MB (default: 400)',
        )
        parser.add_argument(
            '--watch',
            action='store_true',
            help='Monitor memory continuously every 5 seconds',
        )
        parser.add_argument(
            '--duration',
            type=int,
            default=60,
            help='Duration to monitor in seconds (with --watch, default: 60)',
        )

    def handle(self, *args, **options):
        threshold = options['threshold']
        watch = options['watch']
        duration = options['duration']

        if watch:
            self.watch_memory(threshold, duration)
        else:
            self.check_memory(threshold)

    def get_memory_info(self):
        """Get memory information for current process and system."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            # Get virtual memory
            vm = psutil.virtual_memory()
            
            return {
                'process_rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'process_vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'system_total_mb': vm.total / 1024 / 1024,
                'system_available_mb': vm.available / 1024 / 1024,
                'system_used_mb': vm.used / 1024 / 1024,
                'system_percent': vm.percent,
                'process_percent': process.memory_percent(),
            }
        except Exception as e:
            raise CommandError(f'Failed to get memory info: {e}')

    def check_memory(self, threshold):
        """Check memory usage once."""
        info = self.get_memory_info()
        
        self.stdout.write('\n' + '=' * 60)
        self.stdout.write('MEMORY USAGE REPORT')
        self.stdout.write('=' * 60)
        self.stdout.write(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.stdout.write(f'Process ID: {os.getpid()}')
        
        self.stdout.write('\n📊 DJANGO PROCESS MEMORY:')
        self.stdout.write(f'  RSS (Physical): {info["process_rss_mb"]:.1f} MB')
        self.stdout.write(f'  VMS (Virtual):  {info["process_vms_mb"]:.1f} MB')
        self.stdout.write(f'  % of System:    {info["process_percent"]:.1f}%')
        
        self.stdout.write('\n🖥️  SYSTEM MEMORY:')
        self.stdout.write(f'  Total:         {info["system_total_mb"]:.1f} MB')
        self.stdout.write(f'  Used:          {info["system_used_mb"]:.1f} MB ({info["system_percent"]:.1f}%)')
        self.stdout.write(f'  Available:     {info["system_available_mb"]:.1f} MB')
        
        # Check threshold
        self.stdout.write('\n⚠️  THRESHOLD ANALYSIS:')
        process_rss = info['process_rss_mb']
        
        if process_rss > threshold:
            self.stdout.write(
                self.style.ERROR(
                    f'  ❌ CRITICAL: Process using {process_rss:.1f} MB (threshold: {threshold} MB)'
                )
            )
            self.stdout.write(self.style.ERROR('  Consider: Restarting workers, disabling features, or upgrading plan'))
        elif process_rss > threshold * 0.8:
            self.stdout.write(
                self.style.WARNING(
                    f'  ⚠️  WARNING: Process using {process_rss:.1f} MB (threshold: {threshold} MB)'
                )
            )
            self.stdout.write(self.style.WARNING('  Monitor closely for potential issues'))
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f'  ✅ OK: Process using {process_rss:.1f} MB (threshold: {threshold} MB)'
                )
            )
        
        # Render tier info
        if settings.ENVIRONMENT == 'production':
            self.stdout.write('\n🚀 RENDER TIER INFORMATION:')
            self.stdout.write('  Free Tier Limit:        512 MB')
            self.stdout.write(f'  Current Usage:          {process_rss:.1f} MB')
            remaining = 512 - process_rss
            if remaining > 0:
                self.stdout.write(
                    self.style.SUCCESS(f'  Available Headroom:     {remaining:.1f} MB')
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f'  Available Headroom:     {remaining:.1f} MB (OVER LIMIT!)')
                )
        
        self.stdout.write('=' * 60 + '\n')

    def watch_memory(self, threshold, duration):
        """Monitor memory usage continuously."""
        self.stdout.write(f'\n📈 Monitoring memory for {duration} seconds...\n')
        self.stdout.write('Time                 | Process (MB) | System (%) | Status')
        self.stdout.write('-' * 70)
        
        start_time = time.time()
        max_process = 0
        measurements = []
        
        try:
            while time.time() - start_time < duration:
                info = self.get_memory_info()
                process_mb = info['process_rss_mb']
                system_pct = info['system_percent']
                
                max_process = max(max_process, process_mb)
                measurements.append(process_mb)
                
                # Status indicator
                if process_mb > threshold:
                    status = self.style.ERROR('❌ CRITICAL')
                elif process_mb > threshold * 0.8:
                    status = self.style.WARNING('⚠️  WARNING')
                else:
                    status = self.style.SUCCESS('✅ OK')
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.stdout.write(f'{timestamp} | {process_mb:>11.1f} | {system_pct:>9.1f} | {status}')
                
                time.sleep(5)
        except KeyboardInterrupt:
            self.stdout.write('\n\nMonitoring stopped by user.')
        
        # Summary
        if measurements:
            avg_process = sum(measurements) / len(measurements)
            min_process = min(measurements)
            
            self.stdout.write('-' * 70)
            self.stdout.write(f'\n📊 MONITORING SUMMARY ({len(measurements)} measurements):')
            self.stdout.write(f'  Minimum:  {min_process:.1f} MB')
            self.stdout.write(f'  Average:  {avg_process:.1f} MB')
            self.stdout.write(f'  Maximum:  {max_process:.1f} MB')
            self.stdout.write(f'  Threshold: {threshold} MB\n')
