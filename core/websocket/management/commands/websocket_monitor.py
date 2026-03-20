"""
Django management command for WebSocket health checks and monitoring
"""
from django.core.management.base import BaseCommand
from core.websocket.monitoring import WebSocketHealthCheck, WebSocketMetrics
import json


class Command(BaseCommand):
    help = 'Monitor and check WebSocket system health'

    def add_arguments(self, parser):
        parser.add_argument(
            '--check',
            action='store_true',
            help='Run health check',
        )
        parser.add_argument(
            '--metrics',
            action='store_true',
            help='Show current metrics',
        )
        parser.add_argument(
            '--watch',
            action='store_true',
            help='Watch metrics in real-time',
        )
        parser.add_argument(
            '--interval',
            type=int,
            default=5,
            help='Update interval in seconds (for --watch)',
        )

    def handle(self, *args, **options):
        if options['check']:
            self.check_health()
        elif options['metrics']:
            self.show_metrics()
        elif options['watch']:
            self.watch_metrics(options['interval'])
        else:
            self.check_health()
    
    def check_health(self):
        """Run and display health check results."""
        self.stdout.write('\n' + '='*60)
        self.stdout.write('WebSocket System Health Check')
        self.stdout.write('='*60 + '\n')
        
        health = WebSocketHealthCheck.check_health()
        
        # Status line
        status_colors = {
            'healthy': self.style.SUCCESS,
            'degraded': self.style.WARNING,
            'unhealthy': self.style.ERROR,
        }
        status_style = status_colors.get(health['status'], self.style.WARNING)
        self.stdout.write(
            f"Status: {status_style(health['status'].upper())}",
            ending='\n'
        )
        
        # Metrics
        metrics = health['metrics']
        self.stdout.write(f"\nMetrics:")
        self.stdout.write(f"  • Active Connections: {metrics.get('active_connections', 0)}")
        self.stdout.write(f"  • Active Users: {metrics.get('active_users', 0)}")
        self.stdout.write(f"  • Lifetime Connections: {metrics.get('total_connections_lifetime', 0)}")
        
        # Issues
        if health['issues']:
            self.stdout.write(f"\n{self.style.WARNING('Issues:')}")
            for issue in health['issues']:
                severity = issue['severity'].upper()
                message = issue['message']
                if issue['severity'] == 'error':
                    self.stdout.write(f"  ❌ [{severity}] {message}")
                else:
                    self.stdout.write(f"  ⚠️  [{severity}] {message}")
        else:
            self.stdout.write(f"\n{self.style.SUCCESS('✓ No issues detected')}")
        
        self.stdout.write('\n' + '='*60 + '\n')
    
    def show_metrics(self):
        """Display current metrics."""
        self.stdout.write('\n' + '='*60)
        self.stdout.write('WebSocket Metrics')
        self.stdout.write('='*60 + '\n')
        
        metrics = WebSocketMetrics.get_metrics()
        
        self.stdout.write(f"Active Connections: {metrics.get('active_connections', 0)}")
        self.stdout.write(f"Active Users: {metrics.get('active_users', 0)}")
        self.stdout.write(f"Lifetime Connections: {metrics.get('total_connections_lifetime', 0)}")
        
        self.stdout.write('\n' + '='*60 + '\n')
    
    def watch_metrics(self, interval: int):
        """Watch metrics in real-time."""
        import time
        import os
        
        self.stdout.write(
            f"\nWatching WebSocket metrics (update every {interval}s)... "
            f"Press Ctrl+C to stop\n"
        )
        
        try:
            while True:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Display header
                self.stdout.write('='*60)
                self.stdout.write('WebSocket Metrics (Live)')
                self.stdout.write('='*60)
                
                # Get metrics
                metrics = WebSocketMetrics.get_metrics()
                health = WebSocketHealthCheck.check_health()
                
                # Display metrics
                status_colors = {
                    'healthy': self.style.SUCCESS,
                    'degraded': self.style.WARNING,
                    'unhealthy': self.style.ERROR,
                }
                status_style = status_colors.get(health['status'], self.style.WARNING)
                
                self.stdout.write(f"\nStatus: {status_style(health['status'].upper())}")
                self.stdout.write(f"Active Connections:    {metrics.get('active_connections', 0)}")
                self.stdout.write(f"Active Users:          {metrics.get('active_users', 0)}")
                self.stdout.write(f"Lifetime Connections:  {metrics.get('total_connections_lifetime', 0)}")
                
                # Display issues
                if health['issues']:
                    self.stdout.write(f"\n{self.style.WARNING('Issues:')}")
                    for issue in health['issues']:
                        self.stdout.write(f"  • [{issue['severity'].upper()}] {issue['message']}")
                
                self.stdout.write('\nPress Ctrl+C to stop...\n')
                
                # Wait for next update
                time.sleep(interval)
        
        except KeyboardInterrupt:
            self.stdout.write('\n\nMonitoring stopped.')
