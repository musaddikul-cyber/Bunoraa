"""
Celery tasks for monitoring and maintenance.
"""

import logging
from datetime import datetime, timedelta
from celery import shared_task
from django.conf import settings
from django.core.mail import mail_admins
from core.monitoring import MetricsCollector, SystemMetrics

logger = logging.getLogger('bunoraa.tasks.monitoring')


@shared_task
def send_health_report():
    """
    Send daily health report to admins.
    Includes system metrics and error summaries.
    """
    try:
        # Collect system metrics
        system_metrics = MetricsCollector.collect_system_metrics()
        
        # Get performance summary
        performance = MetricsCollector.get_performance_summary(minutes=1440)  # Last 24 hours
        
        # Build report
        report_lines = [
            "Bunoraa Daily Health Report",
            "=" * 40,
            f"Timestamp: {datetime.now().isoformat()}",
            f"Environment: {getattr(settings, 'ENVIRONMENT', 'unknown')}",
            "",
            "System Metrics:",
            f"  CPU Usage: {system_metrics.cpu_percent:.1f}%",
            f"  Memory Used: {system_metrics.memory_used_mb:.0f} MB",
            f"  Memory %: {system_metrics.memory_percent:.1f}%",
            f"  Disk Usage: {system_metrics.disk_usage_percent:.1f}%",
            "",
            "Performance (Last 24h):",
        ]
        
        if performance:
            report_lines.extend([
                f"  Avg Response Time: {performance.get('avg_duration_ms', 0):.2f}ms",
                f"  P95 Response Time: {performance.get('p95_duration_ms', 0):.2f}ms",
                f"  Total Requests: {performance.get('total_requests', 0)}",
                f"  Error Rate: {performance.get('error_rate', 0):.2f}%",
            ])
        
        report_text = "\n".join(report_lines)
        
        # Send to admins
        mail_admins(
            subject=f"Bunoraa Health Report - {datetime.now().strftime('%Y-%m-%d')}",
            message=report_text
        )
        
        logger.info("Health report sent to admins")
        return {'status': 'success'}
        
    except Exception as exc:
        logger.exception("Failed to send health report")
        return {'status': 'error', 'message': str(exc)}


@shared_task
def cleanup_old_logs():
    """
    Clean up old log files to prevent disk space issues.
    Keeps logs for 30 days by default.
    """
    try:
        log_dir = Path(settings.BASE_DIR) / 'logs'
        if not log_dir.exists():
            return {'status': 'skipped', 'reason': 'log_dir_not_found'}
        
        retention_days = getattr(settings, 'LOG_RETENTION_DAYS', 30)
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted = 0
        
        for log_file in log_dir.glob('*.log*'):
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff:
                try:
                    log_file.unlink()
                    deleted += 1
                except OSError:
                    pass
        
        logger.info(f"Cleaned up {deleted} old log files")
        return {'status': 'success', 'deleted_count': deleted}
        
    except Exception as exc:
        logger.exception("Log cleanup failed")
        return {'status': 'error', 'message': str(exc)}
