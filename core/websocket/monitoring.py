"""
WebSocket Monitoring & Utilities - Production Grade

Provides:
- Connection metrics and monitoring
- Health checks
- Debugging tools
- Performance tracking
"""
import logging
import time
from typing import Dict, Set, Optional
from django.conf import settings
from django.core.cache import caches
from django.utils import timezone
from datetime import timedelta

logger = logging.getLogger('bunoraa.websocket')
realtime_cache = caches[getattr(settings, 'REALTIME_CACHE_ALIAS', 'default')]


class WebSocketMetrics:
    """Track WebSocket connection metrics."""
    
    # Cache keys
    ACTIVE_CONNECTIONS_KEY = "ws_metrics:active_connections"
    TOTAL_CONNECTIONS_KEY = "ws_metrics:total_connections"
    ACTIVE_USERS_KEY = "ws_metrics:active_users"
    CONSUMER_STATS_KEY = "ws_metrics:consumer_stats:{consumer_name}"
    CONNECTION_HISTORY_KEY = "ws_metrics:connection_history"
    
    @staticmethod
    def increment_active_connections(consumer_name: str, user_id: Optional[str] = None):
        """Increment active connection counter."""
        try:
            # Global counter
            current = realtime_cache.get(WebSocketMetrics.ACTIVE_CONNECTIONS_KEY, 0)
            realtime_cache.set(WebSocketMetrics.ACTIVE_CONNECTIONS_KEY, current + 1, 3600)
            
            # Total counter
            total = realtime_cache.get(WebSocketMetrics.TOTAL_CONNECTIONS_KEY, 0)
            realtime_cache.set(WebSocketMetrics.TOTAL_CONNECTIONS_KEY, total + 1, None)  # Permanent
            
            # Per-consumer stats
            stats_key = WebSocketMetrics.CONSUMER_STATS_KEY.format(consumer_name=consumer_name)
            stats = realtime_cache.get(stats_key, {'connections': 0})
            stats['connections'] = stats.get('connections', 0) + 1
            stats['last_update'] = time.time()
            realtime_cache.set(stats_key, stats, 3600)
            
            # Track active users
            if user_id:
                active_users = realtime_cache.get(WebSocketMetrics.ACTIVE_USERS_KEY, {})
                active_users[str(user_id)] = time.time()
                realtime_cache.set(WebSocketMetrics.ACTIVE_USERS_KEY, active_users, 3600)
            
            logger.debug(
                f"[Metrics] Connection + | Consumer: {consumer_name} | "
                f"Active: {current + 1} | Total: {total + 1}"
            )
        except Exception as e:
            logger.error(f"[Metrics] Error incrementing connections: {e}")
    
    @staticmethod
    def decrement_active_connections(consumer_name: str, user_id: Optional[str] = None):
        """Decrement active connection counter."""
        try:
            current = realtime_cache.get(WebSocketMetrics.ACTIVE_CONNECTIONS_KEY, 1)
            new_count = max(0, current - 1)
            realtime_cache.set(WebSocketMetrics.ACTIVE_CONNECTIONS_KEY, new_count, 3600)
            
            # Update per-consumer stats
            stats_key = WebSocketMetrics.CONSUMER_STATS_KEY.format(consumer_name=consumer_name)
            stats = realtime_cache.get(stats_key, {'connections': 0})
            stats['connections'] = max(0, stats.get('connections', 1) - 1)
            stats['last_update'] = time.time()
            realtime_cache.set(stats_key, stats, 3600)
            
            logger.debug(
                f"[Metrics] Connection - | Consumer: {consumer_name} | "
                f"Active: {new_count}"
            )
        except Exception as e:
            logger.error(f"[Metrics] Error decrementing connections: {e}")
    
    @staticmethod
    def get_metrics() -> Dict:
        """Get current WebSocket metrics."""
        try:
            active = realtime_cache.get(WebSocketMetrics.ACTIVE_CONNECTIONS_KEY, 0)
            total = realtime_cache.get(WebSocketMetrics.TOTAL_CONNECTIONS_KEY, 0)
            active_users = realtime_cache.get(WebSocketMetrics.ACTIVE_USERS_KEY, {})
            
            return {
                'active_connections': active,
                'total_connections_lifetime': total,
                'active_users': len(active_users),
                'timestamp': time.time(),
            }
        except Exception as e:
            logger.error(f"[Metrics] Error getting metrics: {e}")
            return {
                'error': str(e),
                'timestamp': time.time(),
            }
    
    @staticmethod
    def get_consumer_metrics(consumer_name: str) -> Dict:
        """Get metrics for a specific consumer."""
        try:
            stats_key = WebSocketMetrics.CONSUMER_STATS_KEY.format(consumer_name=consumer_name)
            stats = realtime_cache.get(stats_key, {
                'connections': 0,
                'last_update': time.time(),
            })
            return stats
        except Exception as e:
            logger.error(f"[Metrics] Error getting consumer metrics: {e}")
            return {}


class WebSocketHealthCheck:
    """Health check utilities for WebSocket system."""
    
    @staticmethod
    def check_health() -> Dict:
        """Comprehensive WebSocket health check."""
        checks = {
            'timestamp': time.time(),
            'status': 'healthy',
            'metrics': WebSocketMetrics.get_metrics(),
            'issues': [],
        }
        
        # Check active connections
        active = checks['metrics'].get('active_connections', 0)
        if active > 1000:
            checks['issues'].append({
                'severity': 'warning',
                'message': f'High active connections: {active}',
            })
        
        # Check metrics retrieval
        if 'error' in checks['metrics']:
            checks['issues'].append({
                'severity': 'error',
                'message': f'Metrics error: {checks["metrics"]["error"]}',
            })
            checks['status'] = 'degraded'
        
        # Determine overall status
        if any(issue['severity'] == 'error' for issue in checks['issues']):
            checks['status'] = 'unhealthy'
        elif any(issue['severity'] == 'warning' for issue in checks['issues']):
            checks['status'] = 'degraded'
        
        return checks
    
    @staticmethod
    def log_health():
        """Log health check results."""
        health = WebSocketHealthCheck.check_health()
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'unhealthy': '❌',
        }
        emoji = status_emoji.get(health['status'], '❓')
        
        logger.info(
            f"{emoji} WebSocket Health: {health['status']} | "
            f"Active: {health['metrics'].get('active_connections')} | "
            f"Users: {health['metrics'].get('active_users')}"
        )
        
        for issue in health['issues']:
            logger.warning(
                f"[{issue['severity'].upper()}] {issue['message']}"
            )


class ConnectionRecovery:
    """Handle connection recovery and resilience."""
    
    RECOVERY_CACHE_KEY = "ws_recovery:{connection_id}"
    
    @staticmethod
    def store_connection_state(connection_id: str, state: Dict, ttl: int = 3600):
        """Store connection state for recovery."""
        try:
            key = ConnectionRecovery.RECOVERY_CACHE_KEY.format(connection_id=connection_id)
            realtime_cache.set(key, state, ttl)
            logger.debug(f"[Recovery] Stored state for {connection_id}")
        except Exception as e:
            logger.error(f"[Recovery] Error storing state: {e}")
    
    @staticmethod
    def retrieve_connection_state(connection_id: str) -> Optional[Dict]:
        """Retrieve stored connection state."""
        try:
            key = ConnectionRecovery.RECOVERY_CACHE_KEY.format(connection_id=connection_id)
            state = realtime_cache.get(key)
            if state:
                logger.debug(f"[Recovery] Retrieved state for {connection_id}")
            return state
        except Exception as e:
            logger.error(f"[Recovery] Error retrieving state: {e}")
            return None
    
    @staticmethod
    def clear_connection_state(connection_id: str):
        """Clear stored connection state."""
        try:
            key = ConnectionRecovery.RECOVERY_CACHE_KEY.format(connection_id=connection_id)
            realtime_cache.delete(key)
        except Exception as e:
            logger.error(f"[Recovery] Error clearing state: {e}")


class WebSocketDebugger:
    """Debug utilities for WebSocket development."""
    
    @staticmethod
    def format_message(message_type: str, data: Dict) -> str:
        """Format message for debug logging."""
        import json
        try:
            return f"[{message_type}] {json.dumps(data, indent=2)}"
        except Exception:
            return f"[{message_type}] {data}"
    
    @staticmethod
    def log_message_in(consumer_name: str, message_type: str, data: Dict, user_id: Optional[str] = None):
        """Log incoming message."""
        if logger.isEnabledFor(logging.DEBUG):
            user_str = f" | User: {user_id}" if user_id else " | Anonymous"
            logger.debug(
                f"[{consumer_name}] IN{user_str} | "
                f"{WebSocketDebugger.format_message(message_type, data)}"
            )
    
    @staticmethod
    def log_message_out(consumer_name: str, message_type: str, data: Dict, user_id: Optional[str] = None):
        """Log outgoing message."""
        if logger.isEnabledFor(logging.DEBUG):
            user_str = f" | User: {user_id}" if user_id else " | Broadcast"
            logger.debug(
                f"[{consumer_name}] OUT{user_str} | "
                f"{WebSocketDebugger.format_message(message_type, data)}"
            )
    
    @staticmethod
    def log_error(consumer_name: str, error: Exception, context: str = ""):
        """Log WebSocket error with context."""
        logger.error(
            f"[{consumer_name}] Error{' - ' + context if context else ''}: {str(error)}",
            exc_info=True
        )
