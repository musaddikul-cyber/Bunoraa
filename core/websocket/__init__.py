"""
WebSocket Module - Production Ready Infrastructure

This module provides:
- Production-grade WebSocket consumers
- Keep-alive mechanisms
- Rate limiting
- Connection monitoring
- Health checks
- Message validation

Consumers:
- NotificationConsumer: Real-time notifications
- LiveCartConsumer: Multi-tab cart sync
- LiveSearchConsumer: Search-as-you-type
- AnalyticsConsumer: Real-time analytics (staff)
- ChatConsumer: Live messaging
- AgentDashboardConsumer: Agent monitoring

Configuration:
- Use WEBSOCKET_PING_INTERVAL for keep-alive interval
- Use RATE_LIMITS for per-consumer rate limiting
- Monitor with: python manage.py websocket_monitor
- Health endpoints: /api/v1/health/websocket/
"""

from .base import ProducerWebSocketConsumer, ProducerJsonWebSocketConsumer
from .settings import (
    WEBSOCKET_PING_INTERVAL,
    RATE_LIMITS,
    get_rate_limit,
    is_authenticated_required,
)
from .monitoring import (
    WebSocketMetrics,
    WebSocketHealthCheck,
    ConnectionRecovery,
    WebSocketDebugger,
)
from .health_views import (
    websocket_health,
    websocket_status,
    websocket_metrics,
)

__all__ = [
    'ProducerWebSocketConsumer',
    'ProducerJsonWebSocketConsumer',
    'WebSocketMetrics',
    'WebSocketHealthCheck',
    'ConnectionRecovery',
    'WebSocketDebugger',
    'websocket_health',
    'websocket_status',
    'websocket_metrics',
    'WEBSOCKET_PING_INTERVAL',
    'RATE_LIMITS',
    'get_rate_limit',
    'is_authenticated_required',
]
