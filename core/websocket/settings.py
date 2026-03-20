"""
WebSocket Settings - Production Configuration

This module centralizes all WebSocket configuration for easy management.
"""
import os
from django.conf import settings

# ============================================================================
# KEEP-ALIVE SETTINGS
# ============================================================================
WEBSOCKET_PING_INTERVAL = int(os.environ.get('WS_PING_INTERVAL', '30'))
WEBSOCKET_PONG_TIMEOUT = int(os.environ.get('WS_PONG_TIMEOUT', '10'))

# ============================================================================
# RATE LIMITING SETTINGS (per consumer type)
# ============================================================================
RATE_LIMITS = {
    'NotificationConsumer': {
        'messages_per_window': 20,
        'window_seconds': 10,
    },
    'LiveCartConsumer': {
        'messages_per_window': 50,
        'window_seconds': 10,
    },
    'LiveSearchConsumer': {
        'messages_per_window': 100,
        'window_seconds': 10,
    },
    'ChatConsumer': {
        'messages_per_window': 30,
        'window_seconds': 10,
    },
    'AnalyticsConsumer': {
        'messages_per_window': 50,
        'window_seconds': 10,
    },
    'AgentDashboardConsumer': {
        'messages_per_window': 50,
        'window_seconds': 10,
    },
}

# Override from environment
for consumer_name in RATE_LIMITS:
    env_key = f"WS_RATE_LIMIT_{consumer_name.upper()}"
    if os.environ.get(env_key):
        messages, window = os.environ.get(env_key).split(',')
        RATE_LIMITS[consumer_name] = {
            'messages_per_window': int(messages),
            'window_seconds': int(window),
        }

# ============================================================================
# CONNECTION SETTINGS
# ============================================================================
# Maximum message size (in bytes)
MAX_MESSAGE_SIZE = int(os.environ.get('WS_MAX_MESSAGE_SIZE', '8192'))

# Connection timeout (seconds)
CONNECTION_TIMEOUT = int(os.environ.get('WS_CONNECTION_TIMEOUT', '60'))

# Maximum concurrent connections per user
MAX_CONNECTIONS_PER_USER = int(os.environ.get('WS_MAX_CONNECTIONS_PER_USER', '5'))

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOG_ALL_MESSAGES = os.environ.get('WS_LOG_ALL_MESSAGES', 'false').lower() == 'true'
LOG_CONNECTION_EVENTS = os.environ.get('WS_LOG_CONNECTION_EVENTS', 'true').lower() == 'true'
LOG_ERRORS_ONLY = os.environ.get('WS_LOG_ERRORS_ONLY', 'false').lower() == 'true'

# ============================================================================
# SECURITY SETTINGS
# ============================================================================
# Require authentication for certain consumers
REQUIRE_AUTHENTICATION = {
    'NotificationConsumer': False,  # Allow anonymous for broadcasts
    'LiveCartConsumer': False,      # Session-based
    'LiveSearchConsumer': False,    # Public search
    'ChatConsumer': True,           # Authenticated users
    'AnalyticsConsumer': True,      # Staff only
    'AgentDashboardConsumer': True, # Agents only
}

# Enable CSRF protection for WebSocket upgrade
WEBSOCKET_REQUIRE_CSRF_PROTECTION = os.environ.get(
    'WS_REQUIRE_CSRF',
    'false'
).lower() == 'true'

# Allowed origins for WebSocket connections
ALLOWED_ORIGINS = [
    'https://bunoraa.com',
    'https://www.bunoraa.com',
    'https://api.bunoraa.com',
]

# Parse from environment
if os.environ.get('WS_ALLOWED_ORIGINS'):
    ALLOWED_ORIGINS = os.environ.get('WS_ALLOWED_ORIGINS').split(',')

# ============================================================================
# CACHE SETTINGS FOR WS
# ============================================================================
# Cache backend for rate limiting and presence tracking
WEBSOCKET_CACHE_BACKEND = os.environ.get('WS_CACHE_BACKEND', 'default')

# Presence timeout (seconds) - how long to keep user presence data
PRESENCE_TIMEOUT = int(os.environ.get('WS_PRESENCE_TIMEOUT', '300'))

# ============================================================================
# CHANNEL LAYERS SETTINGS
# ============================================================================
# These should be set in main settings.py but referenced here for clarity
# CHANNEL_LAYERS = {
#     'default': {
#         'BACKEND': 'channels_redis.core.RedisChannelLayer',
#         'CONFIG': {
#             'hosts': [('127.0.0.1', 6379)],
#             'capacity': 1500,
#             'expiry': 10,
#         },
#     },
# }

# ============================================================================
# FEATURE FLAGS
# ============================================================================
ENABLE_WEBSOCKETS = os.environ.get('ENABLE_WEBSOCKETS', 'true').lower() == 'true'
ENABLE_KEEP_ALIVE = os.environ.get('ENABLE_WS_KEEP_ALIVE', 'true').lower() == 'true'
ENABLE_RATE_LIMITING = os.environ.get('ENABLE_WS_RATE_LIMITING', 'true').lower() == 'true'
ENABLE_PRESENCE_TRACKING = os.environ.get('ENABLE_WS_PRESENCE', 'true').lower() == 'true'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_rate_limit(consumer_name):
    """Get rate limit config for a specific consumer."""
    return RATE_LIMITS.get(consumer_name, {
        'messages_per_window': 100,
        'window_seconds': 60,
    })


def is_authenticated_required(consumer_name):
    """Check if consumer requires authentication."""
    return REQUIRE_AUTHENTICATION.get(consumer_name, False)


def get_ping_interval():
    """Get configured ping interval."""
    return WEBSOCKET_PING_INTERVAL if ENABLE_KEEP_ALIVE else None


def is_origin_allowed(origin):
    """Check if origin is allowed for WebSocket connections."""
    if not ALLOWED_ORIGINS:
        return True  # Allow all if not configured
    return origin in ALLOWED_ORIGINS
