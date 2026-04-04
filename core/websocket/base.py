"""
Production-ready WebSocket base classes and utilities.

Features:
- Keep-alive ping/pong mechanism
- Message validation with schemas
- Rate limiting per connection
- Connection lifecycle management
- Comprehensive logging
- Error handling and recovery
"""
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Set, Callable
from channels.generic.websocket import AsyncWebsocketConsumer, AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from django.core.cache import cache
from django.conf import settings
from django.core.exceptions import ValidationError
from pydantic import BaseModel, ValidationError as PydanticValidationError

logger = logging.getLogger('bunoraa.websocket')


# Message schemas for validation
class PingMessage(BaseModel):
    """Ping keep-alive message schema."""
    type: str = "ping"


class PongMessage(BaseModel):
    """Pong keep-alive response schema."""
    type: str = "pong"


class ProducerWebSocketConsumer(AsyncWebsocketConsumer):
    """
    Base WebSocket consumer with production-ready features:
    - Keep-alive ping/pong
    - Message validation
    - Rate limiting
    - Connection management
    - Comprehensive logging
    """
    
    # Override these in subclasses
    CONSUMER_NAME = "BaseConsumer"
    PING_INTERVAL = int(os.environ.get("WS_PING_INTERVAL", "30"))  # seconds
    PONG_TIMEOUT = int(os.environ.get("WS_PONG_TIMEOUT", "10"))  # seconds
    REQUIRE_APP_PONG = os.environ.get("WS_REQUIRE_APP_PONG", "false").lower() == "true"
    APP_PONG_MISS_LIMIT = int(os.environ.get("WS_APP_PONG_MISS_LIMIT", "3"))
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_MESSAGES = 100  # messages per window
    RATE_LIMIT_WINDOW = 60     # seconds
    MESSAGE_SCHEMAS: Dict[str, type] = {}  # Override with message type -> schema mapping
    
    async def connect(self):
        """Handle connection with keep-alive setup."""
        self.user = self.scope.get('user')
        self.client_ip = self.get_client_ip()
        self.connected_at = time.time()
        self.message_count = 0
        self.last_pong_time = time.time()
        self.pong_received = asyncio.Event()
        self.missed_pongs = 0
        self.keep_alive_task = None
        self._closing = False
        self._rate_limit_key = self.get_rate_limit_key()
        
        logger.info(
            f"[{self.CONSUMER_NAME}] Connection attempt",
            extra={
                'user': self.user.id if self.user else None,
                'ip': self.client_ip,
                'channel': self.channel_name,
            }
        )
        
        # Subclasses should call await self.accept() after their logic
    
    async def disconnect(self, close_code):
        """Handle disconnection and cleanup."""
        self._closing = True
        # Stop keep-alive task
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        connection_duration = time.time() - self.connected_at
        logger.info(
            f"[{self.CONSUMER_NAME}] Disconnected",
            extra={
                'user': self.user.id if self.user else None,
                'close_code': close_code,
                'duration_seconds': connection_duration,
                'messages_received': self.message_count,
            }
        )
    
    async def start_keep_alive(self):
        """Start the keep-alive ping/pong mechanism."""
        self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())

    async def close(self, code=None):
        self._closing = True
        return await super().close(code=code)
    
    async def _keep_alive_loop(self):
        """Periodic keep-alive ping/pong loop."""
        try:
            while True:
                await asyncio.sleep(self.PING_INTERVAL)
                if self._closing:
                    return
                try:
                    # Send ping
                    self.pong_received.clear()
                    try:
                        await self.send(json.dumps({
                            'type': 'ping',
                            'timestamp': time.time(),
                        }))
                    except RuntimeError as e:
                        if "websocket.send" in str(e):
                            return
                        raise

                    # Most browser clients do not implement app-level pong.
                    # Keep strict closes optional and configurable.
                    if not self.REQUIRE_APP_PONG:
                        continue

                    # Wait for pong with timeout
                    try:
                        await asyncio.wait_for(
                            self.pong_received.wait(),
                            timeout=self.PONG_TIMEOUT
                        )
                        self.missed_pongs = 0
                    except asyncio.TimeoutError:
                        self.missed_pongs += 1
                        if self.missed_pongs >= max(1, self.APP_PONG_MISS_LIMIT):
                            logger.warning(
                                f"[{self.CONSUMER_NAME}] No pong received, closing connection",
                                extra={'user': self.user.id if self.user else None}
                            )
                            await self.close(code=1000)
                            return
                        logger.debug(
                            f"[{self.CONSUMER_NAME}] Pong timeout (%s/%s), keeping connection open",
                            self.missed_pongs,
                            self.APP_PONG_MISS_LIMIT,
                        )
                        
                except Exception as e:
                    logger.error(
                        f"[{self.CONSUMER_NAME}] Keep-alive error: {e}",
                        extra={'user': self.user.id if self.user else None},
                        exc_info=True
                    )
                    return
        except asyncio.CancelledError:
            pass
    
    def get_rate_limit_key(self) -> str:
        """Generate rate limit key for this connection."""
        if self.user and self.user.is_authenticated:
            return f"ws_rate_limit:user:{self.user.id}"
        else:
            return f"ws_rate_limit:ip:{self.client_ip}"
    
    def get_client_ip(self) -> str:
        """Extract client IP from connection."""
        if self.scope.get('client'):
            return self.scope['client'][0]
        return 'unknown'
    
    async def is_rate_limited(self) -> bool:
        """Check if connection is rate limited."""
        if not self.RATE_LIMIT_ENABLED:
            return False
        
        try:
            current = cache.get(self._rate_limit_key, 0)
            if current >= self.RATE_LIMIT_MESSAGES:
                logger.warning(
                    f"[{self.CONSUMER_NAME}] Rate limit exceeded",
                    extra={
                        'user': self.user.id if self.user else None,
                        'rate_limit_key': self._rate_limit_key,
                    }
                )
                return True
            
            cache.set(
                self._rate_limit_key,
                current + 1,
                self.RATE_LIMIT_WINDOW
            )
            return False
        except Exception as e:
            logger.error(
                f"[{self.CONSUMER_NAME}] Rate limit check error: {e}",
                exc_info=True
            )
            return False
    
    async def receive(self, text_data):
        """Handle incoming text with validation and rate limiting."""
        self.message_count += 1
        self.last_pong_time = time.time()
        
        # Check rate limit
        if await self.is_rate_limited():
            await self.send_error("Rate limit exceeded")
            await self.close(code=1008)  # Policy violation
            return
        
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            logger.error(
                f"[{self.CONSUMER_NAME}] Invalid JSON received",
                extra={'user': self.user.id if self.user else None}
            )
            await self.send_error("Invalid JSON")
            return
        
        message_type = data.get('type')
        
        # Handle ping/pong
        if message_type == 'ping':
            await self.send(json.dumps({'type': 'pong', 'timestamp': time.time()}))
            return
        elif message_type == 'pong':
            self.last_pong_time = time.time()
            self.pong_received.set()
            return
        
        # Validate message if schema defined
        if message_type in self.MESSAGE_SCHEMAS:
            try:
                self.MESSAGE_SCHEMAS[message_type].parse_obj(data)
            except PydanticValidationError as e:
                logger.error(
                    f"[{self.CONSUMER_NAME}] Message validation failed",
                    extra={
                        'user': self.user.id if self.user else None,
                        'message_type': message_type,
                        'errors': str(e),
                    }
                )
                await self.send_error(f"Invalid message format: {e.errors()}")
                return
        
        # Call message handler
        handler_name = f"handle_{message_type}"
        if hasattr(self, handler_name):
            try:
                await getattr(self, handler_name)(data)
            except Exception as e:
                logger.error(
                    f"[{self.CONSUMER_NAME}] Message handler error",
                    extra={
                        'user': self.user.id if self.user else None,
                        'message_type': message_type,
                    },
                    exc_info=True
                )
                await self.send_error(f"Handler error: {str(e)}")
        else:
            logger.warning(
                f"[{self.CONSUMER_NAME}] No handler for message type",
                extra={
                    'user': self.user.id if self.user else None,
                    'message_type': message_type,
                }
            )
            await self.send_error(f"Unknown message type: {message_type}")
    
    async def send_error(self, message: str):
        """Send error message to client."""
        try:
            if self._closing:
                return
            await self.send(json.dumps({
                'type': 'error',
                'message': message,
                'timestamp': time.time(),
            }))
        except Exception as e:
            logger.error(
                f"[{self.CONSUMER_NAME}] Failed to send error: {e}",
                exc_info=True
            )
    
    async def send_success(self, data: Dict[str, Any]):
        """Send success message with data."""
        try:
            if self._closing:
                return
            data['timestamp'] = time.time()
            await self.send(json.dumps(data))
        except Exception as e:
            logger.error(
                f"[{self.CONSUMER_NAME}] Failed to send success: {e}",
                exc_info=True
            )


class ProducerJsonWebSocketConsumer(AsyncJsonWebsocketConsumer):
    """
    Base JSON WebSocket consumer with production-ready features.
    Use for consumers that primarily work with JSON messages.
    """
    
    CONSUMER_NAME = "BaseJsonConsumer"
    PING_INTERVAL = int(os.environ.get("WS_PING_INTERVAL", "30"))
    PONG_TIMEOUT = int(os.environ.get("WS_PONG_TIMEOUT", "10"))
    REQUIRE_APP_PONG = os.environ.get("WS_REQUIRE_APP_PONG", "false").lower() == "true"
    APP_PONG_MISS_LIMIT = int(os.environ.get("WS_APP_PONG_MISS_LIMIT", "3"))
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_MESSAGES = 100
    RATE_LIMIT_WINDOW = 60
    MESSAGE_SCHEMAS: Dict[str, type] = {}
    
    async def connect(self):
        """Handle connection with keep-alive setup."""
        self.user = self.scope.get('user')
        self.client_ip = self.get_client_ip()
        self.connected_at = time.time()
        self.message_count = 0
        self.last_pong_time = time.time()
        self.pong_received = asyncio.Event()
        self.missed_pongs = 0
        self.keep_alive_task = None
        self._closing = False
        self._rate_limit_key = self.get_rate_limit_key()
        
        logger.info(
            f"[{self.CONSUMER_NAME}] Connection attempt (JSON)",
            extra={
                'user': self.user.id if self.user else None,
                'ip': self.client_ip,
                'channel': self.channel_name,
            }
        )
    
    async def disconnect(self, close_code):
        """Handle disconnection and cleanup."""
        self._closing = True
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        connection_duration = time.time() - self.connected_at
        logger.info(
            f"[{self.CONSUMER_NAME}] Disconnected (JSON)",
            extra={
                'user': self.user.id if self.user else None,
                'close_code': close_code,
                'duration_seconds': connection_duration,
                'messages_received': self.message_count,
            }
        )
    
    async def start_keep_alive(self):
        """Start keep-alive mechanism."""
        self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())

    async def close(self, code=None):
        self._closing = True
        return await super().close(code=code)
    
    async def _keep_alive_loop(self):
        """Periodic keep-alive loop."""
        try:
            while True:
                await asyncio.sleep(self.PING_INTERVAL)
                if self._closing:
                    return
                try:
                    self.pong_received.clear()
                    try:
                        await self.send_json({'type': 'ping', 'timestamp': time.time()})
                    except RuntimeError as e:
                        if "websocket.send" in str(e):
                            return
                        raise

                    if not self.REQUIRE_APP_PONG:
                        continue

                    try:
                        await asyncio.wait_for(
                            self.pong_received.wait(),
                            timeout=self.PONG_TIMEOUT
                        )
                        self.missed_pongs = 0
                    except asyncio.TimeoutError:
                        self.missed_pongs += 1
                        if self.missed_pongs >= max(1, self.APP_PONG_MISS_LIMIT):
                            logger.warning(
                                f"[{self.CONSUMER_NAME}] No pong, closing",
                                extra={'user': self.user.id if self.user else None}
                            )
                            await self.close(code=1000)
                            return
                        logger.debug(
                            f"[{self.CONSUMER_NAME}] Pong timeout (%s/%s), keeping connection open",
                            self.missed_pongs,
                            self.APP_PONG_MISS_LIMIT,
                        )
                except Exception as e:
                    logger.error(
                        f"[{self.CONSUMER_NAME}] Keep-alive error: {e}",
                        exc_info=True
                    )
                    return
        except asyncio.CancelledError:
            pass
    
    def get_rate_limit_key(self) -> str:
        """Generate rate limit key."""
        if self.user and self.user.is_authenticated:
            return f"ws_rate_limit:user:{self.user.id}"
        return f"ws_rate_limit:ip:{self.client_ip}"
    
    def get_client_ip(self) -> str:
        """Extract client IP."""
        if self.scope.get('client'):
            return self.scope['client'][0]
        return 'unknown'
    
    async def is_rate_limited(self) -> bool:
        """Check rate limiting."""
        if not self.RATE_LIMIT_ENABLED:
            return False
        
        try:
            current = cache.get(self._rate_limit_key, 0)
            if current >= self.RATE_LIMIT_MESSAGES:
                logger.warning(f"[{self.CONSUMER_NAME}] Rate limited")
                return True
            cache.set(self._rate_limit_key, current + 1, self.RATE_LIMIT_WINDOW)
            return False
        except Exception as e:
            logger.error(f"[{self.CONSUMER_NAME}] Rate check error: {e}", exc_info=True)
            return False
    
    async def receive_json(self, content):
        """Handle incoming JSON."""
        self.message_count += 1
        self.last_pong_time = time.time()
        
        if await self.is_rate_limited():
            await self.send_error("Rate limit exceeded")
            await self.close(code=1008)
            return
        
        message_type = content.get('type')
        
        if message_type == 'ping':
            await self.send_json({'type': 'pong', 'timestamp': time.time()})
            return
        elif message_type == 'pong':
            self.last_pong_time = time.time()
            self.pong_received.set()
            return
        
        if message_type in self.MESSAGE_SCHEMAS:
            try:
                self.MESSAGE_SCHEMAS[message_type].parse_obj(content)
            except PydanticValidationError as e:
                logger.error(f"[{self.CONSUMER_NAME}] Validation failed: {e}")
                await self.send_error(f"Invalid format: {e.errors()}")
                return
        
        handler_name = f"handle_{message_type}"
        if hasattr(self, handler_name):
            try:
                await getattr(self, handler_name)(content)
            except Exception as e:
                logger.error(f"[{self.CONSUMER_NAME}] Handler error: {e}", exc_info=True)
                await self.send_error(f"Handler error: {str(e)}")
        else:
            logger.warning(f"[{self.CONSUMER_NAME}] Unknown message type: {message_type}")
            await self.send_error(f"Unknown type: {message_type}")
    
    async def send_error(self, message: str):
        """Send error JSON."""
        try:
            if self._closing:
                return
            await self.send_json({'type': 'error', 'message': message, 'timestamp': time.time()})
        except Exception as e:
            logger.error(f"[{self.CONSUMER_NAME}] Send error failed: {e}", exc_info=True)
    
    async def send_success(self, data: Dict[str, Any]):
        """Send success JSON."""
        try:
            if self._closing:
                return
            data['timestamp'] = time.time()
            await self.send_json(data)
        except Exception as e:
            logger.error(f"[{self.CONSUMER_NAME}] Send success failed: {e}", exc_info=True)
