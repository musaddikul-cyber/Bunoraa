"""
Email Service Core Engine
=========================

The core email delivery engine that handles:
- SMTP connection pooling
- Email rendering and signing
- DKIM signing
- Delivery queue management
- Retry logic
"""

import base64
import email
import hashlib
import logging
import re
import smtplib
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import socket

from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.template import Template, Context
from django.utils import timezone
from django.utils.html import strip_tags

logger = logging.getLogger('bunoraa.email_service')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SMTPConfig:
    """SMTP server configuration."""
    host: str = 'localhost'
    port: int = 587
    username: str = ''
    password: str = ''
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30
    local_hostname: str = ''
    
    # Connection pooling
    pool_size: int = 5
    pool_timeout: int = 60
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    @classmethod
    def from_settings(cls) -> 'SMTPConfig':
        """Load configuration from Django settings."""
        import os
        return cls(
            host=os.environ.get('SMTP_HOST', getattr(settings, 'EMAIL_HOST', 'localhost')),
            port=int(os.environ.get('SMTP_PORT', getattr(settings, 'EMAIL_PORT', 587))),
            username=os.environ.get('SMTP_USERNAME', getattr(settings, 'EMAIL_HOST_USER', '')),
            password=os.environ.get('SMTP_PASSWORD', getattr(settings, 'EMAIL_HOST_PASSWORD', '')),
            use_tls=os.environ.get('SMTP_USE_TLS', str(getattr(settings, 'EMAIL_USE_TLS', True))).lower() == 'true',
            use_ssl=os.environ.get('SMTP_USE_SSL', str(getattr(settings, 'EMAIL_USE_SSL', False))).lower() == 'true',
            timeout=int(os.environ.get('SMTP_TIMEOUT', 30)),
        )


# =============================================================================
# DKIM SIGNING
# =============================================================================

class DKIMSigner:
    """
    DKIM (DomainKeys Identified Mail) signer.
    Signs emails with domain's private key.
    """
    
    def __init__(self, domain: str, selector: str, private_key: str):
        self.domain = domain
        self.selector = selector
        self.private_key = private_key
    
    def sign(self, message: str) -> str:
        """
        Sign an email message with DKIM.
        Returns the DKIM-Signature header value.
        """
        try:
            import dkim  # type: ignore  # dkimpy package
            
            signature = dkim.sign(
                message.encode(),
                self.selector.encode(),
                self.domain.encode(),
                self.private_key.encode(),
                include_headers=[
                    b'from', b'to', b'subject', b'date',
                    b'message-id', b'content-type'
                ]
            )
            return signature.decode()
        except ImportError:
            logger.warning("dkimpy package not installed, skipping DKIM signing")
            return ''
        except Exception as e:
            logger.error(f"DKIM signing failed: {e}")
            return ''


# =============================================================================
# LINK REWRITING FOR TRACKING
# =============================================================================

class LinkRewriter(HTMLParser):
    """
    Rewrites links in HTML for click tracking.
    """
    
    def __init__(self, message_id: str, base_url: str):
        super().__init__()
        self.message_id = message_id
        self.base_url = base_url
        self.output = []
    
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            new_attrs = []
            for name, value in attrs:
                if name == 'href' and value and not value.startswith(('#', 'mailto:', 'tel:')):
                    # Encode and rewrite URL for tracking
                    encoded_url = base64.urlsafe_b64encode(value.encode()).decode()
                    track_url = urljoin(
                        self.base_url,
                        f'/email/track/click/{self.message_id}/{encoded_url}/'
                    )
                    new_attrs.append(('href', track_url))
                    # Store original URL in data attribute
                    new_attrs.append(('data-original-href', value))
                else:
                    new_attrs.append((name, value))
            attrs = new_attrs
        
        attrs_str = ' '.join(f'{n}="{v}"' if v else n for n, v in attrs)
        self.output.append(f'<{tag} {attrs_str}>' if attrs_str else f'<{tag}>')
    
    def handle_endtag(self, tag):
        self.output.append(f'</{tag}>')
    
    def handle_data(self, data):
        self.output.append(data)
    
    def handle_comment(self, data):
        self.output.append(f'<!--{data}-->')
    
    def handle_decl(self, decl):
        self.output.append(f'<!{decl}>')
    
    def get_output(self) -> str:
        return ''.join(self.output)


def rewrite_links(html: str, message_id: str, base_url: str) -> str:
    """Rewrite all links in HTML for click tracking."""
    parser = LinkRewriter(message_id, base_url)
    parser.feed(html)
    return parser.get_output()


def add_tracking_pixel(html: str, message_id: str, base_url: str) -> str:
    """Add invisible tracking pixel to HTML email."""
    pixel_url = urljoin(base_url, f'/email/track/open/{message_id}/pixel.gif')
    pixel_tag = f'<img src="{pixel_url}" width="1" height="1" style="display:none;" alt="" />'
    
    # Insert before closing body tag
    if '</body>' in html.lower():
        html = re.sub(
            r'(</body>)',
            f'{pixel_tag}\\1',
            html,
            flags=re.IGNORECASE
        )
    else:
        html += pixel_tag
    
    return html


# =============================================================================
# EMAIL BUILDER
# =============================================================================

@dataclass
class EmailEnvelope:
    """
    Complete email envelope for sending.
    """
    message_id: str
    from_email: str
    from_name: str = ''
    to_email: str = ''
    to_name: str = ''
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    reply_to: str = ''
    subject: str = ''
    html_body: str = ''
    text_body: str = ''
    headers: Dict[str, str] = field(default_factory=dict)
    attachments: List[Tuple[str, bytes, str]] = field(default_factory=list)
    
    # Tracking
    track_opens: bool = True
    track_clicks: bool = True
    
    # DKIM
    dkim_domain: str = ''
    dkim_selector: str = ''
    dkim_private_key: str = ''


class EmailBuilder:
    """
    Builds MIME email messages from EmailEnvelope.
    """
    
    def __init__(self, base_url: str = ''):
        self.base_url = base_url or getattr(settings, 'SITE_URL', 'https://bunoraa.com')
    
    def build(self, envelope: EmailEnvelope) -> MIMEMultipart:
        """Build a complete MIME message from envelope."""
        
        # Create message
        msg = MIMEMultipart('alternative')
        
        # Headers
        msg['Message-ID'] = f'<{envelope.message_id}>'
        msg['From'] = formataddr((envelope.from_name, envelope.from_email)) if envelope.from_name else envelope.from_email
        msg['To'] = formataddr((envelope.to_name, envelope.to_email)) if envelope.to_name else envelope.to_email
        msg['Subject'] = envelope.subject
        msg['Date'] = formatdate(localtime=True)
        
        if envelope.reply_to:
            msg['Reply-To'] = envelope.reply_to
        
        if envelope.cc:
            msg['Cc'] = ', '.join(envelope.cc)
        
        # Custom headers
        for key, value in envelope.headers.items():
            msg[key] = value
        
        # Add List-Unsubscribe header for better deliverability
        unsubscribe_url = urljoin(self.base_url, f'/email/unsubscribe/{envelope.message_id}/')
        msg['List-Unsubscribe'] = f'<{unsubscribe_url}>'
        msg['List-Unsubscribe-Post'] = 'List-Unsubscribe=One-Click'
        
        # Prepare body content
        html_body = envelope.html_body
        text_body = envelope.text_body
        
        # Apply tracking
        if html_body:
            if envelope.track_clicks:
                html_body = rewrite_links(html_body, envelope.message_id, self.base_url)
            if envelope.track_opens:
                html_body = add_tracking_pixel(html_body, envelope.message_id, self.base_url)
        
        # Generate text body if not provided
        if not text_body and html_body:
            text_body = strip_tags(html_body)
        
        # Attach bodies
        if text_body:
            text_part = MIMEText(text_body, 'plain', 'utf-8')
            msg.attach(text_part)
        
        if html_body:
            html_part = MIMEText(html_body, 'html', 'utf-8')
            msg.attach(html_part)
        
        # Attachments
        if envelope.attachments:
            # Convert to mixed multipart
            mixed_msg = MIMEMultipart('mixed')
            mixed_msg.attach(msg)
            
            for filename, content, content_type in envelope.attachments:
                attachment = MIMEBase(*content_type.split('/', 1))
                attachment.set_payload(content)
                email.encoders.encode_base64(attachment)
                attachment.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename=filename
                )
                mixed_msg.attach(attachment)
            
            msg = mixed_msg
        
        return msg
    
    def build_raw(self, envelope: EmailEnvelope) -> str:
        """Build raw email string with optional DKIM signing."""
        msg = self.build(envelope)
        raw_message = msg.as_string()
        
        # DKIM signing
        if envelope.dkim_domain and envelope.dkim_private_key:
            signer = DKIMSigner(
                envelope.dkim_domain,
                envelope.dkim_selector,
                envelope.dkim_private_key
            )
            dkim_header = signer.sign(raw_message)
            if dkim_header:
                raw_message = dkim_header + raw_message
        
        return raw_message


# =============================================================================
# SMTP CONNECTION POOL
# =============================================================================

class SMTPConnection:
    """
    Single SMTP connection wrapper with reconnection support.
    """
    
    def __init__(self, config: SMTPConfig):
        self.config = config
        self.connection: Optional[smtplib.SMTP] = None
        self.connected_at: Optional[datetime] = None
        self.messages_sent: int = 0
        self._lock = False
    
    def connect(self) -> bool:
        """Establish SMTP connection."""
        try:
            if self.config.use_ssl:
                context = ssl.create_default_context()
                self.connection = smtplib.SMTP_SSL(
                    self.config.host,
                    self.config.port,
                    timeout=self.config.timeout,
                    context=context
                )
            else:
                self.connection = smtplib.SMTP(
                    self.config.host,
                    self.config.port,
                    timeout=self.config.timeout
                )
                
                if self.config.use_tls:
                    context = ssl.create_default_context()
                    self.connection.starttls(context=context)
            
            # Authenticate
            if self.config.username and self.config.password:
                self.connection.login(self.config.username, self.config.password)
            
            self.connected_at = datetime.now()
            self.messages_sent = 0
            logger.debug(f"SMTP connected to {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"SMTP connection failed: {e}")
            self.connection = None
            return False
    
    def disconnect(self):
        """Close SMTP connection."""
        if self.connection:
            try:
                self.connection.quit()
            except Exception:
                pass
            self.connection = None
            self.connected_at = None
    
    def is_connected(self) -> bool:
        """Check if connection is still valid."""
        if not self.connection:
            return False
        
        try:
            status = self.connection.noop()[0]
            return status == 250
        except Exception:
            return False
    
    def send(self, from_addr: str, to_addrs: List[str], message: str) -> Tuple[bool, str]:
        """
        Send an email message.
        
        Returns: (success, response/error)
        """
        if not self.is_connected():
            if not self.connect():
                return False, "Failed to connect to SMTP server"
        
        try:
            self.connection.sendmail(from_addr, to_addrs, message)
            self.messages_sent += 1
            return True, "Message sent successfully"
        except smtplib.SMTPRecipientsRefused as e:
            return False, f"Recipients refused: {e.recipients}"
        except smtplib.SMTPSenderRefused as e:
            return False, f"Sender refused: {e.sender}"
        except smtplib.SMTPDataError as e:
            return False, f"Data error: {e.smtp_error}"
        except smtplib.SMTPException as e:
            self.disconnect()
            return False, f"SMTP error: {str(e)}"
        except Exception as e:
            self.disconnect()
            return False, f"Error: {str(e)}"


class SMTPConnectionPool:
    """
    Connection pool for SMTP connections.
    Manages multiple connections for concurrent sending.
    """
    
    _instance = None
    _connections: List[SMTPConnection] = []
    _config: Optional[SMTPConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._connections = []
        return cls._instance
    
    @classmethod
    def initialize(cls, config: Optional[SMTPConfig] = None):
        """Initialize the connection pool."""
        instance = cls()
        instance._config = config or SMTPConfig.from_settings()
        instance._connections = []
        
        # Pre-create connections
        for _ in range(instance._config.pool_size):
            conn = SMTPConnection(instance._config)
            instance._connections.append(conn)
        
        return instance
    
    def get_connection(self) -> Optional[SMTPConnection]:
        """Get an available connection from the pool."""
        # Find an unlocked connection
        for conn in self._connections:
            if not conn._lock:
                conn._lock = True
                if not conn.is_connected():
                    conn.connect()
                return conn
        
        # All connections busy, create a temporary one
        logger.warning("Connection pool exhausted, creating temporary connection")
        temp_conn = SMTPConnection(self._config)
        temp_conn.connect()
        return temp_conn
    
    def release_connection(self, conn: SMTPConnection):
        """Release a connection back to the pool."""
        if conn in self._connections:
            conn._lock = False
        else:
            # Temporary connection, close it
            conn.disconnect()
    
    def close_all(self):
        """Close all connections in the pool."""
        for conn in self._connections:
            conn.disconnect()
        self._connections = []


# =============================================================================
# EMAIL DELIVERY ENGINE
# =============================================================================

class DeliveryResult:
    """Result of email delivery attempt."""
    
    def __init__(
        self,
        success: bool,
        message_id: str,
        response: str = '',
        error: str = '',
        attempt: int = 1,
        timestamp: Optional[datetime] = None
    ):
        self.success = success
        self.message_id = message_id
        self.response = response
        self.error = error
        self.attempt = attempt
        self.timestamp = timestamp or timezone.now()


class DeliveryEngine:
    """
    Core email delivery engine.
    Handles sending emails via SMTP with retry logic.
    """
    
    def __init__(self, config: Optional[SMTPConfig] = None):
        self.config = config or SMTPConfig.from_settings()
        self.pool = SMTPConnectionPool.initialize(self.config)
        self.builder = EmailBuilder()
    
    def send(self, envelope: EmailEnvelope) -> DeliveryResult:
        """
        Send a single email.
        
        Args:
            envelope: Email envelope to send
            
        Returns:
            DeliveryResult with success status
        """
        # Build raw message
        try:
            raw_message = self.builder.build_raw(envelope)
        except Exception as e:
            return DeliveryResult(
                success=False,
                message_id=envelope.message_id,
                error=f"Failed to build message: {e}"
            )
        
        # Collect all recipients
        all_recipients = [envelope.to_email] + envelope.cc + envelope.bcc
        all_recipients = [r for r in all_recipients if r]
        
        if not all_recipients:
            return DeliveryResult(
                success=False,
                message_id=envelope.message_id,
                error="No recipients specified"
            )
        
        # Retry loop
        last_error = ""
        for attempt in range(1, self.config.max_retries + 1):
            conn = self.pool.get_connection()
            
            try:
                success, response = conn.send(
                    envelope.from_email,
                    all_recipients,
                    raw_message
                )
                
                if success:
                    return DeliveryResult(
                        success=True,
                        message_id=envelope.message_id,
                        response=response,
                        attempt=attempt
                    )
                
                last_error = response
                logger.warning(
                    f"Email send attempt {attempt} failed: {response}"
                )
                
            except Exception as e:
                last_error = str(e)
                logger.exception(f"Email send attempt {attempt} error: {e}")
            
            finally:
                self.pool.release_connection(conn)
            
            # Wait before retry
            if attempt < self.config.max_retries:
                delay = self.config.retry_delay * (self.config.retry_backoff ** (attempt - 1))
                time.sleep(delay)
        
        return DeliveryResult(
            success=False,
            message_id=envelope.message_id,
            error=last_error,
            attempt=self.config.max_retries
        )
    
    def send_batch(
        self,
        envelopes: List[EmailEnvelope],
        delay_between: float = 0.1
    ) -> List[DeliveryResult]:
        """
        Send a batch of emails.
        
        Args:
            envelopes: List of email envelopes
            delay_between: Delay between sends (seconds)
            
        Returns:
            List of DeliveryResult objects
        """
        results = []
        
        for envelope in envelopes:
            result = self.send(envelope)
            results.append(result)
            
            if delay_between > 0:
                time.sleep(delay_between)
        
        return results


# =============================================================================
# QUEUE MANAGER
# =============================================================================

class QueueManager:
    """
    Manages email queue for background processing.
    """
    
    @staticmethod
    def enqueue(email_message) -> bool:
        """
        Add email to queue for sending.
        
        Args:
            email_message: EmailMessage model instance
        
        Returns:
            True if successfully queued
        """
        from .models import EmailMessage
        
        email_message.status = EmailMessage.Status.QUEUED
        email_message.save(update_fields=['status'])
        
        # Trigger async task if Celery is available
        celery_available = False
        try:
            from .tasks import process_email_queue
            process_email_queue.delay()
            celery_available = True
        except (ImportError, Exception) as e:
            logger.debug(f"Celery not available: {e}")
        
        # If Celery is not available, process synchronously
        if not celery_available:
            try:
                QueueManager.process_queue(batch_size=1)
            except Exception as e:
                logger.error(f"Failed to process email queue synchronously: {e}")
        
        return True
    
    @staticmethod
    def process_queue(batch_size: int = 100):
        """
        Process queued emails.
        
        Args:
            batch_size: Number of emails to process at once
        """
        from .models import EmailMessage, EmailEvent, DailyStats
        
        engine = DeliveryEngine()
        
        # Get queued messages
        messages = EmailMessage.objects.filter(
            status=EmailMessage.Status.QUEUED
        ).select_related('api_key', 'user', 'template')[:batch_size]
        
        for msg in messages:
            # Check if scheduled for later
            if msg.scheduled_at and msg.scheduled_at > timezone.now():
                continue
            
            # Build envelope
            envelope = EmailEnvelope(
                message_id=msg.message_id,
                from_email=msg.from_email,
                from_name=msg.from_name,
                to_email=msg.to_email,
                to_name=msg.to_name,
                cc=msg.cc or [],
                bcc=msg.bcc or [],
                reply_to=msg.reply_to,
                subject=msg.subject,
                html_body=msg.html_body,
                text_body=msg.text_body,
                headers=msg.headers or {},
            )
            
            # Update status
            msg.status = EmailMessage.Status.SENDING
            msg.attempt_count += 1
            msg.last_attempt_at = timezone.now()
            msg.save(update_fields=['status', 'attempt_count', 'last_attempt_at'])
            
            # Send
            result = engine.send(envelope)
            
            # Update message based on result
            if result.success:
                msg.status = EmailMessage.Status.SENT
                msg.sent_at = timezone.now()
                msg.smtp_response = result.response
                msg.save(update_fields=['status', 'sent_at', 'smtp_response'])
                
                # Create event
                EmailEvent.objects.create(
                    message=msg,
                    event_type=EmailEvent.EventType.SENT
                )
                
                # Update stats
                QueueManager._update_stats(msg.user_id, 'sent')
                
            else:
                msg.error_message = result.error
                
                # Check if should retry
                if msg.attempt_count < 3:
                    msg.status = EmailMessage.Status.QUEUED
                    msg.next_retry_at = timezone.now() + timedelta(
                        minutes=5 * msg.attempt_count
                    )
                else:
                    msg.status = EmailMessage.Status.FAILED
                    
                    # Create event
                    EmailEvent.objects.create(
                        message=msg,
                        event_type=EmailEvent.EventType.DROPPED,
                        data={'error': result.error}
                    )
                    
                    QueueManager._update_stats(msg.user_id, 'dropped')
                
                msg.save()
    
    @staticmethod
    def _update_stats(user_id: int, stat_type: str):
        """Update daily statistics."""
        from .models import DailyStats
        
        today = timezone.now().date()
        stats, created = DailyStats.objects.get_or_create(
            user_id=user_id,
            date=today
        )
        
        if stat_type == 'sent':
            stats.sent += 1
        elif stat_type == 'delivered':
            stats.delivered += 1
        elif stat_type == 'dropped':
            stats.dropped += 1
        elif stat_type == 'bounced':
            stats.bounced += 1
        
        stats.save()
    
    @staticmethod
    def retry_failed(max_age_hours: int = 24):
        """
        Retry failed emails that are due for retry.
        
        Args:
            max_age_hours: Don't retry emails older than this
        """
        from .models import EmailMessage
        
        cutoff = timezone.now() - timedelta(hours=max_age_hours)
        
        # Get failed messages due for retry
        messages = EmailMessage.objects.filter(
            status=EmailMessage.Status.QUEUED,
            next_retry_at__lte=timezone.now(),
            created_at__gte=cutoff
        )
        
        messages.update(
            status=EmailMessage.Status.QUEUED,
            next_retry_at=None
        )
