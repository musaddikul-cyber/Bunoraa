"""
Bunoraa Email Service - Comprehensive HTTP-Based Email Delivery
================================================================

A production-ready email service that uses HTTP APIs instead of SMTP.
Supports multiple providers with automatic failover, templates, and async delivery.

Features:
- Multiple provider support (SendGrid, Mailgun, Resend, Postmark, Amazon SES)
- Automatic failover between providers
- Retry logic with exponential backoff
- HTML/Plain text template rendering
- Attachment support
- Async delivery via Celery (optional)
- Rate limiting
- Detailed logging and error tracking
- Email validation
- Bulk sending support

Usage:
    from core.utils.email_service import EmailService, Email
    
    # Simple send
    EmailService.send(
        to='user@example.com',
        subject='Welcome!',
        template='emails/welcome.html',
        context={'name': 'John'}
    )
    
    # Advanced usage
    email = Email(
        to=['user@example.com'],
        subject='Order Confirmation',
        template='emails/order_confirmation.html',
        context={'order': order},
        attachments=[('receipt.pdf', pdf_data, 'application/pdf')]
    )
    EmailService.send_email(email)
"""

import hashlib
import hmac
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode
import base64

from django.conf import settings
from django.core.cache import cache
from django.template import TemplateDoesNotExist
from django.template.loader import render_to_string
from django.utils.html import strip_tags

logger = logging.getLogger('bunoraa.email')


# =============================================================================
# CONFIGURATION
# =============================================================================

class EmailProvider(Enum):
    """Supported email providers."""
    SENDGRID = 'sendgrid'
    MAILGUN = 'mailgun'
    RESEND = 'resend'
    POSTMARK = 'postmark'
    AMAZON_SES = 'amazon_ses'
    CONSOLE = 'console'  # For development - prints to console


@dataclass
class EmailConfig:
    """Email service configuration."""
    # Provider settings
    primary_provider: EmailProvider = EmailProvider.SENDGRID
    fallback_providers: List[EmailProvider] = field(default_factory=list)
    
    # API Keys (loaded from settings/env)
    sendgrid_api_key: str = ''
    mailgun_api_key: str = ''
    mailgun_domain: str = ''
    resend_api_key: str = ''
    postmark_api_key: str = ''
    aws_access_key: str = ''
    aws_secret_key: str = ''
    aws_region: str = 'us-east-1'
    
    # Default sender
    default_from_email: str = ''
    default_from_name: str = ''
    default_reply_to: str = ''
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # Rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000
    
    # Features
    enable_tracking: bool = True
    enable_async: bool = True
    validate_emails: bool = True
    
    @classmethod
    def from_settings(cls) -> 'EmailConfig':
        """Load configuration from Django settings."""
        import os
        
        # Determine primary provider
        provider_name = os.environ.get('EMAIL_PROVIDER', 'sendgrid').lower()
        try:
            primary_provider = EmailProvider(provider_name)
        except ValueError:
            primary_provider = EmailProvider.SENDGRID
        
        # Parse fallback providers
        fallback_str = os.environ.get('EMAIL_FALLBACK_PROVIDERS', '')
        fallback_providers = []
        if fallback_str:
            for p in fallback_str.split(','):
                try:
                    fallback_providers.append(EmailProvider(p.strip().lower()))
                except ValueError:
                    pass
        
        return cls(
            primary_provider=primary_provider,
            fallback_providers=fallback_providers,
            sendgrid_api_key=os.environ.get('SENDGRID_API_KEY', ''),
            mailgun_api_key=os.environ.get('MAILGUN_API_KEY', ''),
            mailgun_domain=os.environ.get('MAILGUN_DOMAIN', ''),
            resend_api_key=os.environ.get('RESEND_API_KEY', ''),
            postmark_api_key=os.environ.get('POSTMARK_API_KEY', ''),
            aws_access_key=os.environ.get('AWS_SES_ACCESS_KEY', ''),
            aws_secret_key=os.environ.get('AWS_SES_SECRET_KEY', ''),
            aws_region=os.environ.get('AWS_SES_REGION', 'us-east-1'),
            default_from_email=getattr(settings, 'DEFAULT_FROM_EMAIL', 'noreply@bunoraa.com'),
            default_from_name=getattr(settings, 'DEFAULT_FROM_NAME', 'Bunoraa'),
            default_reply_to=getattr(settings, 'DEFAULT_REPLY_TO', ''),
            max_retries=int(os.environ.get('EMAIL_MAX_RETRIES', 3)),
            retry_delay=float(os.environ.get('EMAIL_RETRY_DELAY', 1.0)),
            retry_backoff=float(os.environ.get('EMAIL_RETRY_BACKOFF', 2.0)),
            rate_limit_per_minute=int(os.environ.get('EMAIL_RATE_LIMIT_MINUTE', 100)),
            rate_limit_per_hour=int(os.environ.get('EMAIL_RATE_LIMIT_HOUR', 1000)),
            enable_tracking=os.environ.get('EMAIL_ENABLE_TRACKING', 'true').lower() == 'true',
            enable_async=os.environ.get('EMAIL_ENABLE_ASYNC', 'true').lower() == 'true',
            validate_emails=os.environ.get('EMAIL_VALIDATE', 'true').lower() == 'true',
        )


# =============================================================================
# EMAIL DATA CLASSES
# =============================================================================

@dataclass
class EmailAttachment:
    """Email attachment data."""
    filename: str
    content: bytes
    content_type: str = 'application/octet-stream'
    content_id: Optional[str] = None  # For inline attachments


@dataclass
class Email:
    """Email message data."""
    to: List[str]
    subject: str
    html_body: str = ''
    text_body: str = ''
    from_email: Optional[str] = None
    from_name: Optional[str] = None
    reply_to: Optional[str] = None
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    attachments: List[EmailAttachment] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    template: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    track_opens: bool = True
    track_clicks: bool = True
    
    # Scheduling
    send_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Normalize email data."""
        # Ensure to is a list
        if isinstance(self.to, str):
            self.to = [self.to]
        if isinstance(self.cc, str):
            self.cc = [self.cc]
        if isinstance(self.bcc, str):
            self.bcc = [self.bcc]
        
        # Normalize attachments
        normalized_attachments = []
        for att in self.attachments:
            if isinstance(att, EmailAttachment):
                normalized_attachments.append(att)
            elif isinstance(att, tuple):
                # Handle tuple format: (filename, content, content_type)
                if len(att) >= 2:
                    normalized_attachments.append(EmailAttachment(
                        filename=att[0],
                        content=att[1] if isinstance(att[1], bytes) else att[1].encode(),
                        content_type=att[2] if len(att) > 2 else 'application/octet-stream'
                    ))
        self.attachments = normalized_attachments


@dataclass
class EmailResult:
    """Result of sending an email."""
    success: bool
    provider: EmailProvider
    message_id: Optional[str] = None
    error: Optional[str] = None
    response_data: Optional[Dict] = None
    attempts: int = 1
    sent_at: Optional[datetime] = None


# =============================================================================
# EMAIL VALIDATION
# =============================================================================

class EmailValidator:
    """Email address validation utilities."""
    
    # RFC 5322 compliant email regex (simplified)
    EMAIL_REGEX = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    # Disposable email domains (partial list)
    DISPOSABLE_DOMAINS = {
        'mailinator.com', 'guerrillamail.com', 'tempmail.com', '10minutemail.com',
        'throwaway.email', 'fakeinbox.com', 'trashmail.com', 'maildrop.cc',
        'getnada.com', 'temp-mail.org', 'dispostable.com', 'mohmal.com',
    }
    
    @classmethod
    def is_valid(cls, email: str) -> bool:
        """Check if email address is valid."""
        if not email or not isinstance(email, str):
            return False
        return bool(cls.EMAIL_REGEX.match(email.strip().lower()))
    
    @classmethod
    def is_disposable(cls, email: str) -> bool:
        """Check if email is from a disposable domain."""
        if not email:
            return False
        domain = email.split('@')[-1].lower()
        return domain in cls.DISPOSABLE_DOMAINS
    
    @classmethod
    def normalize(cls, email: str) -> str:
        """Normalize email address."""
        if not email:
            return ''
        return email.strip().lower()
    
    @classmethod
    def validate_list(cls, emails: List[str]) -> Tuple[List[str], List[str]]:
        """Validate a list of emails. Returns (valid, invalid) tuple."""
        valid = []
        invalid = []
        for email in emails:
            if cls.is_valid(email):
                valid.append(cls.normalize(email))
            else:
                invalid.append(email)
        return valid, invalid


# =============================================================================
# HTTP CLIENT (No external dependencies)
# =============================================================================

class HTTPClient:
    """Simple HTTP client using urllib (no requests dependency)."""
    
    @staticmethod
    def request(
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        auth: Optional[Tuple[str, str]] = None,
        timeout: int = 30
    ) -> Tuple[int, Dict, str]:
        """
        Make an HTTP request.
        
        Returns: (status_code, headers, body)
        """
        import urllib.request
        import urllib.error
        
        headers = headers or {}
        
        # Prepare body
        body = None
        if json_data is not None:
            body = json.dumps(json_data).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        elif data is not None:
            body = urlencode(data).encode('utf-8')
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
        
        # Add auth
        if auth:
            credentials = f"{auth[0]}:{auth[1]}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers['Authorization'] = f'Basic {encoded}'
        
        # Create request
        req = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method.upper()
        )
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_body = response.read().decode('utf-8')
                response_headers = dict(response.headers)
                return response.status, response_headers, response_body
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else str(e)
            return e.code, dict(e.headers), error_body
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect: {e.reason}")


# =============================================================================
# EMAIL PROVIDERS
# =============================================================================

class BaseEmailProvider:
    """Base class for email providers."""
    
    provider: EmailProvider
    
    def __init__(self, config: EmailConfig):
        self.config = config
    
    def send(self, email: Email) -> EmailResult:
        """Send an email. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def _build_from_address(self, email: Email) -> str:
        """Build the From address string."""
        from_email = email.from_email or self.config.default_from_email
        from_name = email.from_name or self.config.default_from_name
        if from_name:
            return f"{from_name} <{from_email}>"
        return from_email


class SendGridProvider(BaseEmailProvider):
    """SendGrid email provider using v3 API."""
    
    provider = EmailProvider.SENDGRID
    API_URL = 'https://api.sendgrid.com/v3/mail/send'
    
    def send(self, email: Email) -> EmailResult:
        """Send email via SendGrid API."""
        if not self.config.sendgrid_api_key:
            return EmailResult(
                success=False,
                provider=self.provider,
                error='SendGrid API key not configured'
            )
        
        # Build payload
        payload = {
            'personalizations': [{
                'to': [{'email': addr} for addr in email.to],
            }],
            'from': {
                'email': email.from_email or self.config.default_from_email,
                'name': email.from_name or self.config.default_from_name,
            },
            'subject': email.subject,
            'content': [],
        }
        
        # Add CC/BCC
        if email.cc:
            payload['personalizations'][0]['cc'] = [{'email': addr} for addr in email.cc]
        if email.bcc:
            payload['personalizations'][0]['bcc'] = [{'email': addr} for addr in email.bcc]
        
        # Add reply-to
        reply_to = email.reply_to or self.config.default_reply_to
        if reply_to:
            payload['reply_to'] = {'email': reply_to}
        
        # Add content
        if email.text_body:
            payload['content'].append({'type': 'text/plain', 'value': email.text_body})
        if email.html_body:
            payload['content'].append({'type': 'text/html', 'value': email.html_body})
        
        # Add attachments
        if email.attachments:
            payload['attachments'] = []
            for att in email.attachments:
                payload['attachments'].append({
                    'content': base64.b64encode(att.content).decode(),
                    'filename': att.filename,
                    'type': att.content_type,
                    'disposition': 'inline' if att.content_id else 'attachment',
                    'content_id': att.content_id,
                })
        
        # Add tracking settings
        payload['tracking_settings'] = {
            'click_tracking': {'enable': email.track_clicks and self.config.enable_tracking},
            'open_tracking': {'enable': email.track_opens and self.config.enable_tracking},
        }
        
        # Add categories/tags
        if email.tags:
            payload['categories'] = email.tags[:10]  # SendGrid limit
        
        # Add custom headers
        if email.headers:
            payload['headers'] = email.headers
        
        # Send scheduled
        if email.send_at:
            payload['send_at'] = int(email.send_at.timestamp())
        
        # Make request
        try:
            status, headers, body = HTTPClient.request(
                method='POST',
                url=self.API_URL,
                headers={
                    'Authorization': f'Bearer {self.config.sendgrid_api_key}',
                    'Content-Type': 'application/json',
                },
                json_data=payload
            )
            
            if status in (200, 201, 202):
                message_id = headers.get('X-Message-Id', '')
                return EmailResult(
                    success=True,
                    provider=self.provider,
                    message_id=message_id,
                    sent_at=datetime.now()
                )
            else:
                error_data = json.loads(body) if body else {}
                error_msg = error_data.get('errors', [{}])[0].get('message', f'HTTP {status}')
                return EmailResult(
                    success=False,
                    provider=self.provider,
                    error=error_msg,
                    response_data=error_data
                )
        except Exception as e:
            return EmailResult(
                success=False,
                provider=self.provider,
                error=str(e)
            )


class MailgunProvider(BaseEmailProvider):
    """Mailgun email provider."""
    
    provider = EmailProvider.MAILGUN
    
    @property
    def api_url(self):
        return f'https://api.mailgun.net/v3/{self.config.mailgun_domain}/messages'
    
    def send(self, email: Email) -> EmailResult:
        """Send email via Mailgun API."""
        if not self.config.mailgun_api_key or not self.config.mailgun_domain:
            return EmailResult(
                success=False,
                provider=self.provider,
                error='Mailgun API key or domain not configured'
            )
        
        # Build form data
        data = {
            'from': self._build_from_address(email),
            'to': ','.join(email.to),
            'subject': email.subject,
        }
        
        if email.cc:
            data['cc'] = ','.join(email.cc)
        if email.bcc:
            data['bcc'] = ','.join(email.bcc)
        
        reply_to = email.reply_to or self.config.default_reply_to
        if reply_to:
            data['h:Reply-To'] = reply_to
        
        if email.text_body:
            data['text'] = email.text_body
        if email.html_body:
            data['html'] = email.html_body
        
        # Tags
        if email.tags:
            data['o:tag'] = email.tags
        
        # Tracking
        data['o:tracking'] = 'yes' if self.config.enable_tracking else 'no'
        data['o:tracking-clicks'] = 'htmlonly' if email.track_clicks else 'no'
        data['o:tracking-opens'] = 'yes' if email.track_opens else 'no'
        
        # Scheduled
        if email.send_at:
            data['o:deliverytime'] = email.send_at.strftime('%a, %d %b %Y %H:%M:%S +0000')
        
        # Custom headers
        for key, value in email.headers.items():
            data[f'h:{key}'] = value
        
        try:
            status, headers, body = HTTPClient.request(
                method='POST',
                url=self.api_url,
                auth=('api', self.config.mailgun_api_key),
                data=data
            )
            
            if status == 200:
                response_data = json.loads(body)
                return EmailResult(
                    success=True,
                    provider=self.provider,
                    message_id=response_data.get('id', ''),
                    sent_at=datetime.now()
                )
            else:
                error_data = json.loads(body) if body else {}
                return EmailResult(
                    success=False,
                    provider=self.provider,
                    error=error_data.get('message', f'HTTP {status}'),
                    response_data=error_data
                )
        except Exception as e:
            return EmailResult(
                success=False,
                provider=self.provider,
                error=str(e)
            )


class ResendProvider(BaseEmailProvider):
    """Resend email provider (modern, developer-friendly)."""
    
    provider = EmailProvider.RESEND
    API_URL = 'https://api.resend.com/emails'
    
    def send(self, email: Email) -> EmailResult:
        """Send email via Resend API."""
        if not self.config.resend_api_key:
            return EmailResult(
                success=False,
                provider=self.provider,
                error='Resend API key not configured'
            )
        
        # Build payload
        payload = {
            'from': self._build_from_address(email),
            'to': email.to,
            'subject': email.subject,
        }
        
        if email.cc:
            payload['cc'] = email.cc
        if email.bcc:
            payload['bcc'] = email.bcc
        
        reply_to = email.reply_to or self.config.default_reply_to
        if reply_to:
            payload['reply_to'] = reply_to
        
        if email.html_body:
            payload['html'] = email.html_body
        if email.text_body:
            payload['text'] = email.text_body
        
        # Attachments
        if email.attachments:
            payload['attachments'] = [
                {
                    'filename': att.filename,
                    'content': base64.b64encode(att.content).decode(),
                }
                for att in email.attachments
            ]
        
        # Tags
        if email.tags:
            payload['tags'] = [{'name': tag} for tag in email.tags[:5]]  # Resend limit
        
        # Headers
        if email.headers:
            payload['headers'] = email.headers
        
        try:
            status, headers, body = HTTPClient.request(
                method='POST',
                url=self.API_URL,
                headers={
                    'Authorization': f'Bearer {self.config.resend_api_key}',
                    'Content-Type': 'application/json',
                },
                json_data=payload
            )
            
            if status in (200, 201):
                response_data = json.loads(body)
                return EmailResult(
                    success=True,
                    provider=self.provider,
                    message_id=response_data.get('id', ''),
                    sent_at=datetime.now()
                )
            else:
                error_data = json.loads(body) if body else {}
                return EmailResult(
                    success=False,
                    provider=self.provider,
                    error=error_data.get('message', f'HTTP {status}'),
                    response_data=error_data
                )
        except Exception as e:
            return EmailResult(
                success=False,
                provider=self.provider,
                error=str(e)
            )


class PostmarkProvider(BaseEmailProvider):
    """Postmark email provider (excellent deliverability)."""
    
    provider = EmailProvider.POSTMARK
    API_URL = 'https://api.postmarkapp.com/email'
    
    def send(self, email: Email) -> EmailResult:
        """Send email via Postmark API."""
        if not self.config.postmark_api_key:
            return EmailResult(
                success=False,
                provider=self.provider,
                error='Postmark API key not configured'
            )
        
        # Build payload
        payload = {
            'From': self._build_from_address(email),
            'To': ','.join(email.to),
            'Subject': email.subject,
            'TrackOpens': email.track_opens and self.config.enable_tracking,
            'TrackLinks': 'HtmlAndText' if email.track_clicks and self.config.enable_tracking else 'None',
        }
        
        if email.cc:
            payload['Cc'] = ','.join(email.cc)
        if email.bcc:
            payload['Bcc'] = ','.join(email.bcc)
        
        reply_to = email.reply_to or self.config.default_reply_to
        if reply_to:
            payload['ReplyTo'] = reply_to
        
        if email.html_body:
            payload['HtmlBody'] = email.html_body
        if email.text_body:
            payload['TextBody'] = email.text_body
        
        # Tags (Postmark uses Tag singular)
        if email.tags:
            payload['Tag'] = email.tags[0]  # Postmark only supports one tag
        
        # Attachments
        if email.attachments:
            payload['Attachments'] = [
                {
                    'Name': att.filename,
                    'Content': base64.b64encode(att.content).decode(),
                    'ContentType': att.content_type,
                    'ContentID': att.content_id or '',
                }
                for att in email.attachments
            ]
        
        # Headers
        if email.headers:
            payload['Headers'] = [
                {'Name': k, 'Value': v} for k, v in email.headers.items()
            ]
        
        # Metadata
        if email.metadata:
            payload['Metadata'] = email.metadata
        
        try:
            status, headers, body = HTTPClient.request(
                method='POST',
                url=self.API_URL,
                headers={
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'X-Postmark-Server-Token': self.config.postmark_api_key,
                },
                json_data=payload
            )
            
            if status == 200:
                response_data = json.loads(body)
                return EmailResult(
                    success=True,
                    provider=self.provider,
                    message_id=response_data.get('MessageID', ''),
                    sent_at=datetime.now(),
                    response_data=response_data
                )
            else:
                error_data = json.loads(body) if body else {}
                return EmailResult(
                    success=False,
                    provider=self.provider,
                    error=error_data.get('Message', f'HTTP {status}'),
                    response_data=error_data
                )
        except Exception as e:
            return EmailResult(
                success=False,
                provider=self.provider,
                error=str(e)
            )


class AmazonSESProvider(BaseEmailProvider):
    """Amazon SES email provider using AWS Signature V4."""
    
    provider = EmailProvider.AMAZON_SES
    
    @property
    def api_url(self):
        return f'https://email.{self.config.aws_region}.amazonaws.com/'
    
    def _sign_request(self, method: str, url: str, headers: Dict, payload: str) -> Dict:
        """Sign request with AWS Signature Version 4."""
        # Simplified implementation - in production use boto3 or full AWS4 signer
        import hashlib
        import hmac
        from datetime import datetime
        
        now = datetime.utcnow()
        datestamp = now.strftime('%Y%m%d')
        amzdate = now.strftime('%Y%m%dT%H%M%SZ')
        
        headers['x-amz-date'] = amzdate
        headers['host'] = f'email.{self.config.aws_region}.amazonaws.com'
        
        # This is a simplified version - production should use full AWS4 signing
        # For full implementation, consider using boto3
        return headers
    
    def send(self, email: Email) -> EmailResult:
        """Send email via Amazon SES API."""
        if not self.config.aws_access_key or not self.config.aws_secret_key:
            return EmailResult(
                success=False,
                provider=self.provider,
                error='AWS SES credentials not configured'
            )
        
        # Note: For production, use boto3 for proper AWS4 signature
        # This is a simplified implementation
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            client = boto3.client(
                'ses',
                region_name=self.config.aws_region,
                aws_access_key_id=self.config.aws_access_key,
                aws_secret_access_key=self.config.aws_secret_key
            )
            
            # Build message
            message = {
                'Subject': {'Data': email.subject, 'Charset': 'UTF-8'},
                'Body': {}
            }
            
            if email.text_body:
                message['Body']['Text'] = {'Data': email.text_body, 'Charset': 'UTF-8'}
            if email.html_body:
                message['Body']['Html'] = {'Data': email.html_body, 'Charset': 'UTF-8'}
            
            destination = {'ToAddresses': email.to}
            if email.cc:
                destination['CcAddresses'] = email.cc
            if email.bcc:
                destination['BccAddresses'] = email.bcc
            
            kwargs = {
                'Source': self._build_from_address(email),
                'Destination': destination,
                'Message': message,
            }
            
            reply_to = email.reply_to or self.config.default_reply_to
            if reply_to:
                kwargs['ReplyToAddresses'] = [reply_to]
            
            if email.tags:
                kwargs['Tags'] = [{'Name': 'tag', 'Value': tag} for tag in email.tags[:10]]
            
            response = client.send_email(**kwargs)
            
            return EmailResult(
                success=True,
                provider=self.provider,
                message_id=response.get('MessageId', ''),
                sent_at=datetime.now(),
                response_data=response
            )
        except ImportError:
            return EmailResult(
                success=False,
                provider=self.provider,
                error='boto3 is required for Amazon SES. Install with: pip install boto3'
            )
        except Exception as e:
            return EmailResult(
                success=False,
                provider=self.provider,
                error=str(e)
            )


class ConsoleProvider(BaseEmailProvider):
    """Console email provider for development/testing."""
    
    provider = EmailProvider.CONSOLE
    
    def send(self, email: Email) -> EmailResult:
        """Print email to console instead of sending."""
        separator = '=' * 60
        print(f"\n{separator}")
        print("📧 EMAIL (Console Provider)")
        print(separator)
        print(f"From: {self._build_from_address(email)}")
        print(f"To: {', '.join(email.to)}")
        if email.cc:
            print(f"CC: {', '.join(email.cc)}")
        if email.bcc:
            print(f"BCC: {', '.join(email.bcc)}")
        print(f"Subject: {email.subject}")
        print(f"Reply-To: {email.reply_to or self.config.default_reply_to or 'N/A'}")
        print(separator)
        if email.text_body:
            print("Text Body:")
            print(email.text_body[:500] + ('...' if len(email.text_body) > 500 else ''))
        if email.html_body:
            print("\nHTML Body Preview:")
            print(strip_tags(email.html_body)[:500] + ('...' if len(email.html_body) > 500 else ''))
        if email.attachments:
            print(f"\nAttachments: {[a.filename for a in email.attachments]}")
        print(separator + "\n")
        
        return EmailResult(
            success=True,
            provider=self.provider,
            message_id=f"console-{int(time.time() * 1000)}",
            sent_at=datetime.now()
        )


# =============================================================================
# EMAIL SERVICE - Main Interface
# =============================================================================

class EmailService:
    """
    Main email service interface with automatic provider failover.
    
    Usage:
        # Simple send
        EmailService.send(
            to='user@example.com',
            subject='Welcome!',
            template='emails/welcome.html',
            context={'name': 'John'}
        )
        
        # Async send (queued via Celery)
        EmailService.send_async(
            to='user@example.com',
            subject='Welcome!',
            html_body='<h1>Welcome!</h1>'
        )
    """
    
    _config: Optional[EmailConfig] = None
    _providers: Dict[EmailProvider, BaseEmailProvider] = {}
    
    @classmethod
    def _get_config(cls) -> EmailConfig:
        """Get or create configuration."""
        if cls._config is None:
            cls._config = EmailConfig.from_settings()
        return cls._config
    
    @classmethod
    def _get_provider(cls, provider: EmailProvider) -> BaseEmailProvider:
        """Get or create a provider instance."""
        if provider not in cls._providers:
            config = cls._get_config()
            provider_classes = {
                EmailProvider.SENDGRID: SendGridProvider,
                EmailProvider.MAILGUN: MailgunProvider,
                EmailProvider.RESEND: ResendProvider,
                EmailProvider.POSTMARK: PostmarkProvider,
                EmailProvider.AMAZON_SES: AmazonSESProvider,
                EmailProvider.CONSOLE: ConsoleProvider,
            }
            cls._providers[provider] = provider_classes[provider](config)
        return cls._providers[provider]
    
    @classmethod
    def _render_template(cls, template: str, context: Dict) -> Tuple[str, str]:
        """Render email template and return (html, text) tuple."""
        html_body = ''
        text_body = ''
        
        # Try to render HTML template
        try:
            html_body = render_to_string(template, context)
        except TemplateDoesNotExist:
            logger.warning(f"Email template not found: {template}")
        
        # Try to render text version (template_name.txt)
        text_template = template.replace('.html', '.txt')
        try:
            text_body = render_to_string(text_template, context)
        except TemplateDoesNotExist:
            # Generate text from HTML
            if html_body:
                text_body = strip_tags(html_body)
        
        return html_body, text_body
    
    @classmethod
    def _check_rate_limit(cls) -> bool:
        """Check if we're within rate limits. Returns True if OK."""
        config = cls._get_config()
        now = time.time()
        
        # Check per-minute rate
        minute_key = f"email_rate:{int(now / 60)}"
        minute_count = cache.get(minute_key, 0)
        if minute_count >= config.rate_limit_per_minute:
            logger.warning(f"Email rate limit exceeded: {minute_count}/min")
            return False
        
        # Check per-hour rate
        hour_key = f"email_rate:{int(now / 3600)}"
        hour_count = cache.get(hour_key, 0)
        if hour_count >= config.rate_limit_per_hour:
            logger.warning(f"Email rate limit exceeded: {hour_count}/hr")
            return False
        
        # Increment counters
        cache.set(minute_key, minute_count + 1, 120)
        cache.set(hour_key, hour_count + 1, 7200)
        
        return True
    
    @classmethod
    def send_email(cls, email: Email, force_provider: Optional[EmailProvider] = None) -> EmailResult:
        """
        Send an email with automatic provider failover and retry logic.
        
        Args:
            email: Email object to send
            force_provider: Force using a specific provider (skip failover)
        
        Returns:
            EmailResult with success status and details
        """
        config = cls._get_config()
        
        # Render template if provided
        if email.template and not email.html_body:
            email.html_body, email.text_body = cls._render_template(
                email.template, email.context
            )
        
        # Validate emails
        if config.validate_emails:
            valid_to, invalid_to = EmailValidator.validate_list(email.to)
            if not valid_to:
                return EmailResult(
                    success=False,
                    provider=config.primary_provider,
                    error=f"No valid email addresses: {invalid_to}"
                )
            if invalid_to:
                logger.warning(f"Skipping invalid email addresses: {invalid_to}")
            email.to = valid_to
        
        # Check rate limit
        if not cls._check_rate_limit():
            return EmailResult(
                success=False,
                provider=config.primary_provider,
                error="Rate limit exceeded"
            )
        
        # Determine providers to try
        if force_provider:
            providers_to_try = [force_provider]
        else:
            providers_to_try = [config.primary_provider] + config.fallback_providers
        
        # Try each provider with retries
        last_result = None
        attempts = 0
        
        for provider in providers_to_try:
            provider_instance = cls._get_provider(provider)
            
            # Retry loop for this provider
            for retry in range(config.max_retries):
                attempts += 1
                
                try:
                    result = provider_instance.send(email)
                    result.attempts = attempts
                    
                    if result.success:
                        logger.info(
                            f"Email sent via {provider.value}: "
                            f"to={email.to}, subject={email.subject[:50]}, "
                            f"message_id={result.message_id}"
                        )
                        return result
                    
                    last_result = result
                    logger.warning(
                        f"Email send failed via {provider.value} "
                        f"(attempt {retry + 1}/{config.max_retries}): {result.error}"
                    )
                    
                except Exception as e:
                    logger.exception(f"Email provider {provider.value} error: {e}")
                    last_result = EmailResult(
                        success=False,
                        provider=provider,
                        error=str(e),
                        attempts=attempts
                    )
                
                # Wait before retry (exponential backoff)
                if retry < config.max_retries - 1:
                    delay = config.retry_delay * (config.retry_backoff ** retry)
                    time.sleep(delay)
            
            # Provider exhausted, try next
            logger.warning(f"Email provider {provider.value} exhausted, trying next...")
        
        # All providers failed
        logger.error(
            f"All email providers failed for: to={email.to}, subject={email.subject}"
        )
        return last_result or EmailResult(
            success=False,
            provider=config.primary_provider,
            error="All providers failed",
            attempts=attempts
        )
    
    @classmethod
    def send(
        cls,
        to: Union[str, List[str]],
        subject: str,
        template: Optional[str] = None,
        context: Optional[Dict] = None,
        html_body: Optional[str] = None,
        text_body: Optional[str] = None,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[Tuple]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> EmailResult:
        """
        Convenient method to send an email.
        
        Examples:
            # With template
            EmailService.send(
                to='user@example.com',
                subject='Welcome!',
                template='emails/welcome.html',
                context={'name': 'John'}
            )
            
            # With HTML body
            EmailService.send(
                to=['user1@example.com', 'user2@example.com'],
                subject='Update',
                html_body='<h1>Important Update</h1>'
            )
        """
        email = Email(
            to=to if isinstance(to, list) else [to],
            subject=subject,
            template=template,
            context=context or {},
            html_body=html_body or '',
            text_body=text_body or '',
            from_email=from_email,
            from_name=from_name,
            reply_to=reply_to,
            cc=cc or [],
            bcc=bcc or [],
            attachments=attachments or [],
            tags=tags or [],
            **kwargs
        )
        return cls.send_email(email)
    
    @classmethod
    def send_async(
        cls,
        to: Union[str, List[str]],
        subject: str,
        **kwargs
    ) -> str:
        """
        Send email asynchronously via Celery.
        
        Returns task ID for tracking.
        """
        try:
            from core.tasks import send_email_task
            
            task = send_email_task.delay(
                to=to if isinstance(to, list) else [to],
                subject=subject,
                **kwargs
            )
            return task.id
        except ImportError:
            # Celery not available, send synchronously
            logger.warning("Celery not available, sending email synchronously")
            cls.send(to=to, subject=subject, **kwargs)
            return "sync"
    
    @classmethod
    def send_bulk(
        cls,
        emails: List[Email],
        batch_size: int = 50,
        delay_between_batches: float = 1.0
    ) -> List[EmailResult]:
        """
        Send multiple emails in batches.
        
        Args:
            emails: List of Email objects
            batch_size: Number of emails per batch
            delay_between_batches: Seconds to wait between batches
        
        Returns:
            List of EmailResult objects
        """
        results = []
        
        for i in range(0, len(emails), batch_size):
            batch = emails[i:i + batch_size]
            
            for email in batch:
                result = cls.send_email(email)
                results.append(result)
            
            # Delay between batches
            if i + batch_size < len(emails):
                time.sleep(delay_between_batches)
        
        return results


# =============================================================================
# DJANGO EMAIL BACKEND (Optional Integration)
# =============================================================================

class BunoraaEmailBackend:
    """
    Django email backend that uses the EmailService.
    
    Add to settings.py:
        EMAIL_BACKEND = 'core.utils.email_service.BunoraaEmailBackend'
    """
    
    def __init__(self, fail_silently=False):
        self.fail_silently = fail_silently
    
    def send_messages(self, email_messages):
        """Send one or more EmailMessage objects."""
        num_sent = 0
        
        for message in email_messages:
            try:
                # Convert Django EmailMessage to our Email
                email = Email(
                    to=list(message.to),
                    subject=message.subject,
                    text_body=message.body,
                    from_email=message.from_email,
                    cc=list(message.cc) if message.cc else [],
                    bcc=list(message.bcc) if message.bcc else [],
                    reply_to=message.reply_to[0] if message.reply_to else None,
                    headers=dict(message.extra_headers) if message.extra_headers else {},
                )
                
                # Handle HTML alternative
                if hasattr(message, 'alternatives'):
                    for content, mimetype in message.alternatives:
                        if mimetype == 'text/html':
                            email.html_body = content
                            break
                
                # Handle attachments
                if message.attachments:
                    for attachment in message.attachments:
                        if isinstance(attachment, tuple):
                            email.attachments.append(EmailAttachment(
                                filename=attachment[0],
                                content=attachment[1] if isinstance(attachment[1], bytes) else attachment[1].encode(),
                                content_type=attachment[2] if len(attachment) > 2 else 'application/octet-stream'
                            ))
                
                result = EmailService.send_email(email)
                if result.success:
                    num_sent += 1
                elif not self.fail_silently:
                    raise Exception(f"Email failed: {result.error}")
                    
            except Exception as e:
                if not self.fail_silently:
                    raise
                logger.exception(f"Failed to send email: {e}")
        
        return num_sent
    
    def open(self):
        """Open connection (no-op for HTTP-based providers)."""
        pass
    
    def close(self):
        """Close connection (no-op for HTTP-based providers)."""
        pass


# =============================================================================
# CELERY TASK (Optional)
# =============================================================================

def create_email_task():
    """
    Create Celery task for async email sending.
    
    Add to core/tasks.py:
        from core.utils.email_service import create_email_task
        send_email_task = create_email_task()
    """
    try:
        from celery import shared_task
        
        @shared_task(bind=True, max_retries=3, default_retry_delay=60)
        def send_email_task(self, **kwargs):
            """Send email asynchronously."""
            try:
                result = EmailService.send(**kwargs)
                if not result.success:
                    raise Exception(result.error)
                return {'success': True, 'message_id': result.message_id}
            except Exception as e:
                logger.exception(f"Async email task failed: {e}")
                raise self.retry(exc=e)
        
        return send_email_task
    except ImportError:
        return None
