"""
Email Service API Views
=======================

REST API endpoints for the email service provider.
Provides SendGrid-compatible API for email operations.
"""

import base64
import logging
from datetime import timedelta

from django.db import transaction
from django.db.models import Count, Q, Sum
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import (
    APIKey, SenderDomain, SenderIdentity, EmailTemplate,
    EmailMessage, EmailEvent, EmailAttachment, Suppression,
    UnsubscribeGroup, UnsubscribePreference, Webhook, WebhookLog, DailyStats
)
from .serializers import (
    APIKeySerializer, APIKeyCreateSerializer,
    SenderDomainSerializer, SenderDomainCreateSerializer,
    SenderIdentitySerializer,
    EmailTemplateSerializer, EmailTemplateListSerializer,
    SendEmailSerializer, EmailMessageSerializer, EmailMessageListSerializer,
    EmailEventSerializer,
    SuppressionSerializer, SuppressionCreateSerializer,
    UnsubscribeGroupSerializer, WebhookSerializer, WebhookCreateSerializer,
    DailyStatsSerializer, StatsOverviewSerializer
)
from .engine import EmailEnvelope, QueueManager
from .authentication import APIKeyAuthentication
from .permissions import HasAPIKeyPermission

logger = logging.getLogger('bunoraa.email_service.api')


# =============================================================================
# API KEYS
# =============================================================================

class APIKeyViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing API keys.
    
    list: List all API keys
    create: Create a new API key (returns full key only once)
    retrieve: Get API key details
    update: Update API key settings
    destroy: Delete an API key
    """
    
    serializer_class = APIKeySerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return APIKey.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'create':
            return APIKeyCreateSerializer
        return APIKeySerializer
    
    @action(detail=True, methods=['post'])
    def rotate(self, request, pk=None):
        """Rotate an API key (create new, deactivate old)."""
        old_key = self.get_object()
        
        # Create new key with same settings
        new_key, full_key = APIKey.create_key(
            user=request.user,
            name=f"{old_key.name} (rotated)",
            permission=old_key.permission,
            allowed_ips=old_key.allowed_ips,
            rate_limit_per_minute=old_key.rate_limit_per_minute,
            rate_limit_per_hour=old_key.rate_limit_per_hour,
            rate_limit_per_day=old_key.rate_limit_per_day,
        )
        
        # Deactivate old key
        old_key.is_active = False
        old_key.save(update_fields=['is_active'])
        
        return Response({
            'message': 'API key rotated successfully',
            'old_key_id': str(old_key.id),
            'new_key': {
                'id': str(new_key.id),
                'api_key': full_key
            }
        })


# =============================================================================
# SENDER DOMAINS & IDENTITIES
# =============================================================================

class SenderDomainViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing sender domains.
    
    Provides domain verification, DNS record management,
    and DKIM/SPF/DMARC status.
    """
    
    serializer_class = SenderDomainSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return SenderDomain.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'create':
            return SenderDomainCreateSerializer
        return SenderDomainSerializer
    
    def perform_create(self, serializer):
        domain = serializer.validated_data['domain']
        
        # Create domain with DNS records
        sender_domain = SenderDomain.objects.create(
            user=self.request.user,
            domain=domain
        )
        
        # Generate DKIM keys
        self._generate_dkim_keys(sender_domain)
        
        return sender_domain
    
    def _generate_dkim_keys(self, domain):
        """Generate DKIM key pair for domain."""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.backends import default_backend
            
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # Get private key PEM
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode()
            
            # Get public key
            public_key = private_key.public_key()
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            
            # Extract just the key content for DNS
            public_key_dns = public_pem.replace(
                '-----BEGIN PUBLIC KEY-----', ''
            ).replace(
                '-----END PUBLIC KEY-----', ''
            ).replace('\n', '')
            
            domain.dkim_private_key = private_pem
            domain.dkim_public_key = public_key_dns
            domain.generate_dns_records()
            domain.save()
            
        except ImportError:
            logger.warning("cryptography package not installed, skipping DKIM key generation")
    
    @action(detail=True, methods=['post'])
    def verify(self, request, pk=None):
        """
        Verify domain DNS records.
        Checks SPF, DKIM, and verification TXT records.
        """
        domain = self.get_object()
        
        import dns.resolver
        
        results = {
            'verification': False,
            'spf': False,
            'dkim': False,
            'errors': []
        }
        
        try:
            # Check verification TXT record
            verify_host = f'_bunoraa.{domain.domain}'
            expected_value = f'bunoraa-verify={domain.verification_token}'
            
            try:
                answers = dns.resolver.resolve(verify_host, 'TXT')
                for rdata in answers:
                    if expected_value in str(rdata):
                        results['verification'] = True
                        break
            except dns.resolver.NXDOMAIN:
                results['errors'].append(f"Verification record not found at {verify_host}")
            except Exception as e:
                results['errors'].append(f"Verification check failed: {e}")
            
            # Check SPF
            try:
                answers = dns.resolver.resolve(domain.domain, 'TXT')
                for rdata in answers:
                    txt_value = str(rdata).strip('"')
                    if 'v=spf1' in txt_value and '_spf.bunoraa.com' in txt_value:
                        results['spf'] = True
                        break
            except Exception as e:
                results['errors'].append(f"SPF check failed: {e}")
            
            # Check DKIM
            dkim_host = f'{domain.dkim_selector}._domainkey.{domain.domain}'
            try:
                answers = dns.resolver.resolve(dkim_host, 'TXT')
                for rdata in answers:
                    if 'v=DKIM1' in str(rdata):
                        results['dkim'] = True
                        break
            except dns.resolver.NXDOMAIN:
                results['errors'].append(f"DKIM record not found at {dkim_host}")
            except Exception as e:
                results['errors'].append(f"DKIM check failed: {e}")
            
        except ImportError:
            return Response(
                {'error': 'DNS verification requires dnspython package'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Update domain status
        domain.spf_verified = results['spf']
        domain.dkim_verified = results['dkim']
        domain.last_checked_at = timezone.now()
        
        if results['verification'] and not domain.verified_at:
            domain.verification_status = SenderDomain.VerificationStatus.VERIFIED
            domain.verified_at = timezone.now()
        
        domain.save()
        
        return Response({
            'domain': domain.domain,
            'results': results,
            'is_fully_verified': domain.is_fully_verified
        })
    
    @action(detail=True, methods=['get'])
    def dns_records(self, request, pk=None):
        """Get DNS records that need to be added."""
        domain = self.get_object()
        return Response(domain.dns_records)


class SenderIdentityViewSet(viewsets.ModelViewSet):
    """API endpoint for managing sender identities."""
    
    serializer_class = SenderIdentitySerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return SenderIdentity.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['post'])
    def send_verification(self, request, pk=None):
        """Send verification email to the sender identity."""
        identity = self.get_object()
        
        # TODO: Send verification email
        
        return Response({'message': 'Verification email sent'})


# =============================================================================
# EMAIL TEMPLATES
# =============================================================================

class EmailTemplateViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing email templates.
    
    Supports template versioning and rendering preview.
    """
    
    serializer_class = EmailTemplateSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'template_id'
    
    def get_queryset(self):
        return EmailTemplate.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'list':
            return EmailTemplateListSerializer
        return EmailTemplateSerializer
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['post'])
    def preview(self, request, template_id=None):
        """Render template preview with sample data."""
        template = self.get_object()
        sample_data = request.data.get('data', {})
        
        try:
            rendered = template.render(sample_data)
            return Response(rendered)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=True, methods=['post'])
    def duplicate(self, request, template_id=None):
        """Create a copy of the template."""
        original = self.get_object()
        
        new_template = EmailTemplate.objects.create(
            user=request.user,
            name=f"{original.name} (Copy)",
            template_id=f"{original.template_id}-copy-{timezone.now().strftime('%Y%m%d%H%M%S')}",
            description=original.description,
            subject=original.subject,
            html_content=original.html_content,
            text_content=original.text_content,
            content_type=original.content_type,
            variables=original.variables,
        )
        
        return Response(EmailTemplateSerializer(new_template).data)


# =============================================================================
# MAIL SEND API
# =============================================================================

class MailSendView(APIView):
    """
    Primary API endpoint for sending emails.
    
    Similar to SendGrid's /v3/mail/send endpoint.
    Supports:
    - Single and batch sending
    - Template-based emails
    - Attachments
    - Scheduling
    - Tracking
    """
    
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [HasAPIKeyPermission]
    
    def post(self, request):
        """Send an email."""
        serializer = SendEmailSerializer(data=request.data, context={'request': request})
        
        if not serializer.is_valid():
            return Response(
                {'errors': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        data = serializer.validated_data
        api_key = request.auth
        user = api_key.user
        
        # Check suppression list
        suppressed = self._check_suppressions(user, data['to'])
        if suppressed:
            logger.warning(f"Suppressed recipients: {suppressed}")
            data['to'] = [r for r in data['to'] if r['email'] not in suppressed]
            
            if not data['to']:
                return Response(
                    {'error': 'All recipients are suppressed'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # Get template content if using template
        if data.get('template_id'):
            template = EmailTemplate.objects.get(
                template_id=data['template_id'],
                user=user
            )
            rendered = template.render(data.get('template_data', {}))
            data['subject'] = rendered['subject']
            data['html_body'] = rendered['html_body']
            data['text_body'] = rendered['text_body']
        
        # Create message records
        message_ids = []
        
        with transaction.atomic():
            for recipient in data['to']:
                # Create message
                message = EmailMessage.objects.create(
                    api_key=api_key,
                    user=user,
                    from_email=data.get('from_email', user.email),
                    from_name=data.get('from_name', ''),
                    to_email=recipient['email'],
                    to_name=recipient.get('name', ''),
                    cc=data.get('cc', []),
                    bcc=data.get('bcc', []),
                    reply_to=data.get('reply_to', ''),
                    subject=data['subject'],
                    html_body=data.get('html_body', ''),
                    text_body=data.get('text_body', ''),
                    template=EmailTemplate.objects.filter(
                        template_id=data.get('template_id'),
                        user=user
                    ).first(),
                    template_data=data.get('template_data', {}),
                    headers=data.get('headers', {}),
                    categories=data.get('categories', []),
                    tags=data.get('tags', []),
                    metadata=data.get('metadata', {}),
                    scheduled_at=data.get('send_at'),
                    ip_address=self._get_client_ip(request),
                    has_attachments=bool(data.get('attachments')),
                )
                
                # Save attachments
                if data.get('attachments'):
                    for att in data['attachments']:
                        content = base64.b64decode(att['content'])
                        EmailAttachment.objects.create(
                            message=message,
                            filename=att['filename'],
                            content_type=att['content_type'],
                            size=len(content),
                            # Note: In production, save to file storage
                        )
                
                # Queue for sending
                QueueManager.enqueue(message)
                
                message_ids.append(str(message.message_id))
        
        return Response({
            'message': 'Email(s) queued successfully',
            'message_ids': message_ids
        }, status=status.HTTP_202_ACCEPTED)
    
    def _check_suppressions(self, user, recipients):
        """Check which recipients are suppressed."""
        emails = [r['email'] for r in recipients]
        suppressed = Suppression.objects.filter(
            user=user,
            email__in=emails,
            is_active=True
        ).values_list('email', flat=True)
        return list(suppressed)
    
    def _get_client_ip(self, request):
        """Get client IP address."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR')


# =============================================================================
# EMAIL MESSAGES
# =============================================================================

class EmailMessageViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for viewing sent emails.
    
    Provides email activity, status, and event history.
    """
    
    serializer_class = EmailMessageSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'message_id'
    
    def get_queryset(self):
        queryset = EmailMessage.objects.filter(user=self.request.user)
        
        # Filters
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        to_email = self.request.query_params.get('to_email')
        if to_email:
            queryset = queryset.filter(to_email__icontains=to_email)
        
        start_date = self.request.query_params.get('start_date')
        if start_date:
            queryset = queryset.filter(created_at__date__gte=start_date)
        
        end_date = self.request.query_params.get('end_date')
        if end_date:
            queryset = queryset.filter(created_at__date__lte=end_date)
        
        return queryset.order_by('-created_at')
    
    def get_serializer_class(self):
        if self.action == 'list':
            return EmailMessageListSerializer
        return EmailMessageSerializer
    
    @action(detail=True, methods=['get'])
    def events(self, request, message_id=None):
        """Get all events for a message."""
        message = self.get_object()
        events = message.events.all()
        return Response(EmailEventSerializer(events, many=True).data)


# =============================================================================
# TRACKING ENDPOINTS
# =============================================================================

@method_decorator(csrf_exempt, name='dispatch')
class TrackOpenView(View):
    """
    Track email opens via tracking pixel.
    """
    
    # 1x1 transparent GIF
    PIXEL = base64.b64decode(
        'R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
    )
    
    def get(self, request, message_id):
        """Handle open tracking request."""
        self._record_open(request, message_id)
        
        return HttpResponse(
            self.PIXEL,
            content_type='image/gif'
        )
    
    def _record_open(self, request, message_id):
        """Record the open event."""
        try:
            message = EmailMessage.objects.get(message_id=message_id)
            
            # Create event
            EmailEvent.objects.create(
                message=message,
                event_type=EmailEvent.EventType.OPENED,
                ip_address=self._get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', '')[:500],
            )
            
            # Update message
            if not message.opened_at:
                message.opened_at = timezone.now()
                message.status = EmailMessage.Status.OPENED
                message.save(update_fields=['opened_at', 'status'])
            
            # Trigger webhooks
            self._trigger_webhooks(message, 'opened')
            
        except EmailMessage.DoesNotExist:
            pass
        except Exception as e:
            logger.error(f"Failed to record open: {e}")
    
    def _get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR')
    
    def _trigger_webhooks(self, message, event_type):
        """Trigger webhooks for the event."""
        from .tasks import send_webhook
        
        webhooks = Webhook.objects.filter(
            user=message.user,
            is_active=True
        ).filter(
            Q(events__contains=[event_type]) | Q(events__contains=['all'])
        )
        
        for webhook in webhooks:
            try:
                send_webhook.delay(
                    webhook_id=str(webhook.id),
                    event_type=event_type,
                    message_id=message.message_id
                )
            except Exception:
                pass


@method_decorator(csrf_exempt, name='dispatch')
class TrackClickView(View):
    """
    Track email link clicks and redirect to original URL.
    """
    
    def get(self, request, message_id, url):
        """Handle click tracking request."""
        # Decode original URL
        try:
            original_url = base64.urlsafe_b64decode(url).decode()
        except Exception:
            original_url = '/'
        
        self._record_click(request, message_id, original_url)
        
        # Redirect to original URL
        from django.shortcuts import redirect
        return redirect(original_url)
    
    def _record_click(self, request, message_id, url):
        """Record the click event."""
        try:
            message = EmailMessage.objects.get(message_id=message_id)
            
            # Create event
            EmailEvent.objects.create(
                message=message,
                event_type=EmailEvent.EventType.CLICKED,
                url=url,
                ip_address=self._get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', '')[:500],
            )
            
            # Update message
            if not message.clicked_at:
                message.clicked_at = timezone.now()
                message.status = EmailMessage.Status.CLICKED
                message.save(update_fields=['clicked_at', 'status'])
            
        except EmailMessage.DoesNotExist:
            pass
        except Exception as e:
            logger.error(f"Failed to record click: {e}")
    
    def _get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR')


# =============================================================================
# SUPPRESSIONS
# =============================================================================

class SuppressionViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing suppression lists.
    
    Includes bounces, spam reports, unsubscribes, and manual suppressions.
    """
    
    serializer_class = SuppressionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = Suppression.objects.filter(user=self.request.user)
        
        # Filter by type
        suppression_type = self.request.query_params.get('type')
        if suppression_type:
            queryset = queryset.filter(suppression_type=suppression_type)
        
        return queryset
    
    def get_serializer_class(self):
        if self.action == 'create':
            return SuppressionCreateSerializer
        return SuppressionSerializer
    
    def create(self, request, *args, **kwargs):
        """Add emails to suppression list."""
        serializer = SuppressionCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        data = serializer.validated_data
        created = 0
        
        for email in data['emails']:
            _, was_created = Suppression.objects.get_or_create(
                user=request.user,
                email=email.lower(),
                suppression_type=data['suppression_type'],
                defaults={'reason': data.get('reason', '')}
            )
            if was_created:
                created += 1
        
        return Response({
            'message': f'{created} email(s) added to suppression list',
            'created': created
        })
    
    @action(detail=False, methods=['delete'])
    def remove(self, request):
        """Remove emails from suppression list."""
        emails = request.data.get('emails', [])
        
        if not emails:
            return Response(
                {'error': 'No emails provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        deleted, _ = Suppression.objects.filter(
            user=request.user,
            email__in=[e.lower() for e in emails]
        ).update(
            is_active=False,
            removed_at=timezone.now(),
            removed_by=request.user
        )
        
        return Response({
            'message': f'{deleted} suppression(s) removed',
            'removed': deleted
        })
    
    @action(detail=False, methods=['post'])
    def check(self, request):
        """Check if emails are suppressed."""
        emails = request.data.get('emails', [])
        
        suppressed = Suppression.objects.filter(
            user=request.user,
            email__in=[e.lower() for e in emails],
            is_active=True
        ).values('email', 'suppression_type', 'reason')
        
        return Response({
            'suppressed': list(suppressed),
            'count': len(suppressed)
        })


# =============================================================================
# UNSUBSCRIBE MANAGEMENT
# =============================================================================

class UnsubscribeGroupViewSet(viewsets.ModelViewSet):
    """API endpoint for managing unsubscribe groups."""
    
    serializer_class = UnsubscribeGroupSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return UnsubscribeGroup.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class UnsubscribeView(View):
    """
    Public unsubscribe page for email recipients.
    """
    
    def get(self, request, message_id):
        """Show unsubscribe form."""
        try:
            message = EmailMessage.objects.get(message_id=message_id)
            
            # Record unsubscribe
            Suppression.objects.get_or_create(
                user=message.user,
                email=message.to_email,
                suppression_type=Suppression.SuppressionType.UNSUBSCRIBE,
                defaults={'source_message': message}
            )
            
            # Update message status
            message.status = EmailMessage.Status.UNSUBSCRIBED
            message.save(update_fields=['status'])
            
            # Create event
            EmailEvent.objects.create(
                message=message,
                event_type=EmailEvent.EventType.UNSUBSCRIBE,
            )
            
            return HttpResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Unsubscribed</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
                           display: flex; align-items: center; justify-content: center;
                           min-height: 100vh; margin: 0; background: #f7f7f7; }
                    .container { text-align: center; padding: 40px; background: white;
                                 border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    h1 { color: #1f2937; margin-bottom: 16px; }
                    p { color: #6b7280; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Successfully Unsubscribed</h1>
                    <p>You have been removed from our mailing list.</p>
                </div>
            </body>
            </html>
            """)
            
        except EmailMessage.DoesNotExist:
            return HttpResponse("Invalid unsubscribe link", status=404)


# =============================================================================
# WEBHOOKS
# =============================================================================

class WebhookViewSet(viewsets.ModelViewSet):
    """API endpoint for managing webhooks."""
    
    serializer_class = WebhookSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Webhook.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'create':
            return WebhookCreateSerializer
        return WebhookSerializer
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['post'])
    def test(self, request, pk=None):
        """Send a test webhook."""
        webhook = self.get_object()
        
        # Send test payload
        from .tasks import send_webhook_sync
        
        result = send_webhook_sync(
            webhook,
            event_type='test',
            payload={
                'event': 'test',
                'timestamp': timezone.now().isoformat(),
                'message': 'This is a test webhook'
            }
        )
        
        if result['success']:
            return Response({'message': 'Test webhook sent successfully'})
        else:
            return Response(
                {'error': result.get('error', 'Webhook failed')},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=True, methods=['get'])
    def logs(self, request, pk=None):
        """Get webhook delivery logs."""
        webhook = self.get_object()
        logs = WebhookLog.objects.filter(webhook=webhook)[:100]
        
        return Response([{
            'id': str(log.id),
            'event_type': log.event_type,
            'response_status': log.response_status,
            'success': log.success,
            'error_message': log.error_message,
            'created_at': log.created_at.isoformat()
        } for log in logs])


# =============================================================================
# STATISTICS
# =============================================================================

class StatsView(APIView):
    """
    API endpoint for email statistics and analytics.
    """
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get statistics overview."""
        user = request.user
        
        # Date range
        days = int(request.query_params.get('days', 30))
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Get daily stats
        daily_stats = DailyStats.objects.filter(
            user=user,
            date__gte=start_date,
            date__lte=end_date
        ).order_by('date')
        
        # Aggregate totals
        totals = daily_stats.aggregate(
            total_sent=Sum('sent'),
            total_delivered=Sum('delivered'),
            total_opened=Sum('unique_opens'),
            total_clicked=Sum('unique_clicks'),
            total_bounced=Sum('bounced'),
            total_spam=Sum('spam_reports'),
            total_unsubs=Sum('unsubscribes'),
        )
        
        # Calculate rates
        sent = totals['total_sent'] or 0
        delivered = totals['total_delivered'] or 0
        
        overview = {
            'period': f'{days} days',
            'total_sent': sent,
            'total_delivered': delivered,
            'total_opened': totals['total_opened'] or 0,
            'total_clicked': totals['total_clicked'] or 0,
            'total_bounced': totals['total_bounced'] or 0,
            'total_spam_reports': totals['total_spam'] or 0,
            'total_unsubscribes': totals['total_unsubs'] or 0,
            'delivery_rate': round((delivered / sent * 100) if sent > 0 else 0, 2),
            'open_rate': round((totals['total_opened'] / delivered * 100) if delivered > 0 else 0, 2),
            'click_rate': round((totals['total_clicked'] / delivered * 100) if delivered > 0 else 0, 2),
            'bounce_rate': round((totals['total_bounced'] / sent * 100) if sent > 0 else 0, 2),
            'daily_stats': DailyStatsSerializer(daily_stats, many=True).data
        }
        
        return Response(StatsOverviewSerializer(overview).data)
    
    @action(detail=False, methods=['get'])
    def realtime(self, request):
        """Get real-time stats for today."""
        user = request.user
        today = timezone.now().date()
        
        # Get today's message counts by status
        message_stats = EmailMessage.objects.filter(
            user=user,
            created_at__date=today
        ).values('status').annotate(count=Count('id'))
        
        stats = {s['status']: s['count'] for s in message_stats}
        
        return Response({
            'date': today.isoformat(),
            'queued': stats.get('queued', 0),
            'sending': stats.get('sending', 0),
            'sent': stats.get('sent', 0),
            'delivered': stats.get('delivered', 0),
            'opened': stats.get('opened', 0),
            'clicked': stats.get('clicked', 0),
            'bounced': stats.get('bounced', 0),
            'failed': stats.get('failed', 0),
        })
