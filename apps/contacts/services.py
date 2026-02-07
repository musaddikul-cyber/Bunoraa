"""
Contacts Services
"""
from django.core.mail import send_mail
from django.utils import timezone
from django.conf import settings as django_settings

from .models import (
    ContactCategory, ContactInquiry, ContactResponse,
    ContactAttachment, StoreLocation, ContactSettings
)


class ContactInquiryService:
    """Service for contact inquiry operations."""
    
    @staticmethod
    def create_inquiry(
        name, email, subject, message,
        category_id=None, phone='', company='', order_number='',
        user=None, ip_address=None, user_agent='', source_page=''
    ):
        """Create a new contact inquiry."""
        category = None
        if category_id:
            category = ContactCategory.objects.filter(id=category_id, is_active=True).first()
        
        inquiry = ContactInquiry.objects.create(
            name=name,
            email=email,
            phone=phone,
            company=company,
            category=category,
            subject=subject,
            message=message,
            order_number=order_number,
            user=user,
            ip_address=ip_address,
            user_agent=user_agent[:1000] if user_agent else '',
            source_page=source_page
        )
        
        # Send notifications
        ContactInquiryService._send_notifications(inquiry)
        
        # Send auto-response
        ContactInquiryService._send_auto_response(inquiry)
        
        return inquiry
    
    @staticmethod
    def _send_notifications(inquiry):
        """Send notification emails to staff."""
        settings = ContactSettings.get_settings()
        
        if not settings.notify_on_new_inquiry:
            return
        
        recipients = settings.get_notification_emails_list()
        
        # Add category-specific recipients
        if inquiry.category:
            recipients.extend(inquiry.category.get_recipients_list())
        
        if not recipients:
            return
        
        # Remove duplicates
        recipients = list(set(recipients))
        
        try:
            subject = f'New Contact Inquiry: {inquiry.subject}'
            message = f"""
New contact inquiry received:

Name: {inquiry.name}
Email: {inquiry.email}
Phone: {inquiry.phone or 'Not provided'}
Company: {inquiry.company or 'Not provided'}
Category: {inquiry.category.name if inquiry.category else 'General'}
Subject: {inquiry.subject}

Message:
{inquiry.message}

---
Submitted: {inquiry.created_at}
IP: {inquiry.ip_address or 'Unknown'}
            """
            
            send_mail(
                subject,
                message,
                django_settings.DEFAULT_FROM_EMAIL,
                recipients,
                fail_silently=True
            )
        except Exception:
            pass
    
    @staticmethod
    def _send_auto_response(inquiry):
        """Send auto-response to the customer."""
        settings = ContactSettings.get_settings()
        
        if not settings.enable_auto_response:
            return
        
        # Check for category-specific auto-response
        subject = None
        message = None
        
        if inquiry.category:
            if inquiry.category.auto_response_subject and inquiry.category.auto_response_message:
                subject = inquiry.category.auto_response_subject
                message = inquiry.category.auto_response_message
        
        # Fall back to default auto-response
        if not subject or not message:
            subject = settings.default_auto_response_subject
            message = settings.default_auto_response_message
        
        if not subject or not message:
            return
        
        # Replace placeholders
        message = message.replace('{name}', inquiry.name)
        message = message.replace('{subject}', inquiry.subject)
        
        try:
            send_mail(
                subject,
                message,
                django_settings.DEFAULT_FROM_EMAIL,
                [inquiry.email],
                fail_silently=True
            )
            
            inquiry.auto_response_sent = True
            inquiry.save(update_fields=['auto_response_sent'])
        except Exception:
            pass
    
    @staticmethod
    def update_status(inquiry_id, status, user=None):
        """Update inquiry status."""
        inquiry = ContactInquiry.objects.filter(id=inquiry_id).first()
        if not inquiry:
            return None
        
        inquiry.status = status
        
        if status == 'responded':
            inquiry.responded_by = user
            inquiry.responded_at = timezone.now()
        
        inquiry.save()
        return inquiry
    
    @staticmethod
    def add_note(inquiry_id, note):
        """Add internal note to inquiry."""
        inquiry = ContactInquiry.objects.filter(id=inquiry_id).first()
        if not inquiry:
            return None
        
        if inquiry.internal_notes:
            inquiry.internal_notes += f'\n\n---\n{timezone.now().strftime("%Y-%m-%d %H:%M")}\n{note}'
        else:
            inquiry.internal_notes = f'{timezone.now().strftime("%Y-%m-%d %H:%M")}\n{note}'
        
        inquiry.save(update_fields=['internal_notes'])
        return inquiry
    
    @staticmethod
    def get_inquiry(inquiry_id):
        """Get an inquiry by ID."""
        return ContactInquiry.objects.filter(id=inquiry_id).select_related(
            'category', 'user', 'responded_by'
        ).prefetch_related('attachments', 'responses').first()
    
    @staticmethod
    def get_new_inquiries():
        """Get all new inquiries."""
        return ContactInquiry.objects.filter(status='new').select_related(
            'category', 'user'
        ).order_by('-created_at')
    
    @staticmethod
    def get_user_inquiries(user):
        """Get inquiries for a specific user."""
        return ContactInquiry.objects.filter(user=user).select_related(
            'category'
        ).order_by('-created_at')


class ContactResponseService:
    """Service for contact responses."""
    
    @staticmethod
    def send_response(inquiry_id, subject, message, user):
        """Send a response to an inquiry."""
        inquiry = ContactInquiry.objects.filter(id=inquiry_id).first()
        if not inquiry:
            return None
        
        response = ContactResponse.objects.create(
            inquiry=inquiry,
            subject=subject,
            message=message,
            sent_by=user
        )
        
        # Send email
        try:
            send_mail(
                subject,
                message,
                django_settings.DEFAULT_FROM_EMAIL,
                [inquiry.email],
                fail_silently=False
            )
            response.sent_at = timezone.now()
            response.delivered = True
            response.save(update_fields=['sent_at', 'delivered'])
        except Exception:
            pass
        
        # Update inquiry status
        inquiry.status = 'responded'
        inquiry.responded_by = user
        inquiry.responded_at = timezone.now()
        inquiry.save(update_fields=['status', 'responded_by', 'responded_at'])
        
        return response


class ContactAttachmentService:
    """Service for contact attachments."""
    
    @staticmethod
    def add_attachment(inquiry, file):
        """Add an attachment to an inquiry."""
        settings = ContactSettings.get_settings()
        
        # Validate file size
        max_size = settings.max_attachment_size_mb * 1024 * 1024
        if file.size > max_size:
            return None, f'File too large. Maximum size is {settings.max_attachment_size_mb}MB'
        
        # Validate file type
        ext = file.name.split('.')[-1].lower()
        allowed_types = settings.get_allowed_file_types_list()
        if allowed_types and ext not in allowed_types:
            return None, f'File type not allowed. Allowed types: {", ".join(allowed_types)}'
        
        attachment = ContactAttachment.objects.create(
            inquiry=inquiry,
            file=file,
            filename=file.name,
            file_size=file.size,
            content_type=file.content_type
        )
        
        return attachment, None


class StoreLocationService:
    """Service for store locations."""
    
    @staticmethod
    def get_all_locations():
        """Get all active locations."""
        return StoreLocation.objects.filter(is_active=True).order_by('order', 'name')
    
    @staticmethod
    def get_main_location():
        """Get the main/headquarters location."""
        return StoreLocation.objects.filter(is_active=True, is_main=True).first()
    
    @staticmethod
    def get_pickup_locations():
        """Get locations that offer pickup."""
        return StoreLocation.objects.filter(is_active=True, is_pickup_location=True).order_by('name')
    
    @staticmethod
    def get_returns_locations():
        """Get locations that accept returns."""
        return StoreLocation.objects.filter(is_active=True, is_returns_location=True).order_by('name')
    
    @staticmethod
    def get_location_by_slug(slug):
        """Get a location by slug."""
        return StoreLocation.objects.filter(slug=slug, is_active=True).first()
    
    @staticmethod
    def get_nearby_locations(latitude, longitude, radius_km=50):
        """Get locations within a radius (simple distance calculation)."""
        from django.db.models import F
        from math import radians, cos, sin, sqrt, atan2
        
        locations = StoreLocation.objects.filter(
            is_active=True,
            latitude__isnull=False,
            longitude__isnull=False
        )
        
        # Calculate distances (simple approach)
        nearby = []
        for loc in locations:
            lat1, lon1 = radians(float(latitude)), radians(float(longitude))
            lat2, lon2 = radians(float(loc.latitude)), radians(float(loc.longitude))
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            
            distance = 6371 * c  # Earth's radius in km
            
            if distance <= radius_km:
                loc.distance = distance
                nearby.append(loc)
        
        return sorted(nearby, key=lambda x: x.distance)
