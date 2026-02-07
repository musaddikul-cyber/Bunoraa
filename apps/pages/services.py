"""
Pages services
"""
from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string
from django.utils import timezone

from .models import Page, FAQ, ContactMessage, SiteSettings, Subscriber


class PageService:
    """Service for managing pages."""
    
    @staticmethod
    def get_menu_pages():
        """Get pages that should appear in main menu."""
        return Page.objects.filter(
            is_published=True,
            show_in_header=True
        ).order_by('menu_order')
    
    @staticmethod
    def get_footer_pages():
        """Get pages that should appear in footer."""
        return Page.objects.filter(
            is_published=True,
            show_in_footer=True
        )
    
    @staticmethod
    def get_page_by_slug(slug):
        """Get published page by slug."""
        return Page.objects.filter(
            slug=slug,
            is_published=True
        ).first()


class FAQService:
    """Service for managing FAQs."""
    
    @staticmethod
    def get_active_faqs():
        """Get all active FAQs."""
        return FAQ.objects.filter(is_active=True).order_by('sort_order')
    
    @staticmethod
    def get_faqs_by_category(category):
        """Get FAQs by category."""
        return FAQ.objects.filter(
            is_active=True,
            category=category
        ).order_by('sort_order')
    
    @staticmethod
    def get_faq_categories():
        """Get list of FAQ categories."""
        return FAQ.objects.filter(
            is_active=True
        ).values_list('category', flat=True).distinct()


class ContactService:
    """Service for managing contact messages."""
    
    @staticmethod
    def create_message(name, email, subject, message, phone=None):
        """Create a contact message and send notification."""
        contact = ContactMessage.objects.create(
            name=name,
            email=email,
            phone=phone,
            subject=subject,
            message=message
        )
        
        # Send notification email
        ContactService._send_notification(contact)
        
        return contact
    
    @staticmethod
    def _send_notification(contact):
        """Send notification email to admin."""
        site_settings = SiteSettings.get_settings()
        notification_email = site_settings.support_email or settings.DEFAULT_FROM_EMAIL
        
        subject = f'New Contact Form Submission: {contact.subject}'
        message = f"""
New contact form submission:

Name: {contact.name}
Email: {contact.email}
Phone: {contact.phone or 'Not provided'}
Subject: {contact.subject}

Message:
{contact.message}

Received at: {contact.created_at}
        """
        
        try:
            send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[notification_email],
                fail_silently=True
            )
        except Exception:
            pass
    
    @staticmethod
    def mark_as_read(message_id):
        """Mark a message as read."""
        ContactMessage.objects.filter(id=message_id).update(is_read=True)
    
    @staticmethod
    def reply_to_message(message_id, reply_text, replied_by):
        """Reply to a contact message."""
        message = ContactMessage.objects.filter(id=message_id).first()
        if not message:
            return None
        
        message.is_replied = True
        message.replied_at = timezone.now()
        message.replied_by = replied_by
        message.save()
        
        # Send reply email to customer
        try:
            send_mail(
                subject=f'Re: {message.subject}',
                message=reply_text,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[message.email],
                fail_silently=True
            )
        except Exception:
            pass
        
        return message
    
    @staticmethod
    def get_unread_count():
        """Get count of unread messages."""
        return ContactMessage.objects.filter(is_read=False).count()


class SubscriberService:
    """Service for managing newsletter subscribers."""
    
    @staticmethod
    def subscribe(email, name=None, source='website'):
        """Subscribe an email to the newsletter."""
        existing = Subscriber.objects.filter(email=email).first()
        
        if existing:
            if existing.is_active:
                return {'success': False, 'message': 'Already subscribed'}
            else:
                existing.is_active = True
                existing.unsubscribed_at = None
                existing.save()
                return {'success': True, 'message': 'Re-subscribed successfully'}
        
        Subscriber.objects.create(
            email=email,
            name=name,
            source=source
        )
        
        return {'success': True, 'message': 'Subscribed successfully'}
    
    @staticmethod
    def unsubscribe(email):
        """Unsubscribe an email from the newsletter."""
        subscriber = Subscriber.objects.filter(email=email, is_active=True).first()
        
        if not subscriber:
            return {'success': False, 'message': 'Not found or already unsubscribed'}
        
        subscriber.is_active = False
        subscriber.unsubscribed_at = timezone.now()
        subscriber.save()
        
        return {'success': True, 'message': 'Unsubscribed successfully'}
    
    @staticmethod
    def get_active_subscribers():
        """Get all active subscribers."""
        return Subscriber.objects.filter(is_active=True)
    
    @staticmethod
    def get_subscriber_count():
        """Get count of active subscribers."""
        return Subscriber.objects.filter(is_active=True).count()
    
    @staticmethod
    def export_subscribers():
        """Export subscribers list for email marketing."""
        return list(
            Subscriber.objects.filter(is_active=True).values(
                'email', 'name', 'source', 'created_at'
            )
        )
