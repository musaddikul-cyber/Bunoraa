"""
Email Service Integration for Accounts App
Integrates the new Email Service Provider with user authentication flows
(verification, password reset, welcome emails, etc.)
"""
import secrets
import logging
from datetime import timedelta
from typing import Optional

from django.utils import timezone
from django.conf import settings
from django.template.loader import render_to_string
from django.contrib.auth import get_user_model

from apps.email_service.models import EmailMessage, APIKey, EmailTemplate, UnsubscribeGroup
from apps.email_service.engine import DeliveryEngine

User = get_user_model()
logger = logging.getLogger('bunoraa.email_integration')


class EmailServiceIntegration:
    """
    Integration layer between accounts app and email service provider.
    Handles sending verification, password reset, and other auth emails.
    """
    
    # Template names in the email service
    TEMPLATES = {
        'verify_email': 'Email Verification',
        'reset_password': 'Password Reset',
        'welcome': 'Welcome Email',
        'account_deleted': 'Account Deleted Confirmation',
        'email_changed': 'Email Change Verification',
    }
    
    @staticmethod
    def get_api_key() -> Optional[APIKey]:
        """Get the default API key for system emails."""
        # Try to get a system API key, or the first available one
        try:
            return APIKey.objects.filter(permission='send').first()
        except Exception as e:
            logger.error(f"Failed to get API key: {e}")
            return None
    
    @staticmethod
    def get_or_create_unsubscribe_group(name: str) -> Optional[UnsubscribeGroup]:
        """Get or create an unsubscribe group for transactional emails."""
        try:
            group, created = UnsubscribeGroup.objects.get_or_create(
                name=name,
                defaults={
                    'description': f'System emails: {name}',
                    'is_active': True,
                }
            )
            return group
        except Exception as e:
            logger.error(f"Failed to create unsubscribe group: {e}")
            return None
    
    @staticmethod
    def send_verification_email(user: User, token: str) -> bool:
        """
        Send email verification email using the Email Service Provider.
        
        Args:
            user: User instance
            token: Verification token
            
        Returns:
            True if queued successfully, False otherwise
        """
        try:
            # Build verification URL
            site_url = getattr(settings, 'SITE_URL', 'https://bunoraa.com')
            verification_url = f"{site_url}/account/verify-email/{token}/"
            
            # Render email content
            context = {
                'user': user,
                'verification_url': verification_url,
                'site_name': 'Bunoraa',
                'token_expires_hours': 24,
            }
            
            html_content = render_to_string('emails/verify_email.html', context)
            text_content = f'Verify your email: {verification_url}'
            
            # Get API key
            api_key = EmailServiceIntegration.get_api_key()
            if not api_key:
                logger.error("No API key available for sending verification email")
                return False
            
            # Create email message
            message = EmailMessage.objects.create(
                message_id=f"verify_{user.id}_{secrets.token_hex(4)}@bunoraa.com",
                api_key=api_key,
                user=user,
                to_email=user.email,
                from_email=settings.DEFAULT_FROM_EMAIL,
                from_name='Bunoraa',
                subject='Verify Your Email Address',
                html_body=html_content,
                text_body=text_content,
                status=EmailMessage.Status.QUEUED,
                metadata={
                    'user_id': str(user.id),
                    'email_type': 'verification',
                    'token': token,
                }
            )
            
            # Enqueue for sending
            from apps.email_service.engine import QueueManager
            QueueManager.enqueue(message)
            
            logger.info(f"Verification email queued for {user.email} (Message ID: {message.message_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send verification email to {user.email}: {e}")
            return False
    
    @staticmethod
    def send_password_reset_email(user: User, token: str) -> bool:
        """
        Send password reset email using the Email Service Provider.
        
        Args:
            user: User instance
            token: Password reset token
            
        Returns:
            True if queued successfully, False otherwise
        """
        try:
            # Build reset URL
            site_url = getattr(settings, 'SITE_URL', 'https://bunoraa.com')
            reset_url = f"{site_url}/account/reset-password/{token}/"
            
            # Render email content
            context = {
                'user': user,
                'reset_url': reset_url,
                'site_name': 'Bunoraa',
                'token_expires_hours': 1,
            }
            
            html_content = render_to_string('emails/reset_password.html', context)
            text_content = f'Reset your password: {reset_url}'
            
            # Get API key
            api_key = EmailServiceIntegration.get_api_key()
            if not api_key:
                logger.error("No API key available for sending password reset email")
                return False
            
            # Create email message
            message = EmailMessage.objects.create(
                message_id=f"reset_{user.id}_{secrets.token_hex(4)}@bunoraa.com",
                api_key=api_key,
                user=user,
                to_email=user.email,
                from_email=settings.DEFAULT_FROM_EMAIL,
                from_name='Bunoraa',
                subject='Reset Your Password',
                html_body=html_content,
                text_body=text_content,
                status=EmailMessage.Status.QUEUED,
                metadata={
                    'user_id': str(user.id),
                    'email_type': 'password_reset',
                    'token': token,
                }
            )
            
            # Enqueue for sending
            from apps.email_service.engine import QueueManager
            QueueManager.enqueue(message)
            
            logger.info(f"Password reset email queued for {user.email} (Message ID: {message.message_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send password reset email to {user.email}: {e}")
            return False
    
    @staticmethod
    def send_welcome_email(user: User) -> bool:
        """
        Send welcome email to new user.
        
        Args:
            user: User instance
            
        Returns:
            True if queued successfully, False otherwise
        """
        try:
            site_url = getattr(settings, 'SITE_URL', 'https://bunoraa.com')
            
            context = {
                'user': user,
                'site_name': 'Bunoraa',
                'dashboard_url': f"{site_url}/account/",
            }
            
            html_content = render_to_string('emails/welcome.html', context)
            text_content = f'Welcome to Bunoraa, {user.get_short_name()}!'
            
            api_key = EmailServiceIntegration.get_api_key()
            if not api_key:
                return False
            
            message = EmailMessage.objects.create(
                message_id=f"welcome_{user.id}_{secrets.token_hex(4)}@bunoraa.com",
                api_key=api_key,
                to_email=user.email,
                from_email=settings.DEFAULT_FROM_EMAIL,
                from_name='Bunoraa',
                subject='Welcome to Bunoraa!',
                html_content=html_content,
                text_content=text_content,
                status='queued',
                email_type='transactional',
                metadata={
                    'user_id': str(user.id),
                    'email_type': 'welcome',
                }
            )
            
            logger.info(f"Welcome email queued for {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send welcome email to {user.email}: {e}")
            return False
    
    @staticmethod
    def send_account_deleted_email(user: User) -> bool:
        """
        Send account deletion confirmation email.
        
        Args:
            user: User instance (will be deleted shortly after)
            
        Returns:
            True if queued successfully, False otherwise
        """
        try:
            context = {
                'user': user,
                'site_name': 'Bunoraa',
            }
            
            html_content = render_to_string('emails/account_deleted.html', context)
            text_content = 'Your Bunoraa account has been deleted.'
            
            api_key = EmailServiceIntegration.get_api_key()
            if not api_key:
                return False
            
            message = EmailMessage.objects.create(
                message_id=f"deleted_{user.id}_{secrets.token_hex(4)}@bunoraa.com",
                api_key=api_key,
                to_email=user.email,
                from_email=settings.DEFAULT_FROM_EMAIL,
                from_name='Bunoraa',
                subject='Your Bunoraa Account Has Been Deleted',
                html_content=html_content,
                text_content=text_content,
                status='queued',
                email_type='transactional',
                metadata={
                    'user_id': str(user.id),
                    'email_type': 'account_deleted',
                }
            )
            
            logger.info(f"Account deletion email queued for {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send account deletion email to {user.email}: {e}")
            return False
    
    @staticmethod
    def send_email_change_verification(user: User, new_email: str, token: str) -> bool:
        """
        Send verification email for email address change.
        
        Args:
            user: User instance
            new_email: New email address to verify
            token: Verification token
            
        Returns:
            True if queued successfully, False otherwise
        """
        try:
            site_url = getattr(settings, 'SITE_URL', 'https://bunoraa.com')
            verification_url = f"{site_url}/account/verify-new-email/{token}/"
            
            context = {
                'user': user,
                'new_email': new_email,
                'verification_url': verification_url,
                'site_name': 'Bunoraa',
            }
            
            html_content = render_to_string('emails/verify_new_email.html', context)
            text_content = f'Verify your new email: {verification_url}'
            
            api_key = EmailServiceIntegration.get_api_key()
            if not api_key:
                return False
            
            message = EmailMessage.objects.create(
                message_id=f"email_verify_{user.id}_{secrets.token_hex(4)}@bunoraa.com",
                api_key=api_key,
                to_email=new_email,
                from_email=settings.DEFAULT_FROM_EMAIL,
                from_name='Bunoraa',
                subject='Verify Your New Email Address',
                html_content=html_content,
                text_content=text_content,
                status='queued',
                email_type='transactional',
                metadata={
                    'user_id': str(user.id),
                    'email_type': 'email_change_verification',
                    'new_email': new_email,
                    'token': token,
                }
            )
            
            logger.info(f"Email verification email queued for {new_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email change verification to {new_email}: {e}")
            return False
