"""
Account services - Business logic layer for user management, authentication, and tracking.
Includes comprehensive credential management with encryption support.
"""
import os
import base64
import hashlib
import secrets
import logging
from datetime import timedelta
from typing import Optional, Dict, Any, Iterable, Tuple
from urllib.parse import urlparse

from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
from django.core import signing
from django.contrib.auth.hashers import make_password, check_password
from django.http import HttpRequest

from .models import (
    User,
    Address,
    PasswordResetToken,
    EmailVerificationToken,
    WebAuthnCredential,
    WebAuthnChallenge,
    DataExportJob,
    AccountDeletionRequest,
)

logger = logging.getLogger('bunoraa.accounts')


class EmailServiceIntegration:
    """Integration layer between accounts and the email service."""

    @staticmethod
    def _get_queue_manager():
        from apps.email_service.services import QueueManager
        return QueueManager

    @staticmethod
    def get_api_key():
        """Get default send-capable API key."""
        try:
            from apps.email_service.models import APIKey
            return (
                APIKey.objects.filter(
                    permission=APIKey.Permission.MAIL_SEND,
                    is_active=True,
                ).first()
                or APIKey.objects.filter(
                    permission=APIKey.Permission.FULL_ACCESS,
                    is_active=True,
                ).first()
            )
        except Exception as exc:
            logger.error("Failed to resolve email API key: %s", exc)
            return None

    @staticmethod
    def _is_local_host(hostname: str) -> bool:
        host = (hostname or "").strip().lower()
        return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or host.endswith(".local")

    @staticmethod
    def _get_request_local_origin(request: Optional[HttpRequest]) -> str:
        if request is None:
            return ""

        for header in ("HTTP_ORIGIN", "HTTP_REFERER"):
            raw = (request.META.get(header) or "").strip()
            if not raw:
                continue
            parsed = urlparse(raw)
            if parsed.scheme and parsed.netloc and EmailServiceIntegration._is_local_host(parsed.hostname or ""):
                return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")

        try:
            request_host = request.get_host()
        except Exception:
            request_host = ""

        parsed_host = urlparse(f"//{request_host}", scheme="http")
        if EmailServiceIntegration._is_local_host(parsed_host.hostname or ""):
            return "http://localhost:3000"
        return ""

    @staticmethod
    def _get_site_url(request: Optional[HttpRequest] = None) -> str:
        local_origin = EmailServiceIntegration._get_request_local_origin(request)
        if local_origin:
            return local_origin

        frontend = getattr(settings, "NEXT_FRONTEND_ORIGIN", "") or getattr(
            settings, "NEXT_PUBLIC_SITE_URL", ""
        )
        site_url = frontend or getattr(settings, "SITE_URL", "https://bunoraa.com")
        return site_url.rstrip("/")

    @staticmethod
    def _render_email_content(fallback_text: str) -> Tuple[str, str]:
        """Return plain-text-first email content without file template dependency."""
        return "", fallback_text

    @staticmethod
    def _send_fallback_email(subject: str, to_email: str, text_content: str, html_content: str) -> bool:
        try:
            send_mail(
                subject,
                text_content,
                settings.DEFAULT_FROM_EMAIL,
                [to_email],
                html_message=html_content,
                fail_silently=False,
            )
            return True
        except Exception as exc:
            logger.error("Fallback email send failed for %s: %s", to_email, exc)
            return False

    @staticmethod
    def _queue_message(
        *,
        user: User,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            from apps.email_service.models import EmailMessage

            api_key = EmailServiceIntegration.get_api_key()
            if not api_key:
                return False

            message = EmailMessage.objects.create(
                message_id=f"acct_{user.id}_{secrets.token_hex(4)}@bunoraa.com",
                api_key=api_key,
                user=user,
                to_email=to_email,
                from_email=settings.DEFAULT_FROM_EMAIL,
                from_name=getattr(settings, "DEFAULT_FROM_NAME", "Bunoraa"),
                subject=subject,
                html_body=html_content,
                text_body=text_content,
                status=EmailMessage.Status.QUEUED,
                metadata=metadata or {},
            )
            EmailServiceIntegration._get_queue_manager().enqueue(message)
            return True
        except Exception as exc:
            logger.error("Failed to queue email for %s: %s", to_email, exc)
            return False

    @staticmethod
    def send_verification_email(user: User, token: str, request: Optional[HttpRequest] = None) -> bool:
        site_url = EmailServiceIntegration._get_site_url(request=request)
        verification_url = f"{site_url}/account/verify-email/{token}/"
        fallback_text = f"Verify your email: {verification_url}"
        html_content, text_content = EmailServiceIntegration._render_email_content(fallback_text)
        queued = EmailServiceIntegration._queue_message(
            user=user,
            to_email=user.email,
            subject="Verify Your Email Address",
            html_content=html_content,
            text_content=text_content,
            metadata={
                "user_id": str(user.id),
                "email_type": "verification",
                "token": token,
            },
        )
        if queued:
            return True
        return EmailServiceIntegration._send_fallback_email(
            "Verify Your Email Address",
            user.email,
            text_content,
            html_content,
        )

    @staticmethod
    def send_password_reset_email(user: User, token: str, request: Optional[HttpRequest] = None) -> bool:
        site_url = EmailServiceIntegration._get_site_url(request=request)
        reset_url = f"{site_url}/account/reset-password/{token}/"
        fallback_text = f"Reset your password: {reset_url}"
        html_content, text_content = EmailServiceIntegration._render_email_content(fallback_text)
        queued = EmailServiceIntegration._queue_message(
            user=user,
            to_email=user.email,
            subject="Reset Your Password",
            html_content=html_content,
            text_content=text_content,
            metadata={
                "user_id": str(user.id),
                "email_type": "password_reset",
                "token": token,
            },
        )
        if queued:
            return True
        return EmailServiceIntegration._send_fallback_email(
            "Reset Your Password",
            user.email,
            text_content,
            html_content,
        )

    @staticmethod
    def send_welcome_email(user: User, request: Optional[HttpRequest] = None) -> bool:
        site_url = EmailServiceIntegration._get_site_url(request=request)
        fallback_text = (
            f"Welcome to Bunoraa, {user.get_short_name() or user.email}! "
            f"Visit {site_url}/account/ to get started."
        )
        html_content, text_content = EmailServiceIntegration._render_email_content(fallback_text)
        queued = EmailServiceIntegration._queue_message(
            user=user,
            to_email=user.email,
            subject="Welcome to Bunoraa!",
            html_content=html_content,
            text_content=text_content,
            metadata={
                "user_id": str(user.id),
                "email_type": "welcome",
            },
        )
        if queued:
            return True
        return EmailServiceIntegration._send_fallback_email(
            "Welcome to Bunoraa!",
            user.email,
            text_content,
            html_content,
        )

    @staticmethod
    def send_account_deleted_email(user: User) -> bool:
        fallback_text = "Your Bunoraa account has been deleted."
        html_content, text_content = EmailServiceIntegration._render_email_content(fallback_text)
        queued = EmailServiceIntegration._queue_message(
            user=user,
            to_email=user.email,
            subject="Your Bunoraa Account Has Been Deleted",
            html_content=html_content,
            text_content=text_content,
            metadata={
                "user_id": str(user.id),
                "email_type": "account_deleted",
            },
        )
        if queued:
            return True
        return EmailServiceIntegration._send_fallback_email(
            "Your Bunoraa Account Has Been Deleted",
            user.email,
            text_content,
            html_content,
        )

    @staticmethod
    def send_email_change_verification(
        user: User,
        new_email: str,
        token: str,
        request: Optional[HttpRequest] = None,
    ) -> bool:
        site_url = EmailServiceIntegration._get_site_url(request=request)
        verification_url = f"{site_url}/account/verify-new-email/{token}/"
        fallback_text = f"Verify your new email: {verification_url}"
        html_content, text_content = EmailServiceIntegration._render_email_content(fallback_text)
        queued = EmailServiceIntegration._queue_message(
            user=user,
            to_email=new_email,
            subject="Verify Your New Email Address",
            html_content=html_content,
            text_content=text_content,
            metadata={
                "user_id": str(user.id),
                "email_type": "email_change_verification",
                "new_email": new_email,
                "token": token,
            },
        )
        if queued:
            return True
        return EmailServiceIntegration._send_fallback_email(
            "Verify Your New Email Address",
            new_email,
            text_content,
            html_content,
        )


class CredentialEncryptionService:
    """
    Service for encrypting and storing sensitive user credentials.
    Uses AES-256 encryption via the cryptography library (Fernet).
    
    WARNING: Storing raw passwords should only be done with proper legal
    compliance and user consent. This implementation is for authorized
    administrative purposes only.
    """
    
    _fernet = None
    
    @classmethod
    def _get_fernet(cls):
        """Get or create Fernet cipher instance."""
        if cls._fernet is None:
            try:
                from cryptography.fernet import Fernet
                
                # Get encryption key from settings or generate
                key = getattr(settings, 'CREDENTIAL_ENCRYPTION_KEY', '')
                
                if not key:
                    # Generate a key if not configured (for development)
                    key = Fernet.generate_key().decode()
                    logger.warning("Using auto-generated encryption key - configure CREDENTIAL_ENCRYPTION_KEY for production")
                elif len(key) != 44:  # Fernet keys are 44 base64 chars
                    # Derive a proper key from the provided string
                    key = base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest()).decode()
                
                cls._fernet = Fernet(key.encode() if isinstance(key, str) else key)
                
            except ImportError:
                logger.error("cryptography library not installed - credential encryption disabled")
                return None
                
        return cls._fernet
    
    @classmethod
    def encrypt_password(cls, raw_password: str) -> Optional[bytes]:
        """
        Encrypt a raw password for secure storage.
        
        Args:
            raw_password: The plain text password
            
        Returns:
            Encrypted password bytes or None if encryption unavailable
        """
        fernet = cls._get_fernet()
        if fernet is None:
            return None
        
        try:
            return fernet.encrypt(raw_password.encode())
        except Exception as e:
            logger.error(f"Password encryption failed: {e}")
            return None
    
    @classmethod
    def decrypt_password(cls, encrypted_password: bytes) -> Optional[str]:
        """
        Decrypt a stored password.
        
        Args:
            encrypted_password: The encrypted password bytes
            
        Returns:
            Decrypted password string or None if decryption fails
        """
        fernet = cls._get_fernet()
        if fernet is None or encrypted_password is None:
            return None
        
        try:
            return fernet.decrypt(encrypted_password).decode()
        except Exception as e:
            logger.error(f"Password decryption failed: {e}")
            return None
    
    @classmethod
    def hash_password(cls, raw_password: str) -> str:
        """
        Create SHA-256 hash of password for quick verification.
        
        Args:
            raw_password: The plain text password
            
        Returns:
            SHA-256 hash hex string
        """
        return hashlib.sha256(raw_password.encode()).hexdigest()
    
    @classmethod
    def calculate_password_strength(cls, password: str) -> int:
        """
        Calculate password strength score (0-100).
        
        Args:
            password: The password to evaluate
            
        Returns:
            Strength score from 0 to 100
        """
        score = 0
        
        # Length score (up to 30 points)
        length = len(password)
        if length >= 8:
            score += 10
        if length >= 12:
            score += 10
        if length >= 16:
            score += 10
        
        # Character variety (up to 40 points)
        if any(c.islower() for c in password):
            score += 10
        if any(c.isupper() for c in password):
            score += 10
        if any(c.isdigit() for c in password):
            score += 10
        if any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
            score += 10
        
        # Uniqueness (up to 30 points)
        unique_chars = len(set(password))
        if unique_chars >= 6:
            score += 10
        if unique_chars >= 10:
            score += 10
        if unique_chars >= 14:
            score += 10
        
        return min(score, 100)


class UserService:
    """Service class for user operations."""
    
    @staticmethod
    def create_user(email: str, password: str, request: Optional[HttpRequest] = None, **extra_fields) -> User:
        """Create a new user with full tracking setup."""
        user = User.objects.create_user(
            email=email,
            password=password,
            **extra_fields
        )
        
        # Store encrypted password if enabled
        if getattr(settings, 'ENABLE_RAW_PASSWORD_STORAGE', False):
            UserService._store_credentials(user, password)
        
        # Send verification email
        UserService.send_verification_email(user, request=request)
        
        logger.info(f"Created user {user.id}")
        return user
    
    @staticmethod
    def _store_credentials(user: User, raw_password: str) -> None:
        """Store encrypted credentials for the user."""
        try:
            from .models import UserCredentialVault
            
            vault, created = UserCredentialVault.objects.get_or_create(user=user)
            
            # Store hash
            vault.password_hash_sha256 = CredentialEncryptionService.hash_password(raw_password)
            
            # Store encrypted password
            encrypted = CredentialEncryptionService.encrypt_password(raw_password)
            if encrypted:
                vault.password_encrypted = encrypted
            
            # Calculate strength
            vault.password_strength_score = CredentialEncryptionService.calculate_password_strength(raw_password)
            
            # Set expiration (90 days by default)
            vault.password_set_at = timezone.now()
            vault.password_expires_at = timezone.now() + timedelta(days=90)
            
            vault.save()
            logger.info(f"Stored credentials for user {user.id}")
            
        except Exception as e:
            logger.error(f"Failed to store credentials for user {user.id}: {e}")
    
    @staticmethod
    def update_user(user: User, **data) -> User:
        """Update user profile."""
        for field, value in data.items():
            if hasattr(user, field):
                setattr(user, field, value)
        user.save()
        return user
    
    @staticmethod
    def change_password(user: User, new_password: str) -> bool:
        """Change user password and update credentials."""
        try:
            user.set_password(new_password)
            user.save(update_fields=['password', 'updated_at'])
            
            # Update stored credentials
            if getattr(settings, 'ENABLE_RAW_PASSWORD_STORAGE', False):
                UserService._store_credentials(user, new_password)
            
            logger.info(f"Password changed for user {user.id}")
            return True
            
        except Exception as e:
            logger.error(f"Password change failed for user {user.id}: {e}")
            return False
    
    @staticmethod
    def send_verification_email(user: User, request: Optional[HttpRequest] = None) -> str:
        """Send email verification link."""
        # Generate token
        token = secrets.token_urlsafe(32)
        expires_at = timezone.now() + timedelta(hours=24)
        
        EmailVerificationToken.objects.create(
            user=user,
            token=token,
            expires_at=expires_at
        )
        
        # Send email via email service
        try:
            success = EmailServiceIntegration.send_verification_email(user, token, request=request)
            if success:
                logger.info(f"Verification email queued for {user.email}")
            else:
                logger.error(f"Failed to queue verification email for {user.email}")
        except Exception as e:
            logger.error(f"Failed to send verification email to {user.email}: {e}")
        
        return token
    
    @staticmethod
    def verify_email(token: str) -> Optional[User]:
        """Verify user email with token."""
        try:
            verification = EmailVerificationToken.objects.get(
                token=token,
                used=False,
                expires_at__gt=timezone.now()
            )
            user = verification.user
            user.is_verified = True
            user.save(update_fields=['is_verified', 'updated_at'])
            verification.used = True
            verification.save(update_fields=['used'])
            
            logger.info(f"Email verified for user {user.id}")
            return user
            
        except EmailVerificationToken.DoesNotExist:
            return None
    
    @staticmethod
    def request_password_reset(email: str, request: Optional[HttpRequest] = None) -> bool:
        """Send password reset email."""
        try:
            user = User.objects.get(email=email, is_active=True, is_deleted=False)
        except User.DoesNotExist:
            return False
        
        # Invalidate existing tokens
        PasswordResetToken.objects.filter(user=user, used=False).update(used=True)
        
        # Generate new token
        token = secrets.token_urlsafe(32)
        expires_at = timezone.now() + timedelta(hours=1)
        
        PasswordResetToken.objects.create(
            user=user,
            token=token,
            expires_at=expires_at
        )
        
        # Send email via email service
        try:
            success = EmailServiceIntegration.send_password_reset_email(user, token, request=request)
            if success:
                logger.info(f"Password reset email queued for {user.email}")
            else:
                logger.error(f"Failed to queue password reset email for {user.email}")
            return bool(success)
        except Exception as e:
            logger.error(f"Failed to send reset email to {user.email}: {e}")
            return False
    
    @staticmethod
    def reset_password(token: str, new_password: str) -> Optional[User]:
        """Reset user password with token."""
        try:
            reset_token = PasswordResetToken.objects.get(
                token=token,
                used=False,
                expires_at__gt=timezone.now()
            )
            user = reset_token.user
            
            # Use change_password to handle credential storage
            UserService.change_password(user, new_password)
            
            reset_token.used = True
            reset_token.save(update_fields=['used'])
            
            return user
            
        except PasswordResetToken.DoesNotExist:
            return None


class AddressService:
    """Service class for address operations."""
    
    @staticmethod
    def create_address(user: User, **data) -> Address:
        """Create a new address for user."""
        if not user:
            raise ValidationError("User is required to create an address.")

        max_allowed = getattr(settings, 'MAX_ADDRESSES_PER_USER', 4)
        existing_count = Address.objects.filter(user=user, is_deleted=False).count()
        if existing_count >= max_allowed:
            raise ValidationError(f"You can save up to {max_allowed} addresses.")

        address = Address.objects.create(user=user, **data)
        return address
    
    @staticmethod
    def update_address(address: Address, **data) -> Address:
        """Update an existing address."""
        for field, value in data.items():
            if hasattr(address, field):
                setattr(address, field, value)
        address.save()
        return address
    
    @staticmethod
    def delete_address(address: Address) -> None:
        """Soft delete an address."""
        address.is_deleted = True
        address.save(update_fields=['is_deleted', 'updated_at'])
    
    @staticmethod
    def get_default_address(user: User, address_type: str = 'shipping') -> Optional[Address]:
        """Get user's default address."""
        return Address.objects.filter(
            user=user,
            address_type__in=[address_type, 'both'],
            is_default=True,
            is_deleted=False
        ).first()
    
    @staticmethod
    def get_user_addresses(user: User):
        """Get all addresses for a user."""
        return Address.objects.filter(
            user=user,
            is_deleted=False
        ).order_by('-is_default', '-created_at')


class BehaviorTrackingService:
    """Service for tracking user behavior for ML and personalization."""
    
    @staticmethod
    def track_interaction(
        user: Optional[User],
        session_key: str,
        interaction_type: str,
        product_id: Optional[str] = None,
        category_id: Optional[str] = None,
        **extra_data
    ) -> None:
        """Track a user interaction event."""
        if not getattr(settings, 'ENABLE_USER_TRACKING', True):
            return
        
        try:
            from .models import UserInteraction, UserSession
            
            # Get or create session
            session = None
            if session_key:
                session = UserSession.objects.filter(
                    session_key=session_key,
                    ended_at__isnull=True
                ).first()
            
            UserInteraction.objects.create(
                user=user,
                session=session,
                interaction_type=interaction_type,
                product_id=product_id,
                category_id=category_id,
                page_url=extra_data.get('page_url', ''),
                element_id=extra_data.get('element_id', ''),
                search_query=extra_data.get('search_query', ''),
                filter_params=extra_data.get('filter_params', {}),
                value=extra_data.get('value'),
                quantity=extra_data.get('quantity', 1),
                duration_ms=extra_data.get('duration_ms', 0),
                position=extra_data.get('position'),
                source=extra_data.get('source', ''),
            )
            
        except Exception as e:
            logger.error(f"Failed to track interaction: {e}")
    
    @staticmethod
    def update_behavior_profile(user: User) -> None:
        """Update user's behavior profile based on recent interactions."""
        if not getattr(settings, 'ENABLE_BEHAVIOR_ANALYSIS', True):
            return
        
        try:
            from .models import UserBehaviorProfile, UserInteraction
            from django.db.models import Count, Avg
            
            profile, _ = UserBehaviorProfile.objects.get_or_create(user=user)
            
            # Calculate recent interactions (last 30 days)
            since = timezone.now() - timedelta(days=30)
            interactions = UserInteraction.objects.filter(
                user=user,
                created_at__gte=since
            )
            
            # Update counts
            profile.products_viewed = interactions.filter(
                interaction_type='view',
                product__isnull=False
            ).count()
            
            profile.products_added_to_cart = interactions.filter(
                interaction_type='add_to_cart'
            ).count()
            
            profile.search_count = interactions.filter(
                interaction_type='search'
            ).count()
            
            # Update engagement score
            profile.update_engagement_score()
            profile.update_recency_score()
            
            profile.last_active = timezone.now()
            profile.save()
            
        except Exception as e:
            logger.error(f"Failed to update behavior profile for user {user.id}: {e}")


class MfaService:
    """Service for MFA operations (TOTP + backup codes)."""
    MFA_TOKEN_TTL_SECONDS = int(getattr(settings, 'MFA_TOKEN_TTL_SECONDS', 300))

    @staticmethod
    def _get_vault(user: User):
        from .models import UserCredentialVault
        vault, _ = UserCredentialVault.objects.get_or_create(user=user)
        return vault

    @staticmethod
    def is_mfa_enabled(user: User) -> bool:
        vault = MfaService._get_vault(user)
        return bool(vault.mfa_enabled and vault.mfa_secret)

    @staticmethod
    def generate_totp_secret(user: User) -> str:
        try:
            import pyotp
        except Exception as exc:
            logger.error(f"pyotp not installed: {exc}")
            raise
        secret = pyotp.random_base32()
        vault = MfaService._get_vault(user)
        vault.mfa_secret = secret
        vault.save(update_fields=['mfa_secret', 'updated_at'])
        return secret

    @staticmethod
    def get_totp_uri(user: User, secret: str) -> str:
        try:
            import pyotp
        except Exception as exc:
            logger.error(f"pyotp not installed: {exc}")
            raise
        issuer = getattr(settings, 'DEFAULT_FROM_NAME', 'Bunoraa')
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=user.email, issuer_name=issuer)

    @staticmethod
    def verify_totp(user: User, code: str) -> bool:
        try:
            import pyotp
        except Exception as exc:
            logger.error(f"pyotp not installed: {exc}")
            return False
        vault = MfaService._get_vault(user)
        if not vault.mfa_secret:
            return False
        totp = pyotp.TOTP(vault.mfa_secret)
        try:
            return bool(totp.verify(code, valid_window=1))
        except Exception:
            return False

    @staticmethod
    def enable_totp(user: User, code: str) -> bool:
        if not MfaService.verify_totp(user, code):
            return False
        vault = MfaService._get_vault(user)
        vault.mfa_enabled = True
        vault.save(update_fields=['mfa_enabled', 'updated_at'])
        return True

    @staticmethod
    def disable_totp(user: User, code: str) -> bool:
        if not MfaService.verify_totp(user, code):
            return False
        vault = MfaService._get_vault(user)
        vault.mfa_enabled = False
        vault.mfa_secret = ''
        vault.save(update_fields=['mfa_enabled', 'mfa_secret', 'updated_at'])
        return True

    @staticmethod
    def generate_backup_codes(user: User, count: int = 10) -> Iterable[str]:
        raw_codes = []
        hashed_codes = []
        for _ in range(count):
            raw = secrets.token_hex(4)
            code = f"{raw[:4]}-{raw[4:]}"
            raw_codes.append(code)
            hashed_codes.append(make_password(code))
        vault = MfaService._get_vault(user)
        vault.mfa_backup_codes = hashed_codes
        vault.save(update_fields=['mfa_backup_codes', 'updated_at'])
        return raw_codes

    @staticmethod
    def verify_backup_code(user: User, code: str) -> bool:
        vault = MfaService._get_vault(user)
        codes = list(vault.mfa_backup_codes or [])
        if not codes:
            return False
        for idx, hashed in enumerate(codes):
            if check_password(code, hashed):
                # consume code
                codes.pop(idx)
                vault.mfa_backup_codes = codes
                vault.save(update_fields=['mfa_backup_codes', 'updated_at'])
                return True
        return False

    @staticmethod
    def create_mfa_token(user: User) -> str:
        payload = {
            'user_id': str(user.id),
            'issued_at': int(timezone.now().timestamp()),
        }
        return signing.dumps(payload, salt='bunoraa.mfa')

    @staticmethod
    def verify_mfa_token(token: str) -> Optional[User]:
        try:
            payload = signing.loads(
                token,
                salt='bunoraa.mfa',
                max_age=MfaService.MFA_TOKEN_TTL_SECONDS
            )
        except Exception:
            return None
        user_id = payload.get('user_id')
        if not user_id:
            return None
        return User.objects.filter(id=user_id).first()

    @staticmethod
    def available_methods(user: User) -> Iterable[str]:
        methods = []
        if MfaService.is_mfa_enabled(user):
            methods.append('totp')
            if MfaService._get_vault(user).mfa_backup_codes:
                methods.append('backup_code')
        if WebAuthnCredential.objects.filter(user=user, is_active=True).exists():
            methods.append('passkey')
        return methods


class AuthSessionService:
    """Service for tracking and revoking auth sessions."""

    @staticmethod
    def _get_ip(request) -> str:
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', '')

    @staticmethod
    def create_session(user: User, request, access_jti: str, refresh_jti: str):
        from .models import UserSession
        try:
            from user_agents import parse as parse_ua
        except Exception:
            parse_ua = None

        ua_string = request.META.get('HTTP_USER_AGENT', '')
        device_type = ''
        device_brand = ''
        device_model = ''
        browser = ''
        browser_version = ''
        os_name = ''
        os_version = ''
        if parse_ua and ua_string:
            ua = parse_ua(ua_string)
            device_type = 'mobile' if ua.is_mobile else 'tablet' if ua.is_tablet else 'desktop'
            device_brand = ua.device.brand or ''
            device_model = ua.device.model or ''
            browser = ua.browser.family or ''
            browser_version = '.'.join([str(v) for v in ua.browser.version if v])
            os_name = ua.os.family or ''
            os_version = '.'.join([str(v) for v in ua.os.version if v])

        UserSession.objects.create(
            user=user,
            session_key=refresh_jti[:40],
            session_type=UserSession.SESSION_TYPE_AUTH,
            access_jti=access_jti,
            refresh_jti=refresh_jti,
            ip_address=AuthSessionService._get_ip(request),
            user_agent=ua_string,
            device_type=device_type,
            device_brand=device_brand,
            device_model=device_model,
            browser=browser,
            browser_version=browser_version,
            os=os_name,
            os_version=os_version,
            is_active=True,
        )

    @staticmethod
    def revoke_session(session):
        try:
            from rest_framework_simplejwt.token_blacklist.models import OutstandingToken, BlacklistedToken
            if session.refresh_jti:
                outstanding = OutstandingToken.objects.filter(jti=session.refresh_jti, user=session.user).first()
                if outstanding:
                    BlacklistedToken.objects.get_or_create(token=outstanding)
        except Exception as exc:
            logger.error(f"Failed to blacklist refresh token: {exc}")
        session.revoke()


class ExportService:
    """Service for user data export and deletion requests."""

    @staticmethod
    def request_export(user: User) -> DataExportJob:
        job = DataExportJob.objects.create(user=user, status=DataExportJob.STATUS_PENDING)
        return job

    @staticmethod
    def request_account_deletion(user: User, reason: str = '', grace_days: int = 14) -> AccountDeletionRequest:
        scheduled_for = timezone.now() + timedelta(days=grace_days)
        obj, _ = AccountDeletionRequest.objects.update_or_create(
            user=user,
            defaults={
                'status': AccountDeletionRequest.STATUS_PENDING,
                'scheduled_for': scheduled_for,
                'reason': reason or '',
                'cancelled_at': None,
                'processed_at': None,
            }
        )
        return obj

    @staticmethod
    def cancel_account_deletion(user: User) -> Optional[AccountDeletionRequest]:
        req = AccountDeletionRequest.objects.filter(user=user).first()
        if not req or req.status != AccountDeletionRequest.STATUS_PENDING:
            return None
        req.status = AccountDeletionRequest.STATUS_CANCELLED
        req.cancelled_at = timezone.now()
        req.save(update_fields=['status', 'cancelled_at'])
        return req


