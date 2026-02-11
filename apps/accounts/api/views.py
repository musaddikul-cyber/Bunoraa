"""
Account API views
"""
import json
from datetime import timedelta, datetime, timezone as dt_timezone
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from django.contrib.auth import get_user_model
from django.conf import settings
from django.http import FileResponse
from django.utils import timezone
from rest_framework_simplejwt.tokens import RefreshToken, AccessToken
from rest_framework_simplejwt.token_blacklist.models import OutstandingToken
from ..models import (
    Address,
    WebAuthnCredential,
    WebAuthnChallenge,
    DataExportJob,
    AccountDeletionRequest,
)
from ..behavior_models import UserPreferences, UserSession
from ..services import UserService, AddressService, MfaService, AuthSessionService, ExportService
from .serializers import (
    UserSerializer,
    UserRegistrationSerializer,
    UserUpdateSerializer,
    PasswordChangeSerializer,
    PasswordResetRequestSerializer,
    PasswordResetSerializer,
    AddressSerializer,
    AddressCreateSerializer,
    UserPreferencesSerializer,
    UserSessionSerializer,
    MfaVerifySerializer,
    TotpVerifySerializer,
    BackupCodeRegenerateSerializer,
    WebAuthnCredentialSerializer,
    DataExportJobSerializer,
    AccountDeletionStatusSerializer,
)

User = get_user_model()


class RegisterView(APIView):
    """User registration endpoint."""
    permission_classes = [AllowAny]
    
    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({
                'success': True,
                'message': 'Account created successfully. Please verify your email.',
                'data': UserSerializer(user).data,
                'meta': None
            }, status=status.HTTP_201_CREATED)
        return Response({
            'success': False,
            'message': 'Registration failed.',
            'data': None,
            'meta': {'errors': serializer.errors}
        }, status=status.HTTP_400_BAD_REQUEST)


class ProfileView(APIView):
    """User profile endpoint."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response({
            'success': True,
            'message': 'Profile retrieved successfully.',
            'data': serializer.data,
            'meta': None
        })
    
    def patch(self, request):
        serializer = UserUpdateSerializer(request.user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response({
                'success': True,
                'message': 'Profile updated successfully.',
                'data': UserSerializer(request.user).data,
                'meta': None
            })
        return Response({
            'success': False,
            'message': 'Update failed.',
            'data': None,
            'meta': {'errors': serializer.errors}
        }, status=status.HTTP_400_BAD_REQUEST)


class AvatarUploadView(APIView):
    """Upload and update user avatar."""
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        file_obj = request.FILES.get('avatar')
        if not file_obj:
            return Response({
                'success': False,
                'message': 'No avatar file provided.',
                'data': None,
                'meta': None
            }, status=status.HTTP_400_BAD_REQUEST)

        user = request.user
        user.avatar = file_obj
        user.save(update_fields=['avatar'])

        return Response({
            'success': True,
            'message': 'Avatar updated successfully.',
            'data': {'avatar': user.avatar.url if user.avatar else None},
            'meta': None
        })


class PasswordChangeView(APIView):
    """Password change endpoint."""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        serializer = PasswordChangeSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            request.user.set_password(serializer.validated_data['new_password'])
            request.user.save()
            return Response({
                'success': True,
                'message': 'Password changed successfully.',
                'data': None,
                'meta': None
            })
        return Response({
            'success': False,
            'message': 'Password change failed.',
            'data': None,
            'meta': {'errors': serializer.errors}
        }, status=status.HTTP_400_BAD_REQUEST)


class PasswordResetRequestView(APIView):
    """Password reset request endpoint."""
    permission_classes = [AllowAny]
    
    def post(self, request):
        serializer = PasswordResetRequestSerializer(data=request.data)
        if serializer.is_valid():
            UserService.request_password_reset(serializer.validated_data['email'])
            # Always return success to prevent email enumeration
            return Response({
                'success': True,
                'message': 'If an account exists with this email, a reset link will be sent.',
                'data': None,
                'meta': None
            })
        return Response({
            'success': False,
            'message': 'Invalid request.',
            'data': None,
            'meta': {'errors': serializer.errors}
        }, status=status.HTTP_400_BAD_REQUEST)


class PasswordResetView(APIView):
    """Password reset endpoint."""
    permission_classes = [AllowAny]
    
    def post(self, request):
        serializer = PasswordResetSerializer(data=request.data)
        if serializer.is_valid():
            user = UserService.reset_password(
                serializer.validated_data['token'],
                serializer.validated_data['new_password']
            )
            if user:
                return Response({
                    'success': True,
                    'message': 'Password reset successfully.',
                    'data': None,
                    'meta': None
                })
            return Response({
                'success': False,
                'message': 'Invalid or expired reset token.',
                'data': None,
                'meta': None
            }, status=status.HTTP_400_BAD_REQUEST)
        return Response({
            'success': False,
            'message': 'Invalid request.',
            'data': None,
            'meta': {'errors': serializer.errors}
        }, status=status.HTTP_400_BAD_REQUEST)


class EmailVerifyView(APIView):
    """Email verification endpoint."""
    permission_classes = [AllowAny]
    
    def post(self, request):
        token = request.data.get('token')
        if not token:
            return Response({
                'success': False,
                'message': 'Token is required.',
                'data': None,
                'meta': None
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user = UserService.verify_email(token)
        if user:
            return Response({
                'success': True,
                'message': 'Email verified successfully.',
                'data': UserSerializer(user).data,
                'meta': None
            })
        return Response({
            'success': False,
            'message': 'Invalid or expired verification token.',
            'data': None,
            'meta': None
        }, status=status.HTTP_400_BAD_REQUEST)


class ResendVerificationView(APIView):
    """Resend email verification endpoint."""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        if request.user.is_verified:
            return Response({
                'success': False,
                'message': 'Email is already verified.',
                'data': None,
                'meta': None
            }, status=status.HTTP_400_BAD_REQUEST)
        
        UserService.send_verification_email(request.user)
        return Response({
            'success': True,
            'message': 'Verification email sent.',
            'data': None,
            'meta': None
        })


class SocialTokenView(APIView):
    """Exchange a social-authenticated session for JWT tokens."""
    permission_classes = [IsAuthenticated]

    def _issue_tokens(self, request):
        refresh = RefreshToken.for_user(request.user)
        access = refresh.access_token
        expires_at = datetime.fromtimestamp(refresh['exp'], tz=dt_timezone.utc)
        OutstandingToken.objects.get_or_create(
            jti=str(refresh['jti']),
            user=request.user,
            token=str(refresh),
            expires_at=expires_at,
        )
        AuthSessionService.create_session(
            request.user,
            request,
            str(access['jti']),
            str(refresh['jti'])
        )
        return Response({
            'success': True,
            'message': 'Social login successful.',
            'data': {
                'access': str(access),
                'refresh': str(refresh),
                'mfa_required': False,
            },
            'meta': None
        })

    def get(self, request):
        return self._issue_tokens(request)

    def post(self, request):
        return self._issue_tokens(request)


class AddressViewSet(viewsets.ModelViewSet):
    """Address CRUD endpoints."""
    permission_classes = [IsAuthenticated]
    serializer_class = AddressSerializer
    
    def get_queryset(self):
        return Address.objects.filter(
            user=self.request.user,
            is_deleted=False
        ).order_by('-is_default', '-created_at')
    
    def get_serializer_class(self):
        if self.action == 'create':
            return AddressCreateSerializer
        return AddressSerializer
    
    def list(self, request):
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'message': 'Addresses retrieved successfully.',
            'data': serializer.data,
            'meta': {'count': queryset.count()}
        })
    
    def create(self, request):
        serializer = self.get_serializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            address = serializer.save()
            return Response({
                'success': True,
                'message': 'Address created successfully.',
                'data': AddressSerializer(address).data,
                'meta': None
            }, status=status.HTTP_201_CREATED)
        errors = serializer.errors
        error_message = 'Failed to create address.'
        if isinstance(errors, dict):
            non_field = errors.get('non_field_errors')
            if isinstance(non_field, list) and non_field:
                error_message = str(non_field[0])
        return Response({
            'success': False,
            'message': error_message,
            'data': None,
            'meta': {'errors': serializer.errors}
        }, status=status.HTTP_400_BAD_REQUEST)
    
    def retrieve(self, request, pk=None):
        try:
            address = self.get_queryset().get(pk=pk)
            serializer = self.get_serializer(address)
            return Response({
                'success': True,
                'message': 'Address retrieved successfully.',
                'data': serializer.data,
                'meta': None
            })
        except Address.DoesNotExist:
            return Response({
                'success': False,
                'message': 'Address not found.',
                'data': None,
                'meta': None
            }, status=status.HTTP_404_NOT_FOUND)
    
    def update(self, request, pk=None, *args, **kwargs):
        try:
            partial = kwargs.pop("partial", request.method == "PATCH")
            address = self.get_queryset().get(pk=pk)
            serializer = self.get_serializer(address, data=request.data, partial=partial)
            if serializer.is_valid():
                serializer.save()
                return Response({
                    'success': True,
                    'message': 'Address updated successfully.',
                    'data': serializer.data,
                    'meta': None
                })
            return Response({
                'success': False,
                'message': 'Failed to update address.',
                'data': None,
                'meta': {'errors': serializer.errors}
            }, status=status.HTTP_400_BAD_REQUEST)
        except Address.DoesNotExist:
            return Response({
                'success': False,
                'message': 'Address not found.',
                'data': None,
                'meta': None
            }, status=status.HTTP_404_NOT_FOUND)
    
    def destroy(self, request, pk=None):
        try:
            address = self.get_queryset().get(pk=pk)
            AddressService.delete_address(address)
            return Response({
                'success': True,
                'message': 'Address deleted successfully.',
                'data': None,
                'meta': None
            })
        except Address.DoesNotExist:
            return Response({
                'success': False,
                'message': 'Address not found.',
                'data': None,
                'meta': None
            }, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['post'])
    def set_default(self, request, pk=None):
        """Set address as default."""
        try:
            address = self.get_queryset().get(pk=pk)
            address.is_default = True
            address.save()
            return Response({
                'success': True,
                'message': 'Address set as default.',
                'data': AddressSerializer(address).data,
                'meta': None
            })
        except Address.DoesNotExist:
            return Response({
                'success': False,
                'message': 'Address not found.',
                'data': None,
                'meta': None
            }, status=status.HTTP_404_NOT_FOUND)


def _get_webauthn_origin() -> str:
    origin = (getattr(settings, 'NEXT_FRONTEND_ORIGIN', '') or getattr(settings, 'SITE_URL', '')).rstrip('/')
    return origin


def _get_webauthn_rp_id() -> str:
    from urllib.parse import urlparse
    origin = _get_webauthn_origin()
    parsed = urlparse(origin)
    if parsed.hostname:
        return parsed.hostname
    return origin.replace('https://', '').replace('http://', '').split('/')[0]


def _get_access_jti(request):
    auth = request.META.get('HTTP_AUTHORIZATION', '')
    if not auth.startswith('Bearer '):
        return None
    token = auth.split(' ', 1)[1].strip()
    try:
        access = AccessToken(token)
        return str(access['jti'])
    except Exception:
        return None


def _parse_credential_payload(payload):
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return payload
    return payload


class UserPreferencesView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        prefs, _ = UserPreferences.objects.get_or_create(user=request.user)
        serializer = UserPreferencesSerializer(prefs)
        return Response({
            'success': True,
            'message': 'Preferences retrieved successfully.',
            'data': serializer.data,
            'meta': None
        })

    def patch(self, request):
        prefs, _ = UserPreferences.objects.get_or_create(user=request.user)
        serializer = UserPreferencesSerializer(prefs, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response({
                'success': True,
                'message': 'Preferences updated successfully.',
                'data': serializer.data,
                'meta': None
            })
        return Response({
            'success': False,
            'message': 'Failed to update preferences.',
            'data': None,
            'meta': {'errors': serializer.errors}
        }, status=status.HTTP_400_BAD_REQUEST)


class UserSessionView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        retention_days = int(getattr(settings, 'AUTH_SESSION_RETENTION_DAYS', 90))
        cutoff = timezone.now() - timedelta(days=retention_days)
        sessions = UserSession.objects.filter(
            user=request.user,
            session_type=UserSession.SESSION_TYPE_AUTH,
            started_at__gte=cutoff,
        ).order_by('-started_at')
        current_jti = _get_access_jti(request)
        data = []
        for session in sessions:
            serializer = UserSessionSerializer(session)
            payload = serializer.data
            payload['is_current'] = bool(current_jti and session.access_jti == current_jti)
            data.append(payload)
        return Response({
            'success': True,
            'message': 'Sessions retrieved successfully.',
            'data': data,
            'meta': {'count': len(data)}
        })


class RevokeSessionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        session = UserSession.objects.filter(
            id=session_id,
            user=request.user,
            session_type=UserSession.SESSION_TYPE_AUTH
        ).first()
        if not session:
            return Response({
                'success': False,
                'message': 'Session not found.',
                'data': None,
                'meta': None
            }, status=status.HTTP_404_NOT_FOUND)
        AuthSessionService.revoke_session(session)
        return Response({
            'success': True,
            'message': 'Session revoked.',
            'data': None,
            'meta': None
        })


class RevokeOtherSessionsView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        current_jti = _get_access_jti(request)
        sessions = UserSession.objects.filter(
            user=request.user,
            session_type=UserSession.SESSION_TYPE_AUTH,
            is_active=True
        )
        revoked = 0
        for session in sessions:
            if current_jti and session.access_jti == current_jti:
                continue
            AuthSessionService.revoke_session(session)
            revoked += 1
        return Response({
            'success': True,
            'message': f'{revoked} sessions revoked.',
            'data': {'revoked': revoked},
            'meta': None
        })


class MfaStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        methods = list(MfaService.available_methods(request.user))
        vault = MfaService._get_vault(request.user)
        passkey_count = WebAuthnCredential.objects.filter(user=request.user, is_active=True).count()
        return Response({
            'success': True,
            'message': 'MFA status retrieved.',
            'data': {
                'enabled': bool(vault.mfa_enabled),
                'methods': methods,
                'backup_codes_remaining': len(vault.mfa_backup_codes or []),
                'passkey_count': passkey_count,
            },
            'meta': None
        })


class TotpSetupView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        secret = MfaService.generate_totp_secret(request.user)
        uri = MfaService.get_totp_uri(request.user, secret)
        return Response({
            'success': True,
            'message': 'TOTP secret generated.',
            'data': {'secret': secret, 'otpauth_url': uri},
            'meta': None
        })


class TotpVerifyView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = TotpVerifySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        code = serializer.validated_data['code']
        if MfaService.enable_totp(request.user, code):
            codes = MfaService.generate_backup_codes(request.user)
            return Response({
                'success': True,
                'message': 'TOTP enabled.',
                'data': {'backup_codes': codes},
                'meta': None
            })
        return Response({
            'success': False,
            'message': 'Invalid verification code.',
            'data': None,
            'meta': None
        }, status=status.HTTP_400_BAD_REQUEST)


class TotpDisableView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = TotpVerifySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        code = serializer.validated_data['code']
        if MfaService.disable_totp(request.user, code):
            return Response({
                'success': True,
                'message': 'TOTP disabled.',
                'data': None,
                'meta': None
            })
        return Response({
            'success': False,
            'message': 'Invalid code.',
            'data': None,
            'meta': None
        }, status=status.HTTP_400_BAD_REQUEST)


class BackupCodesRegenerateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        codes = list(MfaService.generate_backup_codes(request.user))
        return Response({
            'success': True,
            'message': 'Backup codes generated.',
            'data': {'backup_codes': codes},
            'meta': None
        })


class MfaVerifyView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = MfaVerifySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        token = serializer.validated_data['mfa_token']
        method = serializer.validated_data['method']
        code = serializer.validated_data.get('code') or ''
        credential = serializer.validated_data.get('credential')

        user = MfaService.verify_mfa_token(token)
        if not user:
            return Response({
                'success': False,
                'message': 'Invalid or expired MFA token.',
                'data': None,
                'meta': None
            }, status=status.HTTP_400_BAD_REQUEST)

        verified = False
        if method == 'totp':
            verified = MfaService.verify_totp(user, code)
        elif method == 'backup_code':
            verified = MfaService.verify_backup_code(user, code)
        elif method == 'passkey':
            verified = _verify_webauthn_authentication(user, credential, WebAuthnChallenge.TYPE_MFA)

        if not verified:
            return Response({
                'success': False,
                'message': 'MFA verification failed.',
                'data': None,
                'meta': None
            }, status=status.HTTP_400_BAD_REQUEST)

        refresh = RefreshToken.for_user(user)
        access = refresh.access_token
        expires_at = datetime.fromtimestamp(refresh['exp'], tz=timezone.utc)
        OutstandingToken.objects.get_or_create(
            jti=str(refresh['jti']),
            user=user,
            token=str(refresh),
            expires_at=expires_at,
        )
        AuthSessionService.create_session(user, request, str(access['jti']), str(refresh['jti']))

        return Response({
            'success': True,
            'message': 'MFA verified.',
            'data': {
                'access': str(access),
                'refresh': str(refresh),
                'mfa_required': False,
            },
            'meta': None
        })


def _verify_webauthn_authentication(user, credential_payload, challenge_type):
    try:
        from webauthn import (
            verify_authentication_response,
            base64url_to_bytes,
        )
        from webauthn.helpers.structs import AuthenticationCredential
    except Exception:
        return False

    payload = _parse_credential_payload(credential_payload)
    if not payload:
        return False

    raw_json = json.dumps(payload)
    credential = AuthenticationCredential.parse_raw(raw_json)

    cred_id = payload.get('id') or payload.get('rawId')
    if not cred_id:
        return False
    try:
        cred_id_bytes = base64url_to_bytes(cred_id)
    except Exception:
        return False

    stored = WebAuthnCredential.objects.filter(
        user=user,
        credential_id=cred_id_bytes,
        is_active=True
    ).first()
    if not stored:
        return False

    challenge = WebAuthnChallenge.objects.filter(
        user=user,
        challenge_type=challenge_type,
        consumed=False,
        expires_at__gt=timezone.now()
    ).order_by('-created_at').first()
    if not challenge:
        return False

    try:
        verification = verify_authentication_response(
            credential=credential,
            expected_challenge=base64url_to_bytes(challenge.challenge),
            expected_rp_id=_get_webauthn_rp_id(),
            expected_origin=_get_webauthn_origin(),
            credential_public_key=stored.public_key,
            credential_current_sign_count=stored.sign_count,
            require_user_verification=True,
        )
    except Exception:
        return False

    stored.sign_count = verification.new_sign_count
    stored.last_used_at = timezone.now()
    stored.save(update_fields=['sign_count', 'last_used_at'])
    challenge.consumed = True
    challenge.save(update_fields=['consumed'])
    return True


class WebAuthnRegisterOptionsView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            from webauthn import generate_registration_options, options_to_json
            from webauthn.helpers.structs import (
                AuthenticatorSelectionCriteria,
                UserVerificationRequirement,
                PublicKeyCredentialDescriptor,
            )
        except Exception as exc:
            return Response({
                'success': False,
                'message': f'WebAuthn not available: {exc}',
                'data': None,
                'meta': None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        existing = WebAuthnCredential.objects.filter(user=request.user, is_active=True)
        exclude = [
            PublicKeyCredentialDescriptor(id=cred.credential_id)
            for cred in existing
        ]

        options = generate_registration_options(
            rp_id=_get_webauthn_rp_id(),
            rp_name=getattr(settings, 'DEFAULT_FROM_NAME', 'Bunoraa'),
            user_id=request.user.id.bytes,
            user_name=request.user.email,
            authenticator_selection=AuthenticatorSelectionCriteria(
                user_verification=UserVerificationRequirement.PREFERRED
            ),
            exclude_credentials=exclude,
        )
        options_json = options_to_json(options)
        payload = json.loads(options_json)
        WebAuthnChallenge.objects.create(
            user=request.user,
            challenge=payload.get('challenge'),
            challenge_type=WebAuthnChallenge.TYPE_REGISTER,
            expires_at=timezone.now() + timedelta(minutes=5)
        )
        return Response({
            'success': True,
            'message': 'Registration options generated.',
            'data': payload,
            'meta': None
        })


class WebAuthnRegisterVerifyView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            from webauthn import verify_registration_response, base64url_to_bytes
            from webauthn.helpers.structs import RegistrationCredential
        except Exception as exc:
            return Response({
                'success': False,
                'message': f'WebAuthn not available: {exc}',
                'data': None,
                'meta': None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        payload = _parse_credential_payload(request.data.get('credential'))
        if not payload:
            return Response({
                'success': False,
                'message': 'Credential is required.',
                'data': None,
                'meta': None
            }, status=status.HTTP_400_BAD_REQUEST)

        challenge = WebAuthnChallenge.objects.filter(
            user=request.user,
            challenge_type=WebAuthnChallenge.TYPE_REGISTER,
            consumed=False,
            expires_at__gt=timezone.now()
        ).order_by('-created_at').first()
        if not challenge:
            return Response({
                'success': False,
                'message': 'Registration challenge expired.',
                'data': None,
                'meta': None
            }, status=status.HTTP_400_BAD_REQUEST)

        credential = RegistrationCredential.parse_raw(json.dumps(payload))
        try:
            verification = verify_registration_response(
                credential=credential,
                expected_challenge=base64url_to_bytes(challenge.challenge),
                expected_rp_id=_get_webauthn_rp_id(),
                expected_origin=_get_webauthn_origin(),
                require_user_verification=True,
            )
        except Exception as exc:
            return Response({
                'success': False,
                'message': str(exc),
                'data': None,
                'meta': None
            }, status=status.HTTP_400_BAD_REQUEST)

        WebAuthnCredential.objects.create(
            user=request.user,
            credential_id=verification.credential_id,
            public_key=verification.credential_public_key,
            sign_count=verification.sign_count,
            transports=payload.get('response', {}).get('transports', []),
            nickname=request.data.get('nickname', '')
        )
        challenge.consumed = True
        challenge.save(update_fields=['consumed'])

        return Response({
            'success': True,
            'message': 'Passkey registered.',
            'data': None,
            'meta': None
        })


class WebAuthnLoginOptionsView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            from webauthn import generate_authentication_options, options_to_json
            from webauthn.helpers.structs import PublicKeyCredentialDescriptor, UserVerificationRequirement
        except Exception as exc:
            return Response({
                'success': False,
                'message': f'WebAuthn not available: {exc}',
                'data': None,
                'meta': None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        mfa_token = request.data.get('mfa_token')
        email = request.data.get('email')
        user = None
        challenge_type = WebAuthnChallenge.TYPE_LOGIN
        if mfa_token:
            user = MfaService.verify_mfa_token(mfa_token)
            challenge_type = WebAuthnChallenge.TYPE_MFA
        if not user and email:
            user = User.objects.filter(email__iexact=email).first()
        if not user:
            return Response({
                'success': False,
                'message': 'User not found.',
                'data': None,
                'meta': None
            }, status=status.HTTP_404_NOT_FOUND)

        creds = WebAuthnCredential.objects.filter(user=user, is_active=True)
        allow = [PublicKeyCredentialDescriptor(id=cred.credential_id) for cred in creds]

        options = generate_authentication_options(
            rp_id=_get_webauthn_rp_id(),
            allow_credentials=allow,
            user_verification=UserVerificationRequirement.PREFERRED,
        )
        options_json = options_to_json(options)
        payload = json.loads(options_json)
        WebAuthnChallenge.objects.create(
            user=user,
            challenge=payload.get('challenge'),
            challenge_type=challenge_type,
            expires_at=timezone.now() + timedelta(minutes=5)
        )
        return Response({
            'success': True,
            'message': 'Authentication options generated.',
            'data': payload,
            'meta': None
        })


class WebAuthnLoginVerifyView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        mfa_token = request.data.get('mfa_token')
        email = request.data.get('email')
        credential = request.data.get('credential')

        user = None
        challenge_type = WebAuthnChallenge.TYPE_LOGIN
        if mfa_token:
            user = MfaService.verify_mfa_token(mfa_token)
            challenge_type = WebAuthnChallenge.TYPE_MFA
        if not user and email:
            user = User.objects.filter(email__iexact=email).first()
        if not user:
            return Response({
                'success': False,
                'message': 'User not found.',
                'data': None,
                'meta': None
            }, status=status.HTTP_404_NOT_FOUND)

        verified = _verify_webauthn_authentication(user, credential, challenge_type)
        if not verified:
            return Response({
                'success': False,
                'message': 'Passkey verification failed.',
                'data': None,
                'meta': None
            }, status=status.HTTP_400_BAD_REQUEST)

        if mfa_token:
            return Response({
                'success': True,
                'message': 'Passkey verified.',
                'data': {'verified': True},
                'meta': None
            })

        refresh = RefreshToken.for_user(user)
        access = refresh.access_token
        expires_at = datetime.fromtimestamp(refresh['exp'], tz=timezone.utc)
        OutstandingToken.objects.get_or_create(
            jti=str(refresh['jti']),
            user=user,
            token=str(refresh),
            expires_at=expires_at,
        )
        AuthSessionService.create_session(user, request, str(access['jti']), str(refresh['jti']))
        return Response({
            'success': True,
            'message': 'Passkey login successful.',
            'data': {
                'access': str(access),
                'refresh': str(refresh),
                'mfa_required': False,
            },
            'meta': None
        })


class WebAuthnCredentialListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        creds = WebAuthnCredential.objects.filter(user=request.user, is_active=True)
        serializer = WebAuthnCredentialSerializer(creds, many=True)
        return Response({
            'success': True,
            'message': 'Passkeys retrieved.',
            'data': serializer.data,
            'meta': {'count': len(serializer.data)}
        })

    def delete(self, request, credential_id):
        cred = WebAuthnCredential.objects.filter(user=request.user, id=credential_id).first()
        if not cred:
            return Response({
                'success': False,
                'message': 'Passkey not found.',
                'data': None,
                'meta': None
            }, status=status.HTTP_404_NOT_FOUND)
        cred.is_active = False
        cred.save(update_fields=['is_active'])
        return Response({
            'success': True,
            'message': 'Passkey removed.',
            'data': None,
            'meta': None
        })


class DataExportView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        jobs = DataExportJob.objects.filter(user=request.user).order_by('-requested_at')
        serializer = DataExportJobSerializer(jobs, many=True)
        return Response({
            'success': True,
            'message': 'Export jobs retrieved.',
            'data': serializer.data,
            'meta': {'count': len(serializer.data)}
        })

    def post(self, request):
        job = ExportService.request_export(request.user)
        try:
            from apps.accounts.tasks import generate_data_export
            generate_data_export.delay(str(job.id))
        except Exception:
            pass
        return Response({
            'success': True,
            'message': 'Export requested.',
            'data': DataExportJobSerializer(job).data,
            'meta': None
        }, status=status.HTTP_201_CREATED)


class DataExportDownloadView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, job_id):
        job = DataExportJob.objects.filter(user=request.user, id=job_id).first()
        if not job or not job.file:
            return Response({
                'success': False,
                'message': 'Export not found.',
                'data': None,
                'meta': None
            }, status=status.HTTP_404_NOT_FOUND)
        if job.expires_at and job.expires_at < timezone.now():
            return Response({
                'success': False,
                'message': 'Export expired.',
                'data': None,
                'meta': None
            }, status=status.HTTP_410_GONE)
        response = FileResponse(job.file.open('rb'), as_attachment=True, filename=job.file.name.split('/')[-1])
        return response


def _deletion_status_response(user):
    req = AccountDeletionRequest.objects.filter(user=user).first()
    if not req:
        return Response({
            'success': True,
            'message': 'No deletion request.',
            'data': None,
            'meta': None
        })
    serializer = AccountDeletionStatusSerializer(req)
    return Response({
        'success': True,
        'message': 'Deletion status retrieved.',
        'data': serializer.data,
        'meta': None
    })


class AccountDeletionView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return _deletion_status_response(request.user)

    def post(self, request):
        reason = request.data.get('reason', '')
        req = ExportService.request_account_deletion(request.user, reason=reason)
        serializer = AccountDeletionStatusSerializer(req)
        return Response({
            'success': True,
            'message': 'Deletion requested.',
            'data': serializer.data,
            'meta': None
        })


class AccountDeletionStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return _deletion_status_response(request.user)


class AccountDeletionCancelView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        req = ExportService.cancel_account_deletion(request.user)
        if not req:
            return Response({
                'success': False,
                'message': 'No pending deletion request.',
                'data': None,
                'meta': None
            }, status=status.HTTP_400_BAD_REQUEST)
        serializer = AccountDeletionStatusSerializer(req)
        return Response({
            'success': True,
            'message': 'Deletion cancelled.',
            'data': serializer.data,
            'meta': None
        })
