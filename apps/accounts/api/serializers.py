"""
Account API serializers
"""
from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.conf import settings
from django.contrib.auth.password_validation import validate_password
from ..models import Address, WebAuthnCredential, DataExportJob, AccountDeletionRequest
from ..behavior_models import UserPreferences, UserSession

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    """User serializer for profile data."""
    
    full_name = serializers.CharField(source='get_full_name', read_only=True)
    
    class Meta:
        model = User
        fields = [
            'id', 'email', 'first_name', 'last_name', 'full_name',
            'phone', 'avatar', 'date_of_birth', 'is_verified',
            'newsletter_subscribed', 'is_staff', 'is_superuser',
            'created_at'
        ]
        read_only_fields = ['id', 'email', 'is_verified', 'is_staff', 'is_superuser', 'created_at']


class UserRegistrationSerializer(serializers.ModelSerializer):
    """Serializer for user registration."""
    
    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password],
        style={'input_type': 'password'}
    )
    password_confirm = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'}
    )
    
    class Meta:
        model = User
        fields = ['email', 'password', 'password_confirm', 'first_name', 'last_name', 'phone']
    
    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({
                'password_confirm': 'Passwords do not match.'
            })
        return attrs
    
    def create(self, validated_data):
        validated_data.pop('password_confirm')
        from ..services import UserService
        return UserService.create_user(**validated_data)


class UserUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating user profile."""
    
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'phone', 'avatar', 'date_of_birth', 'newsletter_subscribed']


class PasswordChangeSerializer(serializers.Serializer):
    """Serializer for password change."""
    
    current_password = serializers.CharField(
        required=True,
        style={'input_type': 'password'}
    )
    new_password = serializers.CharField(
        required=True,
        validators=[validate_password],
        style={'input_type': 'password'}
    )
    new_password_confirm = serializers.CharField(
        required=True,
        style={'input_type': 'password'}
    )
    
    def validate_current_password(self, value):
        user = self.context['request'].user
        if not user.check_password(value):
            raise serializers.ValidationError('Current password is incorrect.')
        return value
    
    def validate(self, attrs):
        if attrs['new_password'] != attrs['new_password_confirm']:
            raise serializers.ValidationError({
                'new_password_confirm': 'Passwords do not match.'
            })
        return attrs


class PasswordResetRequestSerializer(serializers.Serializer):
    """Serializer for password reset request."""
    
    email = serializers.EmailField(required=True)


class PasswordResetSerializer(serializers.Serializer):
    """Serializer for password reset."""
    
    token = serializers.CharField(required=True)
    new_password = serializers.CharField(
        required=True,
        validators=[validate_password],
        style={'input_type': 'password'}
    )
    new_password_confirm = serializers.CharField(
        required=True,
        style={'input_type': 'password'}
    )
    
    def validate(self, attrs):
        if attrs['new_password'] != attrs['new_password_confirm']:
            raise serializers.ValidationError({
                'new_password_confirm': 'Passwords do not match.'
            })
        return attrs


class AddressSerializer(serializers.ModelSerializer):
    """Address serializer."""
    
    full_address = serializers.CharField(read_only=True)
    
    class Meta:
        model = Address
        fields = [
            'id', 'address_type', 'full_name', 'phone',
            'address_line_1', 'address_line_2', 'city', 'state',
            'postal_code', 'country', 'is_default', 'full_address',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)


class AddressCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating addresses."""
    
    class Meta:
        model = Address
        fields = [
            'address_type', 'full_name', 'phone',
            'address_line_1', 'address_line_2', 'city', 'state',
            'postal_code', 'country', 'is_default'
        ]

    def validate(self, attrs):
        request = self.context.get('request')
        user = getattr(request, 'user', None)
        if user and user.is_authenticated:
            max_allowed = getattr(settings, 'MAX_ADDRESSES_PER_USER', 4)
            existing_count = Address.objects.filter(user=user, is_deleted=False).count()
            if existing_count >= max_allowed:
                raise serializers.ValidationError(
                    f"You can save up to {max_allowed} addresses."
                )
        return attrs
    
    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        return Address.objects.create(**validated_data)


class UserPreferencesSerializer(serializers.ModelSerializer):
    """Serializer for user preferences."""

    class Meta:
        model = UserPreferences
        fields = [
            'language', 'currency', 'timezone',
            'theme', 'email_notifications', 'sms_notifications',
            'push_notifications', 'notify_order_updates',
            'notify_promotions', 'notify_price_drops',
            'notify_back_in_stock', 'notify_recommendations',
            'allow_tracking', 'share_data_for_ads',
            'reduce_motion', 'high_contrast', 'large_text',
        ]


class UserSessionSerializer(serializers.ModelSerializer):
    """Serializer for user sessions."""
    is_current = serializers.BooleanField(read_only=True)

    class Meta:
        model = UserSession
        fields = [
            'id', 'session_type', 'ip_address', 'device_type', 'device_brand',
            'device_model', 'browser', 'browser_version', 'os', 'os_version',
            'country', 'country_code', 'region', 'city', 'started_at',
            'last_activity', 'revoked_at', 'is_active', 'is_current',
        ]


class MfaVerifySerializer(serializers.Serializer):
    mfa_token = serializers.CharField()
    method = serializers.ChoiceField(choices=['totp', 'backup_code', 'passkey'])
    code = serializers.CharField(required=False, allow_blank=True)
    credential = serializers.JSONField(required=False)


class TotpVerifySerializer(serializers.Serializer):
    code = serializers.CharField()


class BackupCodeRegenerateSerializer(serializers.Serializer):
    regenerate = serializers.BooleanField(default=True)


class WebAuthnCredentialSerializer(serializers.ModelSerializer):
    class Meta:
        model = WebAuthnCredential
        fields = [
            'id', 'nickname', 'transports', 'last_used_at', 'created_at', 'is_active'
        ]


class DataExportJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataExportJob
        fields = [
            'id', 'status', 'requested_at', 'completed_at', 'expires_at', 'file', 'error_message'
        ]


class AccountDeletionStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = AccountDeletionRequest
        fields = [
            'status', 'requested_at', 'scheduled_for', 'processed_at', 'cancelled_at', 'reason'
        ]
