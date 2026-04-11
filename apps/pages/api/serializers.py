"""
Pages API serializers
"""
from urllib.parse import urlparse

from rest_framework import serializers

from ..models import Page, FAQ, ContactMessage, SiteSettings, Subscriber


def normalize_public_asset_url(value: str | None, request=None) -> str:
    if not value:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        return raw
    if raw.startswith("//"):
        return f"https:{raw}"
    normalized = raw if raw.startswith("/") else f"/{raw.lstrip('/')}"
    if request:
        try:
            return request.build_absolute_uri(normalized)
        except Exception:
            return normalized
    return normalized


class PageListSerializer(serializers.ModelSerializer):
    """Serializer for page list."""

    show_in_menu = serializers.BooleanField(source='show_in_header', required=False)
    
    class Meta:
        model = Page
        fields = [
            'id', 'title', 'slug', 'excerpt', 'template',
            'show_in_menu', 'menu_order', 'created_at'
        ]


class PageDetailSerializer(serializers.ModelSerializer):
    """Serializer for page detail."""

    show_in_menu = serializers.BooleanField(source='show_in_header', required=False)
    
    class Meta:
        model = Page
        fields = [
            'id', 'title', 'slug', 'content', 'excerpt',
            'template', 'featured_image', 'meta_title',
            'meta_description', 'show_in_menu', 'show_in_footer',
            'menu_order', 'created_at', 'updated_at'
        ]


class FAQSerializer(serializers.ModelSerializer):
    """Serializer for FAQ."""
    
    class Meta:
        model = FAQ
        fields = [
            'id', 'question', 'answer', 'category', 'sort_order'
        ]


class FAQGroupedSerializer(serializers.Serializer):
    """Serializer for grouped FAQs."""
    category = serializers.CharField()
    faqs = FAQSerializer(many=True)


class ContactMessageSerializer(serializers.ModelSerializer):
    """Serializer for contact messages."""
    
    class Meta:
        model = ContactMessage
        fields = [
            'id', 'name', 'email', 'phone', 'subject',
            'message', 'is_read', 'is_replied', 'created_at'
        ]
        read_only_fields = ['id', 'is_read', 'is_replied', 'created_at']


class ContactMessageCreateSerializer(serializers.Serializer):
    """Serializer for creating contact messages."""
    name = serializers.CharField(max_length=100)
    email = serializers.EmailField()
    phone = serializers.CharField(max_length=20, required=False, allow_blank=True)
    subject = serializers.CharField(max_length=200)
    message = serializers.CharField()
    
    def create(self, validated_data):
        return ContactMessage.objects.create(**validated_data)


class SiteSettingsSerializer(serializers.ModelSerializer):
    """Serializer for site settings (public)."""
    
    logo = serializers.SerializerMethodField()
    logo_dark = serializers.SerializerMethodField()
    favicon = serializers.SerializerMethodField()
    tagline = serializers.SerializerMethodField()
    address = serializers.SerializerMethodField()
    support_email = serializers.SerializerMethodField()
    currency = serializers.CharField(source='currency_id', read_only=True)
    currency_symbol = serializers.SerializerMethodField()
    
    class Meta:
        model = SiteSettings
        fields = [
            'site_name',
            'site_tagline',
            'site_description',
            'tagline',
            'logo',
            'logo_dark',
            'favicon',
            'contact_email',
            'support_email',
            'contact_phone',
            'contact_address',
            'support_reply_time_note',
            'address',
            'currency',
            'currency_symbol',
            'facebook_url',
            'instagram_url',
            'twitter_url',
            'youtube_url',
            'linkedin_url',
            'tiktok_url',
            'footer_text',
            'copyright_text',
        ]

    def _get_asset_url(self, obj, field_name: str) -> str:
        request = self.context.get('request') if hasattr(self, 'context') else None
        field = getattr(obj, field_name, None)
        if not field:
            return ""
        try:
            url = field.url
        except Exception:
            url = ""
        return normalize_public_asset_url(url, request=request)

    def get_logo(self, obj):
        return self._get_asset_url(obj, 'logo')

    def get_logo_dark(self, obj):
        return self._get_asset_url(obj, 'logo_dark')

    def get_favicon(self, obj):
        return self._get_asset_url(obj, 'favicon')

    def get_tagline(self, obj):
        return getattr(obj, "site_tagline", "")

    def get_address(self, obj):
        return getattr(obj, "contact_address", "")

    def get_support_email(self, obj):
        return getattr(obj, "contact_email", "")

    def get_currency_symbol(self, obj):
        currency = getattr(obj, 'currency', None)
        if not currency:
            return ''
        return getattr(currency, 'native_symbol', None) or getattr(currency, 'symbol', '') or ''


class SubscriberCreateSerializer(serializers.Serializer):
    """Serializer for creating subscribers."""
    email = serializers.EmailField()
    name = serializers.CharField(max_length=100, required=False, allow_blank=True)
    
    def validate_email(self, value):
        existing = Subscriber.objects.filter(email=value, is_active=True).first()
        if existing:
            raise serializers.ValidationError('This email is already subscribed.')
        return value


class UnsubscribeSerializer(serializers.Serializer):
    """Serializer for unsubscribing."""
    email = serializers.EmailField()


class MenuPageSerializer(serializers.ModelSerializer):
    """Serializer for menu pages."""
    url = serializers.SerializerMethodField()
    
    class Meta:
        model = Page
        fields = ['id', 'title', 'slug', 'url', 'menu_order']
    
    def get_url(self, obj):
        return f'/pages/{obj.slug}/'
