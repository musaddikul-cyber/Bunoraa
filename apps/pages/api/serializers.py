"""
Pages API serializers
"""
from rest_framework import serializers

from ..models import Page, FAQ, ContactMessage, SiteSettings, Subscriber


class PageListSerializer(serializers.ModelSerializer):
    """Serializer for page list."""
    
    class Meta:
        model = Page
        fields = [
            'id', 'title', 'slug', 'excerpt', 'template',
            'show_in_menu', 'menu_order', 'created_at'
        ]


class PageDetailSerializer(serializers.ModelSerializer):
    """Serializer for page detail."""
    
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
    
    class Meta:
        model = SiteSettings
        fields = [
            'site_name', 'tagline', 'logo', 'favicon',
            'contact_email', 'support_email', 'contact_phone',
            'address', 'currency', 'facebook_url', 'instagram_url',
            'twitter_url', 'youtube_url', 'linkedin_url',
            'footer_text', 'copyright_text'
        ]


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
