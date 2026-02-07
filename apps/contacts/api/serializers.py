"""
Contacts API Serializers
"""
from rest_framework import serializers

from ..models import (
    ContactCategory, ContactInquiry, ContactResponse,
    ContactAttachment, StoreLocation, ContactSettings,
    CustomizationRequest
)


class ContactCategorySerializer(serializers.ModelSerializer):
    """Serializer for contact categories."""
    
    class Meta:
        model = ContactCategory
        fields = ['id', 'name', 'slug', 'description']


class ContactInquiryCreateSerializer(serializers.Serializer):
    """Serializer for creating contact inquiries."""
    
    name = serializers.CharField(max_length=100)
    email = serializers.EmailField()
    phone = serializers.CharField(max_length=30, required=False, allow_blank=True)
    company = serializers.CharField(max_length=150, required=False, allow_blank=True)
    category_id = serializers.UUIDField(required=False, allow_null=True)
    subject = serializers.CharField(max_length=200)
    message = serializers.CharField()
    order_number = serializers.CharField(max_length=50, required=False, allow_blank=True)


class ContactInquirySerializer(serializers.ModelSerializer):
    """Serializer for contact inquiries."""
    
    category = ContactCategorySerializer(read_only=True)
    
    class Meta:
        model = ContactInquiry
        fields = [
            'id', 'name', 'email', 'phone', 'company',
            'category', 'subject', 'message', 'order_number',
            'status', 'priority', 'created_at', 'responded_at'
        ]
        read_only_fields = ['id', 'status', 'created_at', 'responded_at']


class ContactInquiryDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for contact inquiries."""
    
    category = ContactCategorySerializer(read_only=True)
    responses = serializers.SerializerMethodField()
    attachments = serializers.SerializerMethodField()
    
    class Meta:
        model = ContactInquiry
        fields = [
            'id', 'name', 'email', 'phone', 'company',
            'category', 'subject', 'message', 'order_number',
            'status', 'priority', 'created_at', 'responded_at',
            'responses', 'attachments'
        ]
    
    def get_responses(self, obj):
        responses = obj.responses.all().order_by('created_at')
        return ContactResponseSerializer(responses, many=True).data
    
    def get_attachments(self, obj):
        attachments = obj.attachments.all()
        return ContactAttachmentSerializer(attachments, many=True).data


class ContactResponseSerializer(serializers.ModelSerializer):
    """Serializer for contact responses."""
    
    class Meta:
        model = ContactResponse
        fields = ['id', 'subject', 'message', 'sent_at', 'created_at']


class ContactAttachmentSerializer(serializers.ModelSerializer):
    """Serializer for contact attachments."""
    
    url = serializers.SerializerMethodField()
    
    class Meta:
        model = ContactAttachment
        fields = ['id', 'filename', 'file_size', 'content_type', 'url', 'created_at']
    
    def get_url(self, obj):
        if obj.file:
            return obj.file.url
        return None


class StoreLocationSerializer(serializers.ModelSerializer):
    """Serializer for store locations."""
    
    hours = serializers.SerializerMethodField()
    
    class Meta:
        model = StoreLocation
        fields = [
            'id', 'name', 'slug', 'address_line1', 'address_line2',
            'city', 'state', 'postal_code', 'country', 'full_address',
            'latitude', 'longitude', 'phone', 'email',
            'is_pickup_location', 'is_returns_location', 'is_main',
            'description', 'image', 'hours'
        ]
    
    def get_hours(self, obj):
        return obj.get_hours()


class StoreLocationPickupSerializer(serializers.ModelSerializer):
    """Serializer for pickup locations with checkout details."""

    hours = serializers.SerializerMethodField()

    class Meta:
        model = StoreLocation
        fields = [
            'id', 'name', 'slug',
            'address_line1', 'address_line2', 'city', 'state', 'postal_code', 'country',
            'full_address', 'latitude', 'longitude',
            'phone', 'email',
            'pickup_fee', 'min_pickup_time_hours', 'max_hold_days',
            'hours',
        ]

    def get_hours(self, obj):
        return obj.get_hours()


class StoreLocationMinimalSerializer(serializers.ModelSerializer):
    """Minimal serializer for store locations."""
    
    class Meta:
        model = StoreLocation
        fields = ['id', 'name', 'city', 'country', 'full_address']


class ContactSettingsPublicSerializer(serializers.ModelSerializer):
    """Public serializer for contact settings."""
    
    social_links = serializers.SerializerMethodField()
    
    class Meta:
        model = ContactSettings
        fields = [
            'general_email', 'support_email', 'sales_email', 'phone',
            'social_links', 'allow_attachments', 'max_attachment_size_mb',
            'business_hours_note'
        ]
    
    def get_social_links(self, obj):
        links = {}
        if obj.facebook_url:
            links['facebook'] = obj.facebook_url
        if obj.twitter_url:
            links['twitter'] = obj.twitter_url
        if obj.instagram_url:
            links['instagram'] = obj.instagram_url
        if obj.linkedin_url:
            links['linkedin'] = obj.linkedin_url
        if obj.youtube_url:
            links['youtube'] = obj.youtube_url
        if obj.pinterest_url:
            links['pinterest'] = obj.pinterest_url
        if obj.tiktok_url:
            links['tiktok'] = obj.tiktok_url
        return links


class NearbyLocationRequestSerializer(serializers.Serializer):
    """Serializer for nearby location request."""
    
    latitude = serializers.DecimalField(max_digits=9, decimal_places=6)
    longitude = serializers.DecimalField(max_digits=9, decimal_places=6)
    radius_km = serializers.IntegerField(required=False, default=50, min_value=1, max_value=500)


class CustomizationRequestSerializer(serializers.ModelSerializer):
    """Serializer for creating a customization request."""
    
    class Meta:
        model = CustomizationRequest
        fields = ['product', 'name', 'email', 'phone', 'message']
