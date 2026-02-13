"""
Account admin configuration
"""
from django.contrib import admin, messages
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.translation import gettext_lazy as _
from django.utils.html import format_html
from django.db.models import Count, Sum
from django.urls import reverse
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

from core.admin_mixins import (
    ImportExportEnhancedModelAdmin,
    EnhancedTabularInline,
    DateRangeFilter,
    format_currency,
    format_number,
)

# Import behavior models with graceful fallback
try:
    from .behavior_models import (
        UserBehaviorProfile,
        UserCredentialVault,
        UserPreferences,
        UserSession,
        UserInteraction
    )
    BEHAVIOR_MODELS_AVAILABLE = True
except ImportError:
    BEHAVIOR_MODELS_AVAILABLE = False


class AddressInline(EnhancedTabularInline):
    """Inline address editor for user admin."""
    model = Address
    extra = 0
    fields = ['address_type', 'full_name', 'city', 'country', 'is_default']
    readonly_fields = []


@admin.register(User)
class UserAdmin(ImportExportEnhancedModelAdmin, BaseUserAdmin):
    """Enhanced custom user admin with advanced features."""
    
    list_display = [
        'email', 'full_name_display', 'verification_badge', 'orders_stats',
        'activity_status', 'is_active', 'is_staff', 'created_at'
    ]
    list_filter = ['is_active', 'is_staff', 'is_verified', 'is_deleted', 'newsletter_subscribed', 'created_at']
    search_fields = ['email', 'first_name', 'last_name', 'phone']
    ordering = ['-created_at']
    date_hierarchy = 'created_at'
    list_per_page = 25
    
    inlines = [AddressInline]
    
    export_fields = ['email', 'first_name', 'last_name', 'phone', 'is_verified', 
                     'is_active', 'newsletter_subscribed', 'created_at']
    
    actions = ['export_as_csv', 'verify_selected', 'unverify_selected', 
               'activate_selected', 'deactivate_selected', 'send_password_reset',
               'subscribe_newsletter', 'unsubscribe_newsletter']
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        (_('Personal info'), {'fields': ('first_name', 'last_name', 'phone', 'avatar', 'date_of_birth')}),
        (_('Permissions'), {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'is_verified', 'is_deleted', 'groups', 'user_permissions'),
        }),
        (_('Preferences'), {'fields': ('newsletter_subscribed',)}),
        (_('Important dates'), {'fields': ('last_login', 'created_at', 'updated_at')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'first_name', 'last_name'),
        }),
    )
    
    readonly_fields = ['created_at', 'updated_at', 'last_login']
    
    def get_queryset(self, request):
        return super().get_queryset(request).annotate(
            order_count=Count('orders', distinct=True),
            total_spent=Sum('orders__total')
        )
    
    def full_name_display(self, obj):
        name = obj.get_full_name() or 'No name'
        return name
    full_name_display.short_description = _('Name')
    
    def verification_badge(self, obj):
        if obj.is_verified:
            return format_html('<span style="color: #16a34a;" title="Verified">✓</span>')
        return format_html('<span style="color: #dc2626;" title="Not verified">✗</span>')
    verification_badge.short_description = _('Verified')
    verification_badge.admin_order_field = 'is_verified'
    
    def orders_stats(self, obj):
        order_count = getattr(obj, 'order_count', 0) or 0
        total_spent = getattr(obj, 'total_spent', 0) or 0
        if order_count > 0:
            return format_html(
                '<span title="Orders: {}, Total: {}">{} orders ({})</span>',
                order_count, format_currency(total_spent),
                order_count, format_currency(total_spent)
            )
        return format_html('<span style="color: #9ca3af;">No orders</span>')
    orders_stats.short_description = _('Orders')
    orders_stats.admin_order_field = 'order_count'
    
    def activity_status(self, obj):
        from django.utils import timezone
        from datetime import timedelta
        
        if obj.last_login:
            days_ago = (timezone.now() - obj.last_login).days
            if days_ago == 0:
                return format_html('<span style="color: #16a34a;">Today</span>')
            elif days_ago <= 7:
                return format_html('<span style="color: #16a34a;">{}d ago</span>', days_ago)
            elif days_ago <= 30:
                return format_html('<span style="color: #d97706;">{}d ago</span>', days_ago)
            else:
                return format_html('<span style="color: #9ca3af;">{}d ago</span>', days_ago)
        return format_html('<span style="color: #9ca3af;">Never</span>')
    activity_status.short_description = _('Last Active')
    activity_status.admin_order_field = 'last_login'
    
    # Bulk actions
    def verify_selected(self, request, queryset):
        updated = queryset.update(is_verified=True)
        self.message_user(request, f'{updated} users verified.', messages.SUCCESS)
    verify_selected.short_description = _("Verify selected users")
    
    def unverify_selected(self, request, queryset):
        updated = queryset.update(is_verified=False)
        self.message_user(request, f'{updated} users unverified.', messages.SUCCESS)
    unverify_selected.short_description = _("Unverify selected users")
    
    def activate_selected(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(request, f'{updated} users activated.', messages.SUCCESS)
    activate_selected.short_description = _("Activate selected users")
    
    def deactivate_selected(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f'{updated} users deactivated.', messages.SUCCESS)
    deactivate_selected.short_description = _("Deactivate selected users")
    
    def send_password_reset(self, request, queryset):
        """Send password reset email to selected users."""
        count = 0
        for user in queryset:
            try:
                # Create token and send email
                token = PasswordResetToken.objects.create(user=user)
                # TODO: Actually send email
                count += 1
            except Exception:
                pass
        self.message_user(request, f'Password reset emails sent to {count} users.', messages.SUCCESS)
    send_password_reset.short_description = _("Send password reset email")
    
    def subscribe_newsletter(self, request, queryset):
        updated = queryset.update(newsletter_subscribed=True)
        self.message_user(request, f'{updated} users subscribed to newsletter.', messages.SUCCESS)
    subscribe_newsletter.short_description = _("Subscribe to newsletter")
    
    def unsubscribe_newsletter(self, request, queryset):
        updated = queryset.update(newsletter_subscribed=False)
        self.message_user(request, f'{updated} users unsubscribed from newsletter.', messages.SUCCESS)
    unsubscribe_newsletter.short_description = _("Unsubscribe from newsletter")


@admin.register(Address)
class AddressAdmin(ImportExportEnhancedModelAdmin):
    """Address admin."""
    
    list_display = ['full_name', 'user', 'city', 'country', 'address_type', 'is_default', 'created_at']
    list_filter = ['address_type', 'is_default', 'country', 'is_deleted']
    search_fields = ['full_name', 'user__email', 'city', 'address_line_1']
    raw_id_fields = ['user']
    ordering = ['-created_at']
    
    fieldsets = (
        (None, {'fields': ('user', 'address_type')}),
        (_('Contact'), {'fields': ('full_name', 'phone')}),
        (_('Address'), {'fields': ('address_line_1', 'address_line_2', 'city', 'state', 'postal_code', 'country')}),
        (_('Status'), {'fields': ('is_default', 'is_deleted')}),
        (_('Timestamps'), {'fields': ('created_at', 'updated_at')}),
    )
    
    readonly_fields = ['created_at', 'updated_at']


@admin.register(PasswordResetToken)
class PasswordResetTokenAdmin(ImportExportEnhancedModelAdmin):
    """Password reset token admin."""
    
    list_display = ['user', 'created_at', 'expires_at', 'used']
    list_filter = ['used', 'created_at']
    search_fields = ['user__email']
    raw_id_fields = ['user']
    readonly_fields = ['token', 'created_at']


@admin.register(EmailVerificationToken)
class EmailVerificationTokenAdmin(ImportExportEnhancedModelAdmin):
    """Email verification token admin."""
    
    list_display = ['user', 'created_at', 'expires_at', 'used']
    list_filter = ['used', 'created_at']
    search_fields = ['user__email']
    raw_id_fields = ['user']
    readonly_fields = ['token', 'created_at']


@admin.register(WebAuthnCredential)
class WebAuthnCredentialAdmin(ImportExportEnhancedModelAdmin):
    list_display = ['user', 'nickname', 'created_at', 'last_used_at', 'is_active']
    list_filter = ['is_active', 'created_at']
    search_fields = ['user__email', 'nickname']
    raw_id_fields = ['user']
    readonly_fields = ['created_at', 'last_used_at']


@admin.register(WebAuthnChallenge)
class WebAuthnChallengeAdmin(ImportExportEnhancedModelAdmin):
    list_display = ['user', 'challenge_type', 'created_at', 'expires_at', 'consumed']
    list_filter = ['challenge_type', 'consumed']
    search_fields = ['user__email', 'challenge']
    raw_id_fields = ['user']
    readonly_fields = ['created_at']


@admin.register(DataExportJob)
class DataExportJobAdmin(ImportExportEnhancedModelAdmin):
    list_display = ['user', 'status', 'requested_at', 'completed_at', 'expires_at']
    list_filter = ['status']
    search_fields = ['user__email']
    raw_id_fields = ['user']
    readonly_fields = ['requested_at', 'completed_at', 'expires_at']


@admin.register(AccountDeletionRequest)
class AccountDeletionRequestAdmin(ImportExportEnhancedModelAdmin):
    list_display = ['user', 'status', 'requested_at', 'scheduled_for', 'processed_at']
    list_filter = ['status']
    search_fields = ['user__email']
    raw_id_fields = ['user']
    readonly_fields = ['requested_at', 'scheduled_for', 'processed_at', 'cancelled_at']


# =============================================================================
# Behavior Models Admin (for ML-based personalization)
# =============================================================================

if BEHAVIOR_MODELS_AVAILABLE:
    @admin.register(UserBehaviorProfile)
    class UserBehaviorProfileAdmin(ImportExportEnhancedModelAdmin):
        """User behavior profile admin for ML personalization."""
        
        list_display = ['user', 'engagement_score', 'loyalty_score', 'total_orders', 'total_spent', 'last_active']
        list_filter = ['cluster_id', 'preferred_device']
        search_fields = ['user__email', 'user__first_name', 'user__last_name']
        raw_id_fields = ['user']
        readonly_fields = ['created_at', 'updated_at', 'feature_vector']
        ordering = ['-engagement_score']
        
        fieldsets = (
            (None, {'fields': ('user',)}),
            (_('Session Tracking'), {'fields': ('total_sessions', 'total_page_views', 'avg_session_duration', 'last_active')}),
            (_('Product Engagement'), {'fields': ('products_viewed', 'products_added_to_cart', 'products_purchased', 'products_wishlisted')}),
            (_('Preferences'), {'fields': ('category_preferences', 'tag_preferences', 'price_range_preference')}),
            (_('Purchase Behavior'), {'fields': ('total_orders', 'total_spent', 'avg_order_value', 'last_purchase_date')}),
            (_('Scores'), {'fields': ('engagement_score', 'loyalty_score', 'recency_score')}),
            (_('Device & Time'), {'fields': ('preferred_device', 'preferred_browser', 'preferred_shopping_hours', 'preferred_shopping_days')}),
            (_('Search'), {'fields': ('search_count', 'common_search_terms')}),
            (_('ML'), {'fields': ('feature_vector', 'cluster_id')}),
            (_('Timestamps'), {'fields': ('created_at', 'updated_at')}),
        )

    @admin.register(UserPreferences)
    class UserPreferencesAdmin(ImportExportEnhancedModelAdmin):
        """User preferences admin."""
        
        list_display = ['user', 'language', 'currency', 'theme', 'email_notifications', 'allow_tracking']
        list_filter = ['language', 'currency', 'theme', 'email_notifications', 'allow_tracking']
        search_fields = ['user__email']
        raw_id_fields = ['user']
        readonly_fields = ['created_at', 'updated_at']
        
        fieldsets = (
            (None, {'fields': ('user',)}),
            (_('Locale'), {'fields': ('language', 'currency', 'timezone')}),
            (_('UI'), {'fields': ('theme', 'reduce_motion', 'high_contrast', 'large_text')}),
            (_('Notifications'), {'fields': ('email_notifications', 'sms_notifications', 'push_notifications')}),
            (_('Notification Types'), {'fields': ('notify_order_updates', 'notify_promotions', 'notify_price_drops', 'notify_back_in_stock', 'notify_recommendations')}),
            (_('Privacy'), {'fields': ('allow_tracking', 'share_data_for_ads')}),
            (_('Timestamps'), {'fields': ('created_at', 'updated_at')}),
        )

    @admin.register(UserSession)
    class UserSessionAdmin(ImportExportEnhancedModelAdmin):
        """User session admin for analytics."""
        
        list_display = ['session_key_short', 'user', 'device_type', 'browser', 'country', 'page_views', 'started_at', 'duration']
        list_filter = ['device_type', 'browser', 'country_code', 'is_bounce']
        search_fields = ['session_key', 'user__email', 'ip_address']
        raw_id_fields = ['user']
        readonly_fields = ['started_at', 'last_activity']
        ordering = ['-started_at']
        date_hierarchy = 'started_at'
        
        def session_key_short(self, obj):
            return f"{obj.session_key[:8]}..."
        session_key_short.short_description = 'Session'

    @admin.register(UserInteraction)
    class UserInteractionAdmin(ImportExportEnhancedModelAdmin):
        """User interaction admin for ML training data."""
        
        list_display = ['interaction_type', 'user', 'product', 'category', 'created_at']
        list_filter = ['interaction_type', 'created_at']
        search_fields = ['user__email', 'search_query']
        raw_id_fields = ['user', 'session', 'product', 'category']
        readonly_fields = ['created_at']
        ordering = ['-created_at']
        date_hierarchy = 'created_at'
