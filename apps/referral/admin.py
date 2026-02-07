from django.contrib import admin
from .models import ReferralCode, ReferralReward

@admin.register(ReferralCode)
class ReferralCodeAdmin(admin.ModelAdmin):
    list_display = ('code', 'user', 'is_active', 'expires_at', 'created_at')
    list_filter = ('is_active', 'expires_at')
    search_fields = ('code', 'user__email')
    readonly_fields = ('created_at',)
    raw_id_fields = ('user',)

    actions = ['activate_codes', 'deactivate_codes']

    def activate_codes(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(request, f'{updated} referral codes activated.')
    activate_codes.short_description = 'Activate selected referral codes'

    def deactivate_codes(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f'{updated} referral codes deactivated.')
    deactivate_codes.short_description = 'Deactivate selected referral codes'


@admin.register(ReferralReward)
class ReferralRewardAdmin(admin.ModelAdmin):
    list_display = ('reward_type', 'value', 'referrer_user', 'referee_user', 'status', 'earned_at')
    list_filter = ('reward_type', 'status')
    search_fields = ('referrer_user__email', 'referee_user__email', 'description')
    readonly_fields = ('created_at', 'updated_at', 'earned_at', 'applied_at')
    raw_id_fields = ('referral_code', 'referrer_user', 'referee_user')

    actions = ['mark_earned', 'mark_applied']

    def mark_earned(self, request, queryset):
        from django.utils import timezone
        updated = queryset.update(status='earned', earned_at=timezone.now())
        self.message_user(request, f'{updated} referral rewards marked as earned.')
    mark_earned.short_description = 'Mark selected rewards as earned'

    def mark_applied(self, request, queryset):
        from django.utils import timezone
        updated = queryset.update(status='applied', applied_at=timezone.now())
        self.message_user(request, f'{updated} referral rewards marked as applied.')
    mark_applied.short_description = 'Mark selected rewards as applied'