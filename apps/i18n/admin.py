"""
Internationalization Admin

Admin configuration for i18n models.
"""
from django.contrib import admin
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from django.db.models import Count

from .models import (
    Language, Currency, ExchangeRate, ExchangeRateHistory,
    Timezone, Country, Division, District, Upazila,
    TranslationNamespace, TranslationKey, Translation, ContentTranslation,
    UserLocalePreference, I18nSettings
)


# =============================================================================
# Language Admin
# =============================================================================

@admin.register(Language)
class LanguageAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'code', 'native_name', 'flag_display', 
        'is_active', 'is_default', 'is_rtl', 'translation_progress_display', 'sort_order'
    )
    list_filter = ('is_active', 'is_default', 'is_rtl')
    search_fields = ('name', 'code', 'native_name')
    list_editable = ('is_active', 'is_default', 'sort_order')
    ordering = ('sort_order', 'name')
    
    fieldsets = (
        (None, {
            'fields': ('name', 'native_name', 'code', 'locale_code')
        }),
        (_('Display'), {
            'fields': ('flag_code', 'font_family', 'is_rtl')
        }),
        (_('Status'), {
            'fields': ('is_active', 'is_default', 'sort_order')
        }),
        (_('Translation Progress'), {
            'fields': ('total_strings', 'translated_strings'),
            'classes': ('collapse',)
        }),
    )
    
    def flag_display(self, obj):
        return format_html('<span style="font-size: 1.5em;">{}</span>', obj.flag_code or '')
    flag_display.short_description = _('Flag')
    
    def translation_progress_display(self, obj):
        if obj.total_strings and obj.total_strings > 0:
            percent = (obj.translated_strings / obj.total_strings) * 100
            color = 'green' if percent >= 80 else 'orange' if percent >= 50 else 'red'
            return format_html(
                '<span style="color: {};">{:.1f}%</span>',
                color, percent
            )
        return '-'
    translation_progress_display.short_description = _('Progress')


# =============================================================================
# Currency Admin
# =============================================================================

@admin.register(Currency)
class CurrencyAdmin(admin.ModelAdmin):
    list_display = (
        'code', 'name', 'symbol_display', 'rate_display',
        'is_active', 'is_default', 'sort_order'
    )
    list_filter = ('is_active', 'is_default', 'number_system')
    search_fields = ('code', 'name', 'symbol')
    list_editable = ('is_active', 'is_default', 'sort_order')
    ordering = ('sort_order', 'code')
    
    fieldsets = (
        (None, {
            'fields': ('code', 'name', 'symbol', 'native_symbol')
        }),
        (_('Display'), {
            'fields': ('decimal_places', 'symbol_position', 
                      'thousand_separator', 'decimal_separator')
        }),
        (_('Number Formatting'), {
            'fields': ('number_system',),
            'classes': ('collapse',)
        }),
        (_('Base Currency'), {
            'fields': ('is_base_currency',)
        }),
        (_('Status'), {
            'fields': ('is_active', 'is_default', 'sort_order')
        }),
    )
    
    def symbol_display(self, obj):
        return f"{obj.symbol} ({obj.native_symbol})" if obj.native_symbol else obj.symbol
    symbol_display.short_description = _('Symbol')
    
    def rate_display(self, obj):
        if obj.is_default:
            return format_html('<span style="color: green;">1.00 (Base)</span>')
        return '1.00'  # All currencies have rate display here  
    rate_display.short_description = _('Rate')


# =============================================================================
# Exchange Rate Admin
# =============================================================================

class ExchangeRateHistoryInline(admin.TabularInline):
    model = ExchangeRateHistory
    extra = 0
    readonly_fields = ('date', 'rate', 'open_rate', 'high_rate', 'low_rate', 'close_rate', 'source')
    can_delete = False
    max_num = 10
    ordering = ('-date',)
    
    def has_add_permission(self, request, obj=None):
        return False


@admin.register(ExchangeRate)
class ExchangeRateAdmin(admin.ModelAdmin):
    list_display = (
        'from_currency', 'to_currency', 'rate_display', 
        'spread_display', 'source', 'is_active', 'valid_from'
    )
    list_filter = ('is_active', 'source', 'from_currency', 'to_currency')
    search_fields = ('from_currency__code', 'to_currency__code')
    raw_id_fields = ('from_currency', 'to_currency')
    date_hierarchy = 'valid_from'
    ordering = ('-valid_from',)
    
    fieldsets = (
        (None, {
            'fields': ('from_currency', 'to_currency')
        }),
        (_('Rates'), {
            'fields': ('rate', 'bid_rate', 'ask_rate')
        }),
        (_('Source & Validity'), {
            'fields': ('source', 'valid_from', 'valid_until', 'is_active')
        }),
    )
    
    def rate_display(self, obj):
        return f"{obj.rate:.6f}"
    rate_display.short_description = _('Rate')
    
    def spread_display(self, obj):
        spread = obj.spread
        if spread:
            return format_html('<span title="Bid-Ask Spread">{:.4f}%</span>', spread * 100)
        return '-'
    spread_display.short_description = _('Spread')


@admin.register(ExchangeRateHistory)
class ExchangeRateHistoryAdmin(admin.ModelAdmin):
    list_display = (
        'from_currency', 'to_currency', 'date', 
        'rate', 'ohlc_display', 'source'
    )
    list_filter = ('source', 'from_currency', 'to_currency')
    search_fields = ('from_currency__code', 'to_currency__code')
    date_hierarchy = 'date'
    ordering = ('-date',)
    readonly_fields = ('from_currency', 'to_currency', 'date')
    
    def ohlc_display(self, obj):
        if obj.open_rate and obj.close_rate:
            change = obj.close_rate - obj.open_rate
            color = 'green' if change >= 0 else 'red'
            return format_html(
                '<span style="color: {};">O:{:.4f} H:{:.4f} L:{:.4f} C:{:.4f}</span>',
                color, obj.open_rate, obj.high_rate or obj.rate, 
                obj.low_rate or obj.rate, obj.close_rate
            )
        return '-'
    ohlc_display.short_description = _('OHLC')


# =============================================================================
# Timezone Admin
# =============================================================================

@admin.register(Timezone)
class TimezoneAdmin(admin.ModelAdmin):
    list_display = (
        'display_name', 'name', 'offset_display', 
        'is_common', 'is_active', 'current_time'
    )
    list_filter = ('is_active', 'is_common', 'has_dst')
    search_fields = ('name', 'display_name')
    list_editable = ('is_active', 'is_common')
    ordering = ('name',)
    
    def offset_display(self, obj):
        return obj.formatted_offset
    offset_display.short_description = _('Offset')
    
    def current_time(self, obj):
        from django.utils import timezone
        import pytz
        try:
            tz = pytz.timezone(obj.name)
            now = timezone.now().astimezone(tz)
            return now.strftime('%H:%M')
        except Exception:
            return '-'
    current_time.short_description = _('Current Time')


# =============================================================================
# Country Admin
# =============================================================================

class DivisionInline(admin.TabularInline):
    model = Division
    extra = 0
    fields = ('name', 'native_name', 'code', 'is_active', 'sort_order')
    ordering = ('sort_order', 'name')


@admin.register(Country)
class CountryAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'code',
        'phone_code_display', 'default_currency', 
        'is_shipping_available', 'is_active'
    )
    list_filter = ('is_active', 'is_shipping_available', 'continent')
    search_fields = ('name', 'native_name', 'code', 'code_alpha3')
    list_editable = ('is_active', 'is_shipping_available')
    ordering = ('name',)
    inlines = [DivisionInline]
    
    fieldsets = (
        (None, {
            'fields': ('name', 'native_name', 'code', 'code_alpha3', 'code_numeric')
        }),
        (_('Geographic'), {
            'fields': ('continent', 'region',)
        }),
        (_('Localization'), {
            'fields': ('phone_code', 'default_currency', 'default_language', 'default_timezone')
        }),
        (_('Address Format'), {
            'fields': ('address_format', 'postal_code_format',),
            'classes': ('collapse',)
        }),
        (_('Tax'), {
            'fields': ('default_tax_rate', 'vat_number_format'),
            'classes': ('collapse',)
        }),
        (_('Status'), {
            'fields': ('is_active', 'is_shipping_available', 'is_billing_allowed')
        }),
    )
    

    def phone_code_display(self, obj):
        return f"+{obj.phone_code}" if obj.phone_code else '-'
    phone_code_display.short_description = _('Phone')


# =============================================================================
# Division/District/Upazila Admin
# =============================================================================

class DistrictInline(admin.TabularInline):
    model = District
    extra = 0
    fields = ('name', 'native_name', 'code', 'is_active', 'sort_order')
    ordering = ('sort_order', 'name')


@admin.register(Division)
class DivisionAdmin(admin.ModelAdmin):
    list_display = ('name', 'native_name', 'code', 'country', 'district_count', 'is_active', 'sort_order')
    list_filter = ('is_active', 'country')
    search_fields = ('name', 'native_name', 'code')
    list_editable = ('is_active', 'sort_order')
    ordering = ('country', 'sort_order', 'name')
    inlines = [DistrictInline]
    
    def district_count(self, obj):
        return obj.districts.count()
    district_count.short_description = _('Districts')


class UpazilaInline(admin.TabularInline):
    model = Upazila
    extra = 0
    fields = ('name', 'native_name', 'code', 'is_active', 'sort_order')
    ordering = ('sort_order', 'name')


@admin.register(District)
class DistrictAdmin(admin.ModelAdmin):
    list_display = ('name', 'native_name', 'code', 'division', 'upazila_count', 'is_active', 'sort_order')
    list_filter = ('is_active', 'division__country', 'division')
    search_fields = ('name', 'native_name', 'code')
    list_editable = ('is_active', 'sort_order')
    ordering = ('division', 'sort_order', 'name')
    inlines = [UpazilaInline]
    
    def upazila_count(self, obj):
        return obj.upazilas.count()
    upazila_count.short_description = _('Upazilas')


@admin.register(Upazila)
class UpazilaAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'native_name', 'code', 'district', 
        'is_active', 'sort_order'
    )
    list_filter = ('is_active', 'district__division__country', 'district__division', 'district')
    search_fields = ('name', 'native_name', 'code', 'post_codes')
    list_editable = ('is_active', 'sort_order')
    ordering = ('district', 'sort_order', 'name')


# =============================================================================
# Translation Admin
# =============================================================================

class TranslationInline(admin.TabularInline):
    model = Translation
    extra = 1
    fields = ('language', 'translated_text', 'status', 'is_machine_translated')
    ordering = ('language__sort_order',)


@admin.register(TranslationNamespace)
class TranslationNamespaceAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'key_count')
    search_fields = ('name', 'description')
    ordering = ('name',)
    
    def key_count(self, obj):
        return obj.keys.count()
    key_count.short_description = _('Keys')


@admin.register(TranslationKey)
class TranslationKeyAdmin(admin.ModelAdmin):
    list_display = ('key', 'namespace', 'context', 'translation_count', 'created_at')
    list_filter = ('namespace',)
    search_fields = ('key', 'context', 'source_text')
    ordering = ('namespace', 'key')
    inlines = [TranslationInline]
    
    def translation_count(self, obj):
        total = obj.translations.count()
        approved = obj.translations.filter(status='approved').count()
        if total == 0:
            return format_html('<span style="color: red;">0</span>')
        color = 'green' if approved == total else 'orange'
        return format_html('<span style="color: {};">{}/{}</span>', color, approved, total)
    translation_count.short_description = _('Translations')


@admin.register(Translation)
class TranslationAdmin(admin.ModelAdmin):
    list_display = (
        'key', 'language', 'status', 'status_badge',
        'is_machine_translated', 'translated_by', 'created_at'
    )
    list_filter = ('status', 'language', 'is_machine_translated', 'key__namespace')
    search_fields = ('key__key', 'translated_text')
    raw_id_fields = ('key',)
    list_editable = ('status',)
    ordering = ('-created_at',)
    
    fieldsets = (
        (None, {
            'fields': ('key', 'language')
        }),
        (_('Translation'), {
            'fields': ('translated_text', 'plural_forms')
        }),
        (_('Status'), {
            'fields': ('status', 'is_machine_translated', 'translated_by', 'reviewed_by')
        }),
    )
    
    def status_badge(self, obj):
        colors = {
            'draft': 'gray',
            'pending': 'orange',
            'approved': 'green',
            'rejected': 'red',
        }
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 6px; border-radius: 3px;">{}</span>',
            colors.get(obj.status, 'gray'),
            obj.get_status_display()
        )
    status_badge.short_description = _('Status')


@admin.register(ContentTranslation)
class ContentTranslationAdmin(admin.ModelAdmin):
    list_display = (
        'content_type', 'content_id', 'field_name', 
        'language', 'is_approved', 'is_machine_translated', 'updated_at'
    )
    list_filter = ('content_type', 'language', 'is_approved', 'is_machine_translated')
    search_fields = ('content_id', 'original_text', 'translated_text')
    list_editable = ('is_approved',)
    ordering = ('-updated_at',)
    date_hierarchy = 'updated_at'
    
    fieldsets = (
        (None, {
            'fields': ('content_type', 'content_id', 'field_name', 'language')
        }),
        (_('Content'), {
            'fields': ('original_text', 'translated_text')
        }),
        (_('Status'), {
            'fields': ('is_approved', 'is_machine_translated', 'translated_by')
        }),
    )


# =============================================================================
# User Preference Admin
# =============================================================================

@admin.register(UserLocalePreference)
class UserLocalePreferenceAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'language', 'currency', 'timezone', 
        'country', 'updated_at'
    )
    list_filter = ('language', 'currency', 'timezone')
    search_fields = ('user__username', 'user__email')
    raw_id_fields = ('user',)
    ordering = ('-updated_at',)
    
    fieldsets = (
        (None, {
            'fields': ('user',)
        }),
        (_('Preferences'), {
            'fields': ('language', 'currency', 'timezone', 'country')
        }),
        (_('Auto-Detection'), {
            'fields': ('auto_detect_language', 'auto_detect_currency', 'auto_detect_timezone'),
            'classes': ('collapse',)
        }),
        (_('Formatting'), {
            'fields': ('date_format', 'time_format', 'first_day_of_week', 
                      'measurement_system', 'temperature_unit'),
            'classes': ('collapse',)
        }),
    )


# =============================================================================
# Settings Admin
# =============================================================================

@admin.register(I18nSettings)
class I18nSettingsAdmin(admin.ModelAdmin):
    list_display = (
        'default_language', 'default_currency', 'default_timezone',
        'auto_update_exchange_rates', 'last_exchange_rate_update'
    )
    
    fieldsets = (
        (_('Language Settings'), {
            'fields': ('default_language', 'fallback_language', 'auto_detect_language', 'show_language_selector')
        }),
        (_('Currency Settings'), {
            'fields': ('default_currency', 'auto_detect_currency', 'show_currency_selector',
                      'show_original_price', 'rounding_method')
        }),
        (_('Timezone Settings'), {
            'fields': ('default_timezone', 'auto_detect_timezone', 'show_timezone_selector')
        }),
        (_('Exchange Rates'), {
            'fields': ('auto_update_exchange_rates', 'exchange_rate_update_frequency',
                      'exchange_rate_provider', 'exchange_rate_api_key',
                      'last_exchange_rate_update')
        }),
        (_('Machine Translation'), {
            'fields': ('enable_machine_translation', 'translation_provider', 'translation_api_key',
                      'auto_translate_new_content', 'require_human_review'),
            'classes': ('collapse',)
        }),
        (_('Geo Detection'), {
            'fields': ('enable_geo_detection', 'geo_ip_provider', 'geo_ip_api_key'),
            'classes': ('collapse',)
        }),
    )
    
    readonly_fields = ('last_exchange_rate_update',)
    
    def has_add_permission(self, request):
        # Only allow one settings instance
        return not I18nSettings.objects.exists()
    
    def has_delete_permission(self, request, obj=None):
        return False


# =============================================================================
# Admin Site Configuration
# =============================================================================

# Group models under a custom admin section
admin.site.site_header = _('Bunoraa Administration')
admin.site.site_title = _('Bunoraa Admin')
