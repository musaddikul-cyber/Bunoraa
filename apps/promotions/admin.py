"""
Promotions admin configuration
"""
from decimal import Decimal
from urllib.parse import urlparse

from django.contrib import admin
from django import forms
from django.conf import settings
from .models import Coupon, CouponUsage, Banner, Sale
from core.admin_mixins import ImportExportEnhancedModelAdmin


@admin.register(Coupon)
class CouponAdmin(ImportExportEnhancedModelAdmin):
    list_display = [
        'code', 'discount_type', 'discount_value', 'currency', 'is_valid',
        'times_used', 'usage_limit', 'valid_from', 'valid_until', 'is_active'
    ]
    list_filter = ['discount_type', 'currency', 'is_active', 'first_order_only', 'created_at']
    search_fields = ['code', 'description']
    filter_horizontal = ['categories', 'products', 'users']
    readonly_fields = ['times_used', 'created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('code', 'description')
        }),
        ('Discount', {
            'fields': ('discount_type', 'discount_value', 'maximum_discount', 'currency')
        }),
        ('Requirements', {
            'fields': ('minimum_order_amount', 'first_order_only')
        }),
        ('Usage Limits', {
            'fields': ('usage_limit', 'usage_limit_per_user', 'times_used')
        }),
        ('Validity', {
            'fields': ('valid_from', 'valid_until', 'is_active')
        }),
        ('Restrictions', {
            'fields': ('categories', 'products', 'users'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(CouponUsage)
class CouponUsageAdmin(ImportExportEnhancedModelAdmin):
    list_display = ['coupon', 'user', 'order', 'discount_applied', 'currency', 'created_at']
    list_filter = ['currency', 'created_at']
    search_fields = ['coupon__code', 'user__email', 'order__order_number']
    readonly_fields = ['coupon', 'user', 'order', 'discount_applied', 'currency', 'created_at']


class BannerAdminForm(forms.ModelForm):
    link_url = forms.CharField(
        required=False,
        help_text="Use a full URL or a path like /products/. Relative paths auto-use this site's origin.",
    )
    overlay_opacity_percent = forms.IntegerField(
        required=False,
        min_value=0,
        max_value=100,
        label="Overlay opacity (%)",
        help_text="Set overlay transparency as a percentage (0 to 100).",
    )

    class Meta:
        model = Banner
        fields = "__all__"

    def __init__(self, *args, request=None, **kwargs):
        self.request = request
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.overlay_opacity is not None:
            self.fields["overlay_opacity_percent"].initial = int(
                round(float(self.instance.overlay_opacity) * 100)
            )

    def _normalize_origin(self, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")

    def _resolve_site_origin(self) -> str:
        if self.request is not None:
            try:
                host = self.request.get_host()
            except Exception:
                host = ""
            if host:
                scheme = "https" if self.request.is_secure() else "http"
                return f"{scheme}://{host}"

        configured = (
            getattr(settings, "NEXT_FRONTEND_ORIGIN", "")
            or getattr(settings, "NEXT_PUBLIC_SITE_URL", "")
            or getattr(settings, "SITE_URL", "")
        )
        normalized = self._normalize_origin(configured)
        if normalized:
            return normalized
        return "http://localhost:3000"

    def clean_link_url(self):
        raw = (self.cleaned_data.get("link_url") or "").strip()
        if not raw:
            return ""

        parsed = urlparse(raw)
        if parsed.scheme and parsed.netloc:
            return raw

        base = self._resolve_site_origin()
        path = raw if raw.startswith("/") else f"/{raw}"
        return f"{base}{path}"

    def clean(self):
        cleaned_data = super().clean()
        percentage = cleaned_data.get("overlay_opacity_percent")
        cleaned_data["overlay_opacity"] = (
            None if percentage in (None, "") else (Decimal(percentage) / Decimal("100"))
        )
        return cleaned_data


@admin.register(Banner)
class BannerAdmin(ImportExportEnhancedModelAdmin):
    form = BannerAdminForm
    list_display = [
        'title', 'position', 'is_visible', 'sort_order',
        'start_date', 'end_date', 'is_active'
    ]
    list_filter = ['position', 'is_active', 'created_at']
    search_fields = ['title', 'subtitle']
    list_editable = ['sort_order', 'is_active']
    ordering = ['position', 'sort_order']
    
    fieldsets = (
        ('Content', {
            'fields': ('title', 'subtitle')
        }),
        ('Images', {
            'fields': ('image', 'image_mobile')
        }),
        ('Link', {
            'fields': ('link_url', 'link_text')
        }),
        ('Style', {
            'fields': (
                'style_height', 'style_width', 'style_max_width',
                'style_border_radius', 'style_border_width',
                'style_border_color', 'style_background_color',
                'overlay_color', 'overlay_opacity_percent', 'text_color',
                'content_vertical_position', 'content_horizontal_alignment',
                'title_font_size', 'subtitle_font_size',
                'button_alignment', 'button_font_size',
                'button_padding', 'button_min_height'
            ),
            'classes': ('collapse',),
        }),
        ('Display', {
            'fields': ('position', 'sort_order')
        }),
        ('Validity', {
            'fields': ('start_date', 'end_date', 'is_active')
        }),
    )

    def get_form(self, request, obj=None, **kwargs):
        base_form = super().get_form(request, obj, **kwargs)

        class RequestAwareBannerAdminForm(base_form):
            def __init__(self, *args, **inner_kwargs):
                inner_kwargs["request"] = request
                super().__init__(*args, **inner_kwargs)

        return RequestAwareBannerAdminForm


@admin.register(Sale)
class SaleAdmin(ImportExportEnhancedModelAdmin):
    list_display = [
        'name', 'slug', 'discount_type', 'discount_value', 'currency',
        'is_running', 'start_date', 'end_date', 'is_active'
    ]
    list_filter = ['discount_type', 'currency', 'is_active', 'start_date']
    search_fields = ['name', 'description']
    prepopulated_fields = {'slug': ('name',)}
    filter_horizontal = ['products', 'categories']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'slug', 'description')
        }),
        ('Discount', {
            'fields': ('discount_type', 'discount_value', 'currency')
        }),
        ('Products', {
            'fields': ('products', 'categories')
        }),
        ('Banner', {
            'fields': ('banner_image',)
        }),
        ('Schedule', {
            'fields': ('start_date', 'end_date', 'is_active')
        }),
    )
