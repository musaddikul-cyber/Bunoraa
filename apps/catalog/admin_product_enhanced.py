"""
Enhanced ProductAdmin with variant and attribute-friendly interface.

This module extends ProductAdmin with:
- Enhanced variant inline with drag-drop, bulk pricing, and variant generation
- Product attribute value management inline
- Reorderable image inline
- Custom JavaScript tools for variant management
"""
import logging
from decimal import Decimal

from django import forms
from django.contrib import admin, messages
from django.contrib.admin import helpers
from django.db import models, transaction
from django.http import JsonResponse
from django.shortcuts import render
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.decorators import method_decorator
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.views.decorators.http import require_http_methods

from core.admin import ImportExportEnhancedModelAdmin
from core.admin.mixins import BulkActivateMixin, BulkFeaturedMixin

from .admin_inlines import (
    ProductImageEnhancedInline,
    ProductVariantEnhancedInline,
    ProductAttributeValueInline,
    Product3DAssetEnhancedInline,
    ProductMakingOfInline,
)
from .forms import ProductAdminForm
from .models import (
    Attribute,
    AttributeValue,
    Product,
    ProductVariant,
    Option,
    OptionValue,
    ProductAttributeValue,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Actions & Forms for Variant Generation
# =============================================================================

class VariantGeneratorForm(forms.Form):
    """Form for generating variants from option combinations."""
    
    options = forms.ModelMultipleChoiceField(
        queryset=Option.objects.prefetch_related('values'),
        required=True,
        widget=forms.CheckboxSelectMultiple,
        help_text="Select the options to combine (e.g., Size, Color)"
    )
    
    generate_skus = forms.BooleanField(
        required=False,
        initial=True,
        help_text="Auto-generate SKUs from option combinations"
    )
    
    base_price = forms.DecimalField(
        max_digits=12,
        decimal_places=2,
        required=False,
        help_text="Optional base price for all new variants"
    )
    
    base_stock = forms.IntegerField(
        min_value=0,
        initial=0,
        required=False,
        help_text="Optional initial stock quantity"
    )


class BulkVariantPricingForm(forms.Form):
    """Form for bulk updating variant pricing."""
    
    ACTION_CHOICES = [
        ('set_fixed', 'Set to fixed price'),
        ('increase_percent', 'Increase by %'),
        ('decrease_percent', 'Decrease by %'),
        ('increase_fixed', 'Increase by fixed amount'),
        ('decrease_fixed', 'Decrease by fixed amount'),
    ]
    
    action = forms.ChoiceField(
        choices=ACTION_CHOICES,
        required=True,
        widget=forms.RadioSelect
    )
    
    value = forms.DecimalField(
        max_digits=12,
        decimal_places=2,
        required=True,
        label="Value"
    )
    
    apply_to_empty_only = forms.BooleanField(
        required=False,
        initial=False,
        label="Only apply to variants without a price",
        help_text="If checked, only variants using the parent product price will be updated"
    )


# =============================================================================
# Enhanced Product Admin
# =============================================================================

class ProductAdminEnhanced(ImportExportEnhancedModelAdmin, BulkActivateMixin, BulkFeaturedMixin):
    """
    Enhanced ProductAdmin with:
    - Drag-drop reorderable inlines
    - Variant generation tools
    - Bulk pricing actions
    - Attribute management inline
    - Visual stock indicators
    """
    
    form = ProductAdminForm
    # We'll keep the same list_display and filtering as the original
    
    change_form_template = "admin/catalog/product/change_form.html"
    
    # Enhanced inlines - replace originals
    inlines = [
        ProductImageEnhancedInline,
        ProductVariantEnhancedInline,
        ProductAttributeValueInline,
        Product3DAssetEnhancedInline,
        ProductMakingOfInline,
    ]
    
    # Fieldsets - reorganized for better flow
    fieldsets = (
        ("Basic Information", {
            "fields": ("name", "slug", "sku", "description", "short_description"),
            "classes": ("wide",)
        }),
        ("Categories & Classification", {
            "fields": ("primary_category", "categories", "tags", "product_type"),
            "classes": ("wide",)
        }),
        ("Pricing & Inventory", {
            "fields": (
                "price", "sale_price", "cost", "currency",
                "stock_quantity", "low_stock_threshold", "allow_backorder"
            ),
            "classes": ("wide",),
            "description": "Set base pricing and inventory. Variants can have their own prices."
        }),
        ("SEO", {
            "fields": ("meta_title", "meta_description", "meta_keywords"),
            "classes": ("collapse",),
        }),
        ("Status & Flags", {
            "fields": (
                "is_active", "is_featured", "is_bestseller",
                "is_new_arrival", "can_be_customized"
            ),
        }),
        ("Display & Media", {
            "fields": ("aspect_ratio", "is_mobile_optimized"),
            "classes": ("collapse",),
        }),
        ("Shipping", {
            "fields": ("weight", "length", "width", "height", "shipping_material"),
            "classes": ("collapse",),
        }),
        ("Sustainability", {
            "fields": (
                "carbon_footprint_kg", "recycled_content_percentage",
                "sustainability_score", "ethical_sourcing_notes", "eco_certifications"
            ),
            "classes": ("collapse",),
        }),
        ("Publishing", {
            "fields": ("publish_from", "publish_until", "voice_keywords"),
            "classes": ("collapse",),
        }),
    )
    
    class Media:
        css = {
            "all": (
                "css/admin/variant_manager.css",
                "css/admin/attribute_manager.css",
                "css/admin/image_manager.css",
            ),
        }
        js = (
            "admin/js/vendor/jquery/jquery.min.js",
            "admin/js/jquery.init.js",
            "https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js",
            "js/admin/sortable_inlines.js",
            "js/admin/variant_manager.js",
            "js/admin/attribute_manager.js",
            "js/admin/image_manager.js",
        )
    
    # =========================================================================
    # Custom URLs for Tools
    # =========================================================================
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                '<uuid:pk>/variant-generator/',
                self.admin_site.admin_view(self.variant_generator_view),
                name='catalog_product_variant_generator'
            ),
            path(
                '<uuid:pk>/bulk-pricing/',
                self.admin_site.admin_view(self.bulk_pricing_view),
                name='catalog_product_bulk_pricing'
            ),
            path(
                'api/options/',
                self.admin_site.admin_view(self.available_options_api),
                name='catalog_product_options_api'
            ),
        ]
        return custom_urls + urls
    
    # =========================================================================
    # Variant Generation View
    # =========================================================================
    
    def variant_generator_view(self, request, pk):
        """View for generating product variants from options."""
        product = self.get_object(request, pk)
        
        if not product:
            messages.error(request, "Product not found.")
            return self.response_post_save_change(request, product)
        
        if request.method == 'POST':
            form = VariantGeneratorForm(request.POST)
            if form.is_valid():
                selected_options = form.cleaned_data['options']
                generate_skus = form.cleaned_data['generate_skus']
                base_price = form.cleaned_data['base_price']
                base_stock = form.cleaned_data['base_stock']
                
                # Generate combinations
                combinations = self.generate_option_combinations(selected_options)
                created_count = 0
                
                for combo in combinations:
                    variant = self.create_variant_from_combination(
                        product, combo, generate_skus, base_price, base_stock
                    )
                    if variant:
                        created_count += 1
                
                messages.success(
                    request,
                    f"Successfully created {created_count} variant(s)."
                )
                return self.response_post_save_change(request, product)
        else:
            form = VariantGeneratorForm()
        
        context = {
            **self.admin_site.each_context(request),
            'title': f'Generate Variants for: {product.name}',
            'product': product,
            'form': form,
            'opts': self.opts,
        }
        
        return TemplateResponse(
            request,
            'admin/catalog/product/variant_generator.html',
            context
        )
    
    def generate_option_combinations(self, options):
        """Generate all combinations of option values."""
        if not options:
            return []
        
        # Get values for each option
        option_values = []
        for option in options:
            values = list(option.values.all())
            if values:
                option_values.append([
                    {'option': option, 'value': v}
                    for v in values
                ])
        
        if not option_values:
            return []
        
        # Cartesian product
        import itertools
        combinations = list(itertools.product(*option_values))
        
        return combinations
    
    def create_variant_from_combination(self, product, combination, generate_sku, base_price, base_stock):
        """Create a single variant from an option combination."""
        try:
            with transaction.atomic():
                # Generate SKU
                if generate_sku:
                    parts = [ov['value'].value.replace(' ', '-') for ov in combination]
                    sku = f"{product.sku or 'PROD'}-{'-'.join(parts)}".upper()[:80]
                else:
                    sku = None
                
                # Check if variant with same options exists
                existing = ProductVariant.objects.filter(
                    product=product,
                    option_values__in=[ov['value'].id for ov in combination]
                ).distinct()
                
                # Simple check - could be more sophisticated
                for variant in existing:
                    variant_options = set(variant.option_values.values_list('id', flat=True))
                    combo_options = {ov['value'].id for ov in combination}
                    if variant_options == combo_options:
                        return None  # Already exists
                
                # Create variant
                variant = ProductVariant.objects.create(
                    product=product,
                    sku=sku,
                    price=base_price,
                    stock_quantity=base_stock or 0,
                    is_default=False
                )
                
                # Add option values
                for ov in combination:
                    variant.option_values.add(ov['value'])
                
                return variant
                
        except Exception as e:
            logger.error(f"Error creating variant: {e}")
            return None
    
    # =========================================================================
    # Bulk Pricing View
    # =========================================================================
    
    def bulk_pricing_view(self, request, pk):
        """View for bulk updating variant pricing."""
        product = self.get_object(request, pk)
        
        if not product:
            messages.error(request, "Product not found.")
            return self.response_post_save_change(request, product)
        
        variants = product.variants.all()
        
        if request.method == 'POST':
            form = BulkVariantPricingForm(request.POST)
            if form.is_valid():
                action = form.cleaned_data['action']
                value = form.cleaned_data['value']
                apply_to_empty = form.cleaned_data['apply_to_empty_only']
                
                updated = 0
                
                for variant in variants:
                    if apply_to_empty and variant.price is not None:
                        continue
                    
                    current_price = variant.price if variant.price is not None else product.price
                    
                    if action == 'set_fixed':
                        new_price = value
                    elif action == 'increase_percent':
                        new_price = current_price * (1 + value / 100)
                    elif action == 'decrease_percent':
                        new_price = current_price * (1 - value / 100)
                    elif action == 'increase_fixed':
                        new_price = current_price + value
                    elif action == 'decrease_fixed':
                        new_price = max(0, current_price - value)
                    else:
                        continue
                    
                    variant.price = round(new_price, 2)
                    variant.save(update_fields=['price'])
                    updated += 1
                
                messages.success(
                    request,
                    f"Updated pricing for {updated} variant(s)."
                )
                return self.response_post_save_change(request, product)
        else:
            form = BulkVariantPricingForm()
        
        context = {
            **self.admin_site.each_context(request),
            'title': f'Bulk Pricing for: {product.name}',
            'product': product,
            'form': form,
            'variants': variants,
            'opts': self.opts,
        }
        
        return TemplateResponse(
            request,
            'admin/catalog/product/bulk_pricing.html',
            context
        )
    
    # =========================================================================
    # API Endpoints
    # =========================================================================
    
    def available_options_api(self, request):
        """API endpoint to get available options for variant generation."""
        options = Option.objects.prefetch_related('values').all()
        
        data = []
        for option in options:
            option_data = {
                'id': str(option.id),
                'name': option.name,
                'slug': option.slug if hasattr(option, 'slug') else None,
                'values': [
                    {
                        'id': str(val.id),
                        'value': val.value,
                    }
                    for val in option.values.all()
                ]
            }
            data.append(option_data)
        
        return JsonResponse({'options': data})
    
    # =========================================================================
    # Custom Actions
    # =========================================================================
    
    actions = ImportExportEnhancedModelAdmin.actions + [
        'generate_variants_action',
        'bulk_pricing_action',
        'duplicate_products',
    ]
    
    @admin.action(description='Generate variants for selected products')
    def generate_variants_action(self, request, queryset):
        """Admin action to bulk generate variants."""
        selected = queryset.count()
        if selected > 1:
            self.message_user(
                request,
                f"Please select only one product at a time for variant generation.",
                level='warning'
            )
            return
        
        product = queryset.first()
        return self.response_post_save_change(request, product)
    
    @admin.action(description='Bulk pricing for selected products')
    def bulk_pricing_action(self, request, queryset):
        """Admin action to bulk update pricing."""
        selected = queryset.count()
        if selected > 1:
            self.message_user(
                request,
                f"Please select only one product at a time for bulk pricing.",
                level='warning'
            )
            return
        
        product = queryset.first()
        return self.response_post_save_change(request, product)
    
    @admin.action(description='Duplicate selected products')
    def duplicate_products(self, request, queryset):
        """Duplicate selected products with all related data."""
        for product in queryset:
            product.duplicate()
        
        self.message_user(
            request,
            f"Successfully duplicated {queryset.count()} product(s)."
        )
    
    # =========================================================================
    # Rendering Helpers
    # =========================================================================
    
    def render_change_form(self, request, context, add=False, change=False, form_url='', obj=None):
        """Enhance context with additional data for templates."""
        # Add available options to context
        context['available_options'] = Option.objects.prefetch_related('values').all()
        context['available_attributes'] = Attribute.objects.prefetch_related('values').all()
        
        # Add tool URLs
        if obj:
            context['variant_generator_url'] = reverse(
                'admin:catalog_product_variant_generator',
                kwargs={'pk': obj.pk}
            )
            context['bulk_pricing_url'] = reverse(
                'admin:catalog_product_bulk_pricing',
                kwargs={'pk': obj.pk}
            )
        
        return super().render_change_form(request, context, add, change, form_url, obj)


# =============================================================================
# Utility Functions for Admin
# =============================================================================

def get_admin_product_stats(product):
    """Get statistics about a product for admin display."""
    stats = {
        'total_variants': product.variants.count(),
        'total_images': product.images.count(),
        'total_attributes': ProductAttributeValue.objects.filter(product=product).count(),
        'is_parent': product.variants.filter(is_default=True).exists(),
    }
    return stats
