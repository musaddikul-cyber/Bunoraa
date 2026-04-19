"""
Enhanced admin inlines for variant and attribute management.
Redesigned to make the product add/edit page variant and attribute friendly.
"""
from django import forms
from django.contrib import admin
from django.db import models
from django.utils.html import format_html, mark_safe
from django.urls import reverse
from django.utils.safestring import mark_safe as safe_mark

from .models import (
    ProductImage,
    ProductVariant,
    ProductAttributeValue,
    Attribute,
    AttributeValue,
    Option,
    OptionValue,
    VariantOptionValue,
    Product3DAsset,
    ProductMakingOf,
    CustomerPhoto,
)


# =============================================================================
# WIDGETS
# =============================================================================

class AttributeValueWidget(forms.SelectMultiple):
    """Widget for selecting attribute values, grouped by attribute."""
    
    def __init__(self, attrs=None, category_filter=True):
        super().__init__(attrs)
        self.category_filter = category_filter
    
    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        # Group choices by attribute
        choices = []
        current_attr = None
        
        for attr in Attribute.objects.prefetch_related('values').all():
            attr_group = []
            for val in attr.values.all():
                attr_group.append((str(val.pk), f"{attr.name}: {val.value}"))
            if attr_group:
                choices.append((attr.name, attr_group))
        
        context['widget']['choices'] = choices
        context['widget']['attrs']['class'] = (context['widget']['attrs'].get('class', '') + ' attribute-value-select').strip()
        return context


class OptionValueWidget(forms.SelectMultiple):
    """Widget for selecting option values per variant."""
    
    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        # Group by option type (Size, Color, etc.)
        choices = []
        for option in Option.objects.prefetch_related('values').all():
            opt_group = [(str(v.pk), v.value) for v in option.values.all()]
            if opt_group:
                choices.append((option.name, opt_group))
        
        context['widget']['choices'] = choices
        context['widget']['attrs']['class'] = (context['widget']['attrs'].get('class', '') + ' option-value-select').strip()
        return context


# =============================================================================
# ENHANCED INLINE BASE
# =============================================================================

class ReorderableInlineMixin:
    """Mixin to add drag-drop reordering capability to inlines."""
    
    class Media:
        js = (
            'https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js',
            'js/admin/sortable_inlines.js',
        )
        css = {
            'all': ('css/admin/sortable_inlines.css',)
        }
    
    drag_handle = True
    
    def get_fields(self, request, obj=None):
        fields = super().get_fields(request, obj)
        if self.drag_handle and 'ordering' in fields:
            return ['drag_handle'] + [f for f in fields if f != 'drag_handle']
        return fields
    
    def get_readonly_fields(self, request, obj=None):
        readonly = list(super().get_readonly_fields(request, obj) or [])
        if self.drag_handle:
            readonly.append('drag_handle')
        return readonly
    
    def drag_handle(self, obj=None):
        if obj is None:
            return format_html(
                '<span class="drag-handle" title="Drag to reorder">{}</span>',
                '☰'
            )
        return format_html(
            '<span class="drag-handle" title="Drag to reorder">{}</span>',
            '☰'
        )
    drag_handle.short_description = ''


# =============================================================================
# PRODUCT ATTRIBUTE VALUE INLINE
# =============================================================================

class ProductAttributeValueForm(forms.ModelForm):
    """Form for managing product attribute values with filtering."""
    
    class Meta:
        model = ProductAttributeValue
        fields = ['value']
        widgets = {
            'value': forms.Select(attrs={
                'class': 'attribute-value-select vSelectField',
                'style': 'min-width: 250px;',
            })
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enhance the value field queryset
        self.fields['value'].queryset = AttributeValue.objects.select_related('attribute').all()
        self.fields['value'].label_from_instance = lambda obj: f"{obj.attribute.name}: {obj.value}"


class ProductAttributeValueInline(admin.TabularInline):
    """Enhanced inline for managing product attributes."""
    
    model = ProductAttributeValue
    form = ProductAttributeValueForm
    extra = 1
    min_num = 0
    
    fields = ('value', 'attribute_preview')
    readonly_fields = ('attribute_preview',)
    
    classes = ['collapse', 'attribute-values-inline']
    verbose_name = "Product Attribute"
    verbose_name_plural = "Product Attributes"
    
    class Media:
        js = ('js/admin/attribute_manager.js',)
        css = {
            'all': ('css/admin/attribute_manager.css',)
        }
    
    def attribute_preview(self, obj):
        """Display formatted attribute information."""
        if obj.value:
            return format_html(
                '<span class="attribute-badge">'
                '<span class="attr-name">{}</span>: '
                '<span class="attr-value">{}</span>'
                '</span>',
                obj.value.attribute.name,
                obj.value.value
            )
        return "-"
    attribute_preview.short_description = "Attribute"
    
    def get_formset(self, request, obj=None, **kwargs):
        """Customize formset with category-aware filtering."""
        formset = super().get_formset(request, obj, **kwargs)
        
        class EnhancedFormSet(formset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.category_id = getattr(obj, 'primary_category_id', None) if obj else None
        
        return EnhancedFormSet


# =============================================================================
# PRODUCT VARIANT ENHANCED INLINE
# =============================================================================

class ProductVariantForm(forms.ModelForm):
    """Enhanced form for product variants with option management."""
    
    # Add a field for selecting multiple option values
    option_values_field = forms.ModelMultipleChoiceField(
        queryset=OptionValue.objects.select_related('option').all(),
        required=False,
        widget=forms.SelectMultiple(attrs={
            'class': 'option-value-select vSelectField',
            'style': 'min-width: 200px; height: 80px;',
        }),
        label="Options (Size, Color, etc.)"
    )
    
    class Meta:
        model = ProductVariant
        fields = ['sku', 'price', 'stock_quantity', 'is_default', 'option_values_field']
        widgets = {
            'sku': forms.TextInput(attrs={'placeholder': 'Auto-generated if empty', 'class': 'vTextField'}),
            'price': forms.NumberInput(attrs={'placeholder': 'Use product price', 'step': '0.01', 'class': 'vTextField'}),
            'stock_quantity': forms.NumberInput(attrs={'class': 'vIntegerField', 'style': 'width: 80px;'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        instance = kwargs.get('instance')
        
        # Format option value choices
        self.fields['option_values_field'].queryset = OptionValue.objects.select_related('option').all()
        self.fields['option_values_field'].label_from_instance = lambda obj: f"{obj.option.name}: {obj.value}"
        
        # Pre-populate if editing existing variant
        if instance:
            self.fields['option_values_field'].initial = instance.option_values.all()
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        
        if commit:
            instance.save()
            # Save many-to-many relationships
            self.save_m2m()
        
        return instance
    
    def save_m2m(self):
        """Save option values relationship."""
        if hasattr(self, 'cleaned_data'):
            option_values = self.cleaned_data.get('option_values_field', [])
            if self.instance.pk:
                self.instance.option_values.set(option_values)


class ProductVariantEnhancedInline(ReorderableInlineMixin, admin.TabularInline):
    """
    Enhanced variant inline with:
    - SKU, Price, Stock management
    - Option value selection (Size, Color, Material, etc.)
    - Stock status visualization
    - Visual badges for default variant
    """
    
    model = ProductVariant
    form = ProductVariantForm
    extra = 0  # Start with 0, allow adding
    min_num = 0
    max_num = 100  # Prevent abuse
    
    fields = [
        'drag_handle',
        'sku',
        'option_values_field',
        'price',
        'stock_quantity',
        'stock_status_badge',
        'is_default',
        'variant_preview'
    ]
    readonly_fields = ['drag_handle', 'stock_status_badge', 'variant_preview']
    
    ordering = ['-is_default', 'sku']
    
    classes = ['variant-inline-enhanced']
    verbose_name = "Product Variant"
    verbose_name_plural = "Product Variants"
    
    template = 'admin/catalog/product/edit_inline/tabular_variant.html'
    
    class Media:
        js = (
            'https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js',
            'js/admin/variant_manager.js',
            'js/admin/sortable_inlines.js',
        )
        css = {
            'all': ('css/admin/variant_manager.css',)
        }
    
    def stock_status_badge(self, obj):
        """Visual stock status indicator."""
        if obj.stock_quantity <= 0:
            return mark_safe(
                '<span class="status-badge out-of-stock">'
                '<span class="status-dot"></span> Out</span>'
            )
        elif obj.stock_quantity < 10:
            return format_html(
                '<span class="status-badge low-stock" title="Only {} remaining">'
                '<span class="status-dot"></span> Low ({})</span>',
                obj.stock_quantity,
                obj.stock_quantity
            )
        return format_html(
            '<span class="status-badge in-stock">'
            '<span class="status-dot"></span> In Stock ({})</span>',
            obj.stock_quantity
        )
    stock_status_badge.short_description = "Stock"
    
    def variant_preview(self, obj):
        """Show variant options summary."""
        if obj.pk and obj.option_values.exists():
            options = []
            for ov in obj.option_values.select_related('option').all():
                options.append(f"{ov.option.name}: {ov.value}")
            return format_html(
                '<span class="variant-options" title="{}">{}</span>',
                '\n'.join(options),
                ' • '.join(options[:3]) + ('...' if len(options) > 3 else '')
            )
        return mark_safe('<span class="variant-no-options">-</span>')
    variant_preview.short_description = "Configuration"
    
    def get_formset(self, request, obj=None, **kwargs):
        """Pass product instance to formset for context."""
        formset = super().get_formset(request, obj, **kwargs)
        formset.product_instance = obj
        return formset


# =============================================================================
# ENHANCED IMAGE INLINE WITH DRAG-DROP
# =============================================================================

class ProductImageEnhancedInline(ReorderableInlineMixin, admin.TabularInline):
    """Image inline with drag-drop reordering and better preview."""
    
    model = ProductImage
    extra = 1
    min_num = 0
    max_num = 20
    
    fields = [
        'drag_handle',
        'thumbnail_preview',
        'image_preview_large',
        'image',
        'alt_text',
        'is_primary',
        'ordering'
    ]
    readonly_fields = ['drag_handle', 'thumbnail_preview', 'image_preview_large']
    
    ordering = ['ordering', 'id']
    
    classes = ['collapse', 'image-inline-enhanced']
    template = 'admin/catalog/product/edit_inline/tabular_image.html'
    
    class Media:
        js = (
            'https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js',
            'js/admin/sortable_inlines.js',
            'js/admin/image_manager.js',
        )
        css = {
            'all': ('css/admin/image_manager.css',)
        }
    
    def thumbnail_preview(self, obj):
        """Small inline preview."""
        if obj.image:
            return format_html(
                '<img src="{}" class="inline-thumb" '
                'style="max-height: 50px; max-width: 70px; object-fit: cover; border-radius: 4px;" />',
                obj.image.url
            )
        return mark_safe('<span class="no-image">No image</span>')
    thumbnail_preview.short_description = "Thumb"
    
    def image_preview_large(self, obj):
        """Larger preview with zoom capability."""
        if obj.image:
            return format_html(
                '<a href="{}" target="_blank" class="image-preview-link">'
                '<img src="{}" class="inline-preview-large" '
                'style="max-height: 120px; max-width: 150px; object-fit: contain; border: 1px solid #ddd; border-radius: 4px;" />'
                '</a>',
                obj.image.url,
                obj.image.url
            )
        return "-"
    image_preview_large.short_description = "Preview"


# =============================================================================
# UTILITY INLINES (Minimal Enhancements)
# =============================================================================

class Product3DAssetEnhancedInline(admin.TabularInline):
    """Minimal enhancement for 3D assets."""
    
    model = Product3DAsset
    extra = 0
    fields = ['file', 'file_type', 'is_ar_compatible', 'validated', 'preview_link']
    readonly_fields = ['preview_link']
    
    classes = ['collapse']
    
    def preview_link(self, obj):
        if obj.file:
            return format_html(
                '<a href="{}" target="_blank" class="button">Preview</a>',
                obj.file.url
            )
        return "-"
    preview_link.short_description = "Preview"


class ProductMakingOfInline(admin.TabularInline):
    """Making-of content inline."""
    
    model = ProductMakingOf
    extra = 0
    fields = ['order', 'title', 'description', 'image', 'video_url']
    ordering = ['order']
    
    classes = ['collapse']


class CustomerPhotoInline(admin.TabularInline):
    """Customer-submitted photos inline."""
    
    model = CustomerPhoto
    extra = 0
    fields = ['thumbnail_preview', 'image', 'description', 'status', 'created_at']
    readonly_fields = ['thumbnail_preview', 'created_at']
    ordering = ['-created_at']
    
    classes = ['collapse']
    
    def thumbnail_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height: 50px; max-width: 70px; object-fit: cover; border-radius: 4px;" />',
                obj.image.url
            )
        return "-"
    thumbnail_preview.short_description = "Photo"


# =============================================================================
# BACKWARDS COMPATIBILITY
# =============================================================================

# Keep old inline names for backwards compatibility during transition
class ProductVariantInline(ProductVariantEnhancedInline):
    """Alias for backwards compatibility."""
    pass


class ProductImageInline(ProductImageEnhancedInline):
    """Alias for backwards compatibility."""
    pass
