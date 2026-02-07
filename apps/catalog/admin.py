from django.contrib import admin, messages
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from django.db.models import Sum, Count, F, Q
from django.urls import reverse
from django.http import HttpResponse
import csv

from core.admin_mixins import (
    EnhancedModelAdmin,
    EnhancedTabularInline,
    ExportCSVMixin,
    BulkActivateMixin,
    BulkFeaturedMixin,
    StatusBadgeMixin,
    DateRangeFilter,
    PriceRangeFilter,
    StockFilter,
    format_currency,
    format_number,
    truncate_text,
)

from .models import (
    Category,
    Product,
    ProductVariant,
    ProductImage,
    Badge,
    ShippingMaterial,
    Spotlight,
    ProductMakingOf,
    Product3DAsset,
    CustomerPhoto,
    ProductQuestion, 
    ProductAnswer,   
    Attribute,
    AttributeValue,
    Facet,
    CategoryFacet,
    Tag,
)


# =============================================================================
# INLINES
# =============================================================================

class ProductImageInline(EnhancedTabularInline):
    model = ProductImage
    extra = 1
    fields = ("image", "thumbnail_preview", "alt_text", "is_primary", "ordering")
    readonly_fields = ("thumbnail_preview",)
    ordering = ["ordering"]
    
    def thumbnail_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height: 50px; max-width: 80px; object-fit: cover; border-radius: 4px;" />',
                obj.image.url
            )
        return "-"
    thumbnail_preview.short_description = "Preview"


class ProductVariantInline(EnhancedTabularInline):
    model = ProductVariant
    extra = 0
    fields = ("sku", "price", "stock_quantity", "stock_status_badge", "is_default")
    readonly_fields = ("stock_status_badge",)
    
    def stock_status_badge(self, obj):
        if obj.stock_quantity <= 0:
            return format_html('<span style="color: #dc2626; font-weight: 600;">Out of Stock</span>')
        elif obj.stock_quantity < 10:
            return format_html('<span style="color: #d97706; font-weight: 600;">Low ({}))</span>', obj.stock_quantity)
        return format_html('<span style="color: #16a34a; font-weight: 600;">In Stock ({})</span>', obj.stock_quantity)
    stock_status_badge.short_description = "Status"


class Product3DAssetInline(EnhancedTabularInline):
    model = Product3DAsset
    extra = 0
    fields = ("file", "file_type", "validated", "is_ar_compatible")


class ProductMakingOfInline(EnhancedTabularInline):
    model = ProductMakingOf
    extra = 0
    fields = ("order", "title", "description", "image", "video_url")
    ordering = ["order"]


class CustomerPhotoInline(EnhancedTabularInline):
    model = CustomerPhoto
    extra = 0
    fields = ("image", "thumbnail_preview", "description", "status")
    readonly_fields = ("thumbnail_preview",)
    ordering = ["-created_at"]

    def thumbnail_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height: 50px; max-width: 80px; object-fit: cover; border-radius: 4px;" />',
                obj.image.url
            )
        return "-"
    thumbnail_preview.short_description = "Preview"


class ProductAnswerInline(EnhancedTabularInline):
    model = ProductAnswer
    extra = 0
    fields = ("user", "answer_text", "status")
    readonly_fields = ("user",)
    ordering = ["created_at"]


class ProductQuestionInline(EnhancedTabularInline):
    model = ProductQuestion
    extra = 0
    fields = ("user", "question_text", "status")
    readonly_fields = ("user",)
    ordering = ["-created_at"]
    inlines = [ProductAnswerInline] # Nested inline for answers


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "parent", "display_path", "depth", "product_count", "is_visible", "aspect_ratio")
    search_fields = ("name", "slug")
    list_filter = ("is_visible", "aspect_ratio")
    prepopulated_fields = {"slug": ("name",)}
    ordering = ["depth", "name"]
    
    actions = [
        "seed_default_tree",
        "rebuild_paths",
        "make_visible",
        "make_hidden",
        "export_selected_csv",
    ]

    def display_path(self, obj):
        crumbs = obj.breadcrumbs()
        return " / ".join([c.name for c in crumbs])

    display_path.short_description = "Path"

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("parent")

    def seed_default_tree(self, request, queryset):
        """Admin action to run the default category seeder (idempotent)."""
        from django.core.management import call_command
        try:
            call_command("seed_categories")
            self.message_user(request, "Default category tree seeded successfully (idempotent).")
        except Exception as e:
            self.message_user(request, f"Error seeding categories: {e}", level="error")
    seed_default_tree.short_description = "Seed default category tree"

    def rebuild_paths(self, request, queryset):
        """Admin action to rebuild path and depth for selected categories."""
        if queryset.count() == 0:
            # Rebuild entire tree
            fixed = Category.rebuild_all_paths() if hasattr(Category, "rebuild_all_paths") else 0
            self.message_user(request, f"Rebuilt paths for entire tree, fixed {fixed} nodes.")
            return

        fixed = 0
        for cat in queryset:
            if hasattr(cat, "rebuild_subtree"):
                fixed += 1 if cat.rebuild_subtree() else 0
            else:
                # Fallback: manually update depth and path
                cat.depth = cat.calculate_depth() if hasattr(cat, "calculate_depth") else 0
                cat.save(update_fields=["depth"])
                fixed += 1

        self.message_user(request, f"Rebuilt paths for {fixed} selected categories.")
    rebuild_paths.short_description = "Rebuild path/depth for selected categories"

    def make_visible(self, request, queryset):
        """Mark selected categories as visible."""
        updated = queryset.update(is_visible=True)
        self.message_user(request, f"Marked {updated} categories as visible.")
    make_visible.short_description = "Mark selected as visible"

    def make_hidden(self, request, queryset):
        """Mark selected categories as hidden."""
        updated = queryset.update(is_visible=False)
        self.message_user(request, f"Marked {updated} categories as hidden.")
    make_hidden.short_description = "Mark selected as hidden"

    def export_selected_csv(self, request, queryset):
        """Export selected categories to CSV."""
        import csv
        import tempfile

        fd, path = tempfile.mkstemp(prefix="categories_", suffix=".csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "name", "slug", "parent_id", "depth", "is_visible"])
            for c in queryset.order_by("depth", "name"):
                parent_id = c.parent_id if c.parent_id else ""
                writer.writerow([c.id, c.name, c.slug, parent_id, c.depth, c.is_visible])

        self.message_user(request, f"Exported {queryset.count()} categories to {path}")
    export_selected_csv.short_description = "Export selected categories as CSV"


from .forms import ProductAdminForm 

@admin.register(Product)
class ProductAdmin(EnhancedModelAdmin, BulkActivateMixin, BulkFeaturedMixin):
    form = ProductAdminForm 
    list_display = (
        "thumbnail_preview", "name", "sku", "primary_category_display", 
        "price_display", "stock_status", "performance_stats", "is_active",
        "is_active_badge", "is_featured_badge", "created_at"
    )
    list_display_links = ("thumbnail_preview", "name")
    search_fields = ("name", "sku", "description")
    list_filter = (
        "is_active", "is_featured", "is_bestseller", "is_new_arrival",
        StockFilter, PriceRangeFilter, "aspect_ratio", "primary_category"
    )
    inlines = [ProductImageInline, ProductVariantInline, Product3DAssetInline, ProductMakingOfInline, CustomerPhotoInline, ProductQuestionInline]
    prepopulated_fields = {"slug": ("name",)}
    date_hierarchy = "created_at"
    list_per_page = 25
    list_editable = ("is_active",)
    save_on_top = True
    filter_horizontal = ("categories", "tags")
    
    # Export fields
    export_fields = ['sku', 'name', 'price', 'sale_price', 'stock_quantity', 
                     'is_active', 'is_featured', 'views_count', 'sales_count']
    
    actions = [
        'export_as_csv', 'export_as_json',
        'activate_selected', 'deactivate_selected',
        'mark_featured', 'unmark_featured',
        'mark_bestseller', 'unmark_bestseller',
        'mark_new_arrival', 'unmark_new_arrival',
        'duplicate_products',
        'recalculate_stats',
    ]

    fieldsets = (
        (_('Basic Information'), {
            "fields": ("name", "slug", "sku", "description", "short_description"),
            "classes": ("wide",)
        }),
        (_('Categories & Tags'), {
            "fields": ("primary_category", "categories", "tags"),
            "classes": ("wide",)
        }),
        (_('Pricing'), {
            "fields": ("price", "sale_price", "cost", "currency"),
            "description": "Set product pricing. Sale price will override regular price when set."
        }),
        (_('Inventory'), {
            "fields": ("stock_quantity", "low_stock_threshold"),
            "description": "Manage inventory levels and tracking."
        }),
        (_('Shipping'), {
            "fields": ("weight", "length", "width", "height", "shipping_material"),
        }),
        (_('Display'), {
            "fields": ("aspect_ratio",),
        }),
        (_('Sustainability'), {
            "fields": (
                "carbon_footprint_kg", "recycled_content_percentage", 
                "sustainability_score", "ethical_sourcing_notes", "eco_certifications"
            ),
            "classes": ("collapse",),
        }),
        (_('Mobile & Voice'), {
            "fields": ("is_mobile_optimized", "voice_keywords"),
            "classes": ("collapse",),
        }),
        (_('Status & Flags'), {
            "fields": ("is_active", "is_featured", "is_bestseller", "is_new_arrival", "can_be_customized"),
        }),
        (_('SEO'), {
            "fields": ("meta_title", "meta_description"),
            "classes": ("collapse",),
        }),
    )

    class Media:
        css = {
            "all": ("css/admin/category_tree_widget.css",),
        }
        js = (
            "admin/js/vendor/jquery/jquery.min.js",
            "admin/js/jquery.init.js",
            "js/admin/category_tree_widget.js",
        )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'primary_category', 'shipping_material'
        ).prefetch_related('categories', 'images')

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        
        # After saving the product, ensure primary_category is also in categories M2M
        if obj.primary_category and obj.primary_category not in obj.categories.all():
            obj.categories.add(obj.primary_category)
            messages.info(request, _("Primary category automatically added to product categories."))
        
        # Also ensure product_count is updated for the primary category
        if change:
            # If primary category changed, update counts for old and new
            old_primary_category_id = form.initial.get('primary_category')
            if old_primary_category_id and old_primary_category_id != obj.primary_category_id:
                old_primary_category = Category.objects.get(pk=old_primary_category_id)
                old_primary_category.product_count = old_primary_category.products.count()
                old_primary_category.save()
            
            # Update new primary category count
            if obj.primary_category:
                obj.primary_category.product_count = obj.primary_category.products.count()
                obj.primary_category.save()
        else: # New product
            if obj.primary_category:
                obj.primary_category.product_count = obj.primary_category.products.count()
                obj.primary_category.save()
    
    def thumbnail_preview(self, obj):
        # Try to get primary image or first image
        primary_image = obj.images.filter(is_primary=True).first() or obj.images.first()
        if primary_image and primary_image.image:
            return format_html(
                '<img src="{}" style="max-height: 40px; max-width: 60px; object-fit: cover; border-radius: 4px;" />',
                primary_image.image.url
            )
        return format_html('<span style="color: #9ca3af;">No image</span>')
    thumbnail_preview.short_description = ""
    
    def price_display(self, obj):
        if obj.sale_price:
            return format_html(
                '<div><span style="text-decoration: line-through; color: #9ca3af;">{}</span><br/>'
                '<span style="color: #dc2626; font-weight: 600;">{}</span></div>',
                format_currency(obj.price), format_currency(obj.sale_price)
            )
        return format_currency(obj.price)
    price_display.short_description = _("Price")
    price_display.admin_order_field = "price"
    
    def stock_status(self, obj):
        qty = obj.stock_quantity or 0
        threshold = getattr(obj, 'low_stock_threshold', 10) or 10
        
        if qty <= 0:
            return format_html(
                '<span style="display: inline-flex; align-items: center; padding: 2px 8px; '
                'background-color: #fee2e2; color: #991b1b; border-radius: 4px; font-size: 11px; font-weight: 600;">'
                'Out of Stock</span>'
            )
        elif qty <= threshold:
            return format_html(
                '<span style="display: inline-flex; align-items: center; padding: 2px 8px; '
                'background-color: #fef3c7; color: #92400e; border-radius: 4px; font-size: 11px; font-weight: 600;">'
                'Low ({})</span>',
                qty
            )
        return format_html(
            '<span style="display: inline-flex; align-items: center; padding: 2px 8px; '
            'background-color: #dcfce7; color: #166534; border-radius: 4px; font-size: 11px; font-weight: 600;">'
            '{}</span>',
            qty
        )
    stock_status.short_description = _("Stock")
    stock_status.admin_order_field = "stock_quantity"
    
    def performance_stats(self, obj):
        views = obj.views_count or 0
        sales = obj.sales_count or 0
        wishlist = obj.wishlist_count or 0
        
        return format_html(
            '<div style="font-size: 11px; line-height: 1.4;">'
            '<span title="Views">👁️ {}</span> · '
            '<span title="Sales">🛒 {}</span> · '
            '<span title="Wishlist">❤️ {}</span></div>',
            format_number(views), format_number(sales), format_number(wishlist)
        )
    performance_stats.short_description = _("Stats")
    
    def is_active_badge(self, obj):
        if obj.is_active:
            return format_html('<span style="color: #16a34a;">●</span>')
        return format_html('<span style="color: #dc2626;">●</span>')
    is_active_badge.short_description = _("Active")
    is_active_badge.admin_order_field = "is_active"
    
    def is_featured_badge(self, obj):
        if obj.is_featured:
            return format_html('<span style="color: #eab308;">⭐</span>')
        return ""
    is_featured_badge.short_description = _("Featured")
    is_featured_badge.admin_order_field = "is_featured"

    def primary_category_display(self, obj):
        if obj.primary_category:
            return format_html(
                '<a href="{}">{}</a>',
                reverse('admin:catalog_category_change', args=[obj.primary_category.pk]),
                obj.primary_category.name
            )
        return "-"
    primary_category_display.short_description = _("Category")
    primary_category_display.admin_order_field = "primary_category__name"
    
    # Bulk Actions
    def mark_bestseller(self, request, queryset):
        updated = queryset.update(is_bestseller=True)
        self.message_user(request, f'{updated} products marked as bestseller.', messages.SUCCESS)
    mark_bestseller.short_description = _("Mark as bestseller")
    
    def unmark_bestseller(self, request, queryset):
        updated = queryset.update(is_bestseller=False)
        self.message_user(request, f'{updated} products unmarked as bestseller.', messages.SUCCESS)
    unmark_bestseller.short_description = _("Remove bestseller status")
    
    def mark_new_arrival(self, request, queryset):
        updated = queryset.update(is_new_arrival=True)
        self.message_user(request, f'{updated} products marked as new arrival.', messages.SUCCESS)
    mark_new_arrival.short_description = _("Mark as new arrival")
    
    def unmark_new_arrival(self, request, queryset):
        updated = queryset.update(is_new_arrival=False)
        self.message_user(request, f'{updated} products unmarked as new arrival.', messages.SUCCESS)
    unmark_new_arrival.short_description = _("Remove new arrival status")
    
    def duplicate_products(self, request, queryset):
        for product in queryset:
            # Create a copy
            product.pk = None
            product.sku = f"{product.sku}-copy"
            product.slug = f"{product.slug}-copy"
            product.is_active = False
            product.save()
        self.message_user(request, f'{queryset.count()} products duplicated (inactive).', messages.SUCCESS)
    duplicate_products.short_description = _("Duplicate selected products")
    
    def recalculate_stats(self, request, queryset):
        """Recalculate view/sales/wishlist counts from actual data."""
        from apps.analytics.models import ProductView
        from apps.orders.models import OrderItem
        
        updated = 0
        for product in queryset:
            try:
                # Recalculate views
                views = ProductView.objects.filter(product=product).count()
                # Recalculate sales
                sales = OrderItem.objects.filter(
                    product=product, 
                    order__status='delivered'
                ).aggregate(total=Sum('quantity'))['total'] or 0
                
                product.views_count = views
                product.sales_count = sales
                product.save(update_fields=['views_count', 'sales_count'])
                updated += 1
            except Exception:
                pass
        
        self.message_user(request, f'Recalculated stats for {updated} products.', messages.SUCCESS)
    recalculate_stats.short_description = _("Recalculate statistics")


@admin.register(ShippingMaterial)
class ShippingMaterialAdmin(admin.ModelAdmin):
    list_display = ("name", "eco_score", "created_at")
    search_fields = ("name",)


@admin.register(Badge)
class BadgeAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "is_active", "start", "end", "priority")
    search_fields = ("name", "slug")
    prepopulated_fields = {"slug": ("name",)}


@admin.register(Spotlight)
class SpotlightAdmin(admin.ModelAdmin):
    list_display = ("name", "placement", "product", "category", "start", "end", "priority", "is_active")
    list_filter = ("placement", "is_active")


@admin.register(Product3DAsset)
class Product3DAssetAdmin(admin.ModelAdmin):
    list_display = ("product", "file_type", "validated", "is_ar_compatible", "uploaded_at")
    readonly_fields = ("uploaded_at",)


@admin.register(Attribute)
class AttributeAdmin(admin.ModelAdmin):
    list_display = ("name", "slug")
    search_fields = ("name", "slug")


@admin.register(AttributeValue)
class AttributeValueAdmin(admin.ModelAdmin):
    list_display = ("attribute", "value")
    search_fields = ("value",)


@admin.register(Facet)
class FacetAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "type")
    search_fields = ("name",)


@admin.register(CategoryFacet)
class CategoryFacetAdmin(admin.ModelAdmin):
    list_display = ("category", "facet")


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ("name",)
    search_fields = ("name",)


@admin.register(ProductQuestion)
class ProductQuestionAdmin(admin.ModelAdmin):
    list_display = ('product', 'user', 'question_text', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('product__name', 'user__email', 'question_text')
    readonly_fields = ('product', 'user', 'created_at', 'updated_at')
    ordering = ['-created_at']
    inlines = [ProductAnswerInline]

    actions = ['approve_questions', 'reject_questions']

    def approve_questions(self, request, queryset):
        updated = queryset.update(status='approved')
        self.message_user(request, f'{updated} questions approved.')
    approve_questions.short_description = 'Approve selected questions'

    def reject_questions(self, request, queryset):
        updated = queryset.update(status='rejected')
        self.message_user(request, f'{updated} questions rejected.')
    reject_questions.short_description = 'Reject selected questions'


@admin.register(ProductAnswer)
class ProductAnswerAdmin(admin.ModelAdmin):
    list_display = ('question', 'user', 'answer_text', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('question__question_text', 'user__email', 'answer_text')
    readonly_fields = ('question', 'user', 'created_at', 'updated_at')
    ordering = ['created_at']

    actions = ['approve_answers', 'reject_answers']

    def approve_answers(self, request, queryset):
        updated = queryset.update(status='approved')
        self.message_user(request, f'{updated} answers approved.')
    approve_answers.short_description = 'Approve selected answers'

    def reject_answers(self, request, queryset):
        updated = queryset.update(status='rejected')
        self.message_user(request, f'{updated} answers rejected.')
    reject_answers.short_description = 'Reject selected answers'


# Inline registered for ProductVariant to ensure attribute M2M is manageable
@admin.register(ProductVariant)
class ProductVariantAdmin(admin.ModelAdmin):
    list_display = ("sku", "product", "price", "stock_quantity", "is_default")
    search_fields = ("sku", "product__name")


@admin.register(CustomerPhoto)
class CustomerPhotoAdmin(admin.ModelAdmin):
    list_display = ('product', 'user', 'status', 'created_at', 'thumbnail_preview')
    list_filter = ('status', 'created_at')
    search_fields = ('product__name', 'user__email', 'description')
    readonly_fields = ('product', 'user', 'created_at', 'updated_at', 'thumbnail_preview')
    
    actions = ['approve_photos', 'reject_photos']

    def thumbnail_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height: 80px; max-width: 120px; object-fit: cover; border-radius: 4px;" />',
                obj.image.url
            )
        return "-"
    thumbnail_preview.short_description = "Image Preview"

    def approve_photos(self, request, queryset):
        updated = queryset.update(status='approved')
        self.message_user(request, f'{updated} photos approved.')
    approve_photos.short_description = 'Approve selected photos'

    def reject_photos(self, request, queryset):
        updated = queryset.update(status='rejected')
        self.message_user(request, f'{updated} photos rejected.')
    reject_photos.short_description = 'Reject selected photos'
