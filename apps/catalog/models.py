import uuid
from decimal import Decimal
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models, transaction
from django.db.models import Q, F, Case, When, Value, IntegerField, Avg, Count
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey
from django.urls import reverse
from django.utils import timezone
from django.utils.text import slugify
from django.db.models.functions import Lower
from django.core.cache import cache
from django.core.validators import MinValueValidator, MaxValueValidator

from .managers import SoftDeleteManager


CURRENCY_DEFAULT = getattr(settings, "DEFAULT_CURRENCY", "BDT")

# Standard aspect ratio choices for product/category card display
ASPECT_CHOICES = (
    ("1:1", "1:1"),
    ("4:3", "4:3"),
    ("16:9", "16:9"),
    ("3:2", "3:2"),
    ("free", "Free / responsive"),
)

# Aspect unit choices for custom dimensions
ASPECT_UNIT_CHOICES = [
    ('ratio', 'Ratio (unitless)'),
    ('in', 'Inches'),
    ('ft', 'Feet'),
    ('cm', 'Centimeters'),
    ('mm', 'Millimeters'),
    ('px', 'Pixels'),
]


class ProductFilterGroup(models.Model):
    """
    Groups filters for better organization.
    E.g., "Embroidery Details", "Materials", "Customization"
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    name = models.CharField(max_length=100)
    slug = models.SlugField(max_length=120, unique=True)
    display_order = models.PositiveIntegerField(default=0)
    
    class Meta:
        verbose_name = 'filter group'
        verbose_name_plural = 'filter groups'
        ordering = ['display_order']
    
    def __str__(self):
        return self.name


class TimeStampedMixin(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Category(TimeStampedMixin):
    """Category model using materialized path.

    Utilities:
    - get_ancestors(), get_descendants(), breadcrumbs(), get_slug_path(), get_absolute_url(), get_tree()
    - Caching for category tree to avoid repeated build costs
    """

    """Materialized path category. Path stores UUIDs separated by '/'.

    Examples:
        root: path=<id>
        child: path=<root_id>/<child_id>
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200)
    parent = models.ForeignKey("self", null=True, blank=True, related_name="children", on_delete=models.PROTECT)
    path = models.CharField(max_length=2000, db_index=True, editable=False)
    depth = models.PositiveSmallIntegerField(default=0, db_index=True)

    # Visibility & soft-delete
    is_visible = models.BooleanField(default=True, db_index=True)
    is_deleted = models.BooleanField(default=False, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    # SEO
    meta_title = models.CharField(max_length=255, blank=True)
    meta_description = models.CharField(max_length=500, blank=True)
    meta_keywords = models.CharField(max_length=500, blank=True)

    # image/icon
    image = models.ImageField(null=True, blank=True, upload_to="catalog/category_images/")
    icon = models.CharField(max_length=100, blank=True)
    # Display preferences
    aspect_ratio = models.CharField(max_length=10, choices=ASPECT_CHOICES, default="1:1", blank=True)
    
    # Custom aspect ratio dimensions (for inheritance and precise control)
    aspect_width = models.DecimalField(
        'aspect width', max_digits=8, decimal_places=4, null=True, blank=True,
        help_text='Width value used to compute aspect ratio. Leave blank to inherit from parent.'
    )
    aspect_height = models.DecimalField(
        'aspect height', max_digits=8, decimal_places=4, null=True, blank=True,
        help_text='Height value used to compute aspect ratio. Leave blank to inherit from parent.'
    )
    aspect_unit = models.CharField(
        'aspect unit', max_length=10, choices=ASPECT_UNIT_CHOICES, default='ratio'
    )

    objects = SoftDeleteManager()

    # Denormalized counter for number of (non-deleted) products in this category (fast lookup)
    product_count = models.IntegerField(default=0, db_index=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(Lower("slug"), "parent", name="unique_category_parent_slug_ci"),
        ]
        indexes = [models.Index(fields=["slug"]), models.Index(fields=["is_visible"]), models.Index(fields=["path"]), models.Index(fields=["product_count"])]
        ordering = ["path"]
        verbose_name = "category"
        verbose_name_plural = "categories"

    def __str__(self):
        return self.name

    def clean(self):
        if self.parent and self.parent_id == self.id:
            raise ValidationError("Category cannot be parent of itself")

    def save(self, *args, **kwargs):
        # slugify
        if not self.slug:
            self.slug = slugify(self.name)[:200]

        # set path & depth
        if self.parent:
            self.path = f"{self.parent.path}/{self.id}"
            self.depth = self.parent.depth + 1
        else:
            self.path = str(self.id)
            self.depth = 0

        # Capture old path for bulk descendant updates
        old_path = None
        if self.pk:
            old_path = Category.objects.filter(pk=self.pk).values_list("path", flat=True).first()

        with transaction.atomic():
            super().save(*args, **kwargs)

            if old_path and old_path != self.path:
                # Bulk update all descendants' path and depth using in-memory transformation + bulk_update
                descendants = list(Category.objects.filter(path__startswith=old_path + "/").exclude(pk=self.pk))
                if descendants:
                    updated = []
                    for desc in descendants:
                        # replace prefix once
                        new_path = desc.path.replace(old_path, self.path, 1)
                        if new_path != desc.path:
                            desc.path = new_path
                            desc.depth = new_path.count("/")
                            updated.append(desc)
                    if updated:
                        Category.objects.bulk_update(updated, ["path", "depth"])
                # Clear cached category tree
                self.clear_tree_cache()

    # Utilities
    def get_ancestors(self, include_self=False):
        """Return ordered ancestor queryset (root -> parent)."""
        ids = [uuid.UUID(i) for i in self.path.split("/")]
        if not include_self:
            ids = ids[:-1]
        if not ids:
            return Category.objects.none()
        return Category.objects.filter(id__in=ids).order_by("depth")

    def get_slug_path(self, include_self=True):
        """Return a slash separated slug path for SEO URLs (e.g. home/pottery/vases)."""
        crumbs = self.get_ancestors(include_self=include_self)
        return "/".join([c.slug for c in crumbs])

    def get_absolute_url(self):
        """Permalink path for the category. You should wire `category-detail` in urls to accept the slug path."""
        return reverse("catalog:category-detail", args=[self.get_slug_path()])

    def get_descendants(self, include_self=False):
        """Return queryset of descendants (ordered by path)."""
        prefix = self.path
        qs = Category.objects.filter(path__startswith=prefix)
        if not include_self:
            qs = qs.exclude(id=self.id)
        return qs.order_by("path")

    @classmethod
    def get_tree(cls, use_cache=True):
        """Return a cached nested category tree structure (list of dicts)."""
        cache_key = "catalog:category_tree"
        if use_cache:
            tree = cache.get(cache_key)
            if tree is not None:
                return tree
        qs = cls.objects.filter(is_visible=True, is_deleted=False).order_by("path").select_related("parent")
        nodes = {}
        roots = []
        for c in qs:
            node = {"id": str(c.id), "name": c.name, "slug": c.slug, "children": [], "depth": c.depth}
            nodes[str(c.id)] = node
            if c.parent_id:
                parent = nodes.get(str(c.parent_id))
                if parent is not None:
                    parent["children"].append(node)
            else:
                roots.append(node)
        if use_cache:
            cache.set(cache_key, roots, 60 * 60)
        return roots

    def clear_tree_cache(self):
        cache.delete("catalog:category_tree")

    def breadcrumbs(self):
        return list(self.get_ancestors(include_self=True))

    def get_effective_aspect(self):
        """Return effective aspect (width, height, unit) inheriting from ancestors.

        Returns a dict: {'width': Decimal, 'height': Decimal, 'unit': str, 'ratio': Decimal}
        Defaults to 1:1 if nothing is set on the category chain.
        """
        # If this category has both width and height specified, use it
        if self.aspect_width and self.aspect_height:
            try:
                ratio = Decimal(self.aspect_width) / Decimal(self.aspect_height)
            except Exception:
                ratio = Decimal('1')
            return {
                'width': self.aspect_width,
                'height': self.aspect_height,
                'unit': self.aspect_unit or 'ratio',
                'ratio': ratio
            }

        # Walk up ancestors
        parent = self.parent
        while parent:
            if parent.aspect_width and parent.aspect_height:
                try:
                    ratio = Decimal(parent.aspect_width) / Decimal(parent.aspect_height)
                except Exception:
                    ratio = Decimal('1')
                return {
                    'width': parent.aspect_width,
                    'height': parent.aspect_height,
                    'unit': parent.aspect_unit or 'ratio',
                    'ratio': ratio
                }
            parent = parent.parent

        # Default 1:1
        return {
            'width': Decimal('1'),
            'height': Decimal('1'),
            'unit': 'ratio',
            'ratio': Decimal('1')
        }

    def get_products(self, include_subcategories=True, qs=None):
        from catalog.models import Product

        if qs is None:
            qs = Product.objects.active()

        if include_subcategories:
            categories = self.get_descendants(include_self=True)
            return qs.filter(categories__in=categories).distinct()
        return qs.filter(categories=self)

    def soft_delete(self):
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.save(update_fields=["is_deleted", "deleted_at"])


class Tag(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class Attribute(TimeStampedMixin):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    slug = models.SlugField(max_length=100)

    def __str__(self):
        return self.name


class AttributeValue(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    attribute = models.ForeignKey(Attribute, related_name="values", on_delete=models.CASCADE)
    value = models.CharField(max_length=200)

    class Meta:
        unique_together = ("attribute", "value")

    def __str__(self):
        return f"{self.attribute.name}: {self.value}"


# ----------------------
# Product Types
# ----------------------
class ProductType(models.Model):
    """Different product handling types (simple, variant, bundle, digital, service)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=50)
    slug = models.SlugField(max_length=50, unique=True)
    description = models.TextField(blank=True)

    class Meta:
        indexes = [models.Index(fields=["slug"])]

    def __str__(self):
        return self.name


# ----------------------
# Per-category facet cache helpers (used by attribute validation)
# ----------------------
_CATEGORY_FACETS_CACHE_TTL = 60 * 15  # 15 minutes


def _get_category_facets_cache_key(category_id):
    return f"catalog:category_facets:{category_id}"


def _get_category_facet_by_slug(category_id, facet_slug):
    """Return Facet instance for given (category_id, facet_slug) using a per-category cache."""
    if category_id is None or facet_slug is None:
        return None
    cache_key = _get_category_facets_cache_key(category_id)
    mapping = cache.get(cache_key)
    if mapping is None:
        qs = CategoryFacet.objects.filter(category_id=category_id).select_related("facet")
        mapping = {cf.facet.slug: cf.facet for cf in qs}
        cache.set(cache_key, mapping, _CATEGORY_FACETS_CACHE_TTL)
    return mapping.get(facet_slug)


def _clear_category_facets_cache(category_id):
    cache.delete(_get_category_facets_cache_key(category_id))


def _update_category_product_counts(category_ids, delta):
    """Atomically increment/decrement category.product_count and clamp at zero."""
    ids = list(category_ids or [])
    if not ids:
        return
    with transaction.atomic():
        # Apply delta (can be negative)
        Category.objects.filter(id__in=ids).update(product_count=F("product_count") + Value(delta))
        # Clamp negatives to zero
        Category.objects.filter(id__in=ids, product_count__lt=0).update(product_count=0)


class ProductQuerySet(models.QuerySet):
    def active(self):
        return self.filter(is_active=True, is_deleted=False)

    def with_stock(self):
        # product-level stock or any variant with stock
        return self.filter(Q(stock_quantity__gt=0) | Q(variants__stock_quantity__gt=0)).distinct()

    def prefetch_for_list(self):
        return self.select_related("primary_category", "artisan").prefetch_related("images", "variants", "categories", "tags")


from .managers import SoftDeleteManager


class ProductManager(SoftDeleteManager):
    def get_queryset(self):
        # select_related excluded: keep base small; use prefetch_for_list when needed
        return ProductQuerySet(self.model, using=self._db).filter(is_deleted=False)

    def active(self):
        return self.get_queryset().filter(is_active=True)

    def prefetch_for_list(self):
        return self.get_queryset().prefetch_for_list()


class Product(TimeStampedMixin):
    """Product model with variants, images, attributes and helpers.

    Behaviors:
    - Enforces sale_price < price
    - Ensures `primary_category` is one of `categories` (validated in `clean`)
    - Provides unified stock helpers `available_stock()` and `is_in_stock()`
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255)
    # ensure product slugs are unique (case-insensitive)
    sku = models.CharField(max_length=50, blank=True, null=True, unique=True)

    short_description = models.CharField(max_length=500, blank=True)
    description = models.TextField(blank=True)

    price = models.DecimalField(max_digits=12, decimal_places=2)
    sale_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    cost = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    currency = models.CharField(max_length=10, default=CURRENCY_DEFAULT)

    # Shipping & display
    weight = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    length = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    width = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    height = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    shipping_material = models.ForeignKey("ShippingMaterial", null=True, blank=True, on_delete=models.SET_NULL, related_name="products")
    aspect_ratio = models.CharField(max_length=10, choices=ASPECT_CHOICES, default="1:1", blank=True)

    stock_quantity = models.IntegerField(default=0)
    allow_backorder = models.BooleanField(default=False)
    low_stock_threshold = models.IntegerField(default=5)

    categories = models.ManyToManyField(Category, related_name="products")
    primary_category = models.ForeignKey(Category, null=True, blank=True, related_name="primary_products", on_delete=models.SET_NULL)
    tags = models.ManyToManyField(Tag, blank=True, related_name="products")

    artisan = models.ForeignKey(
        'artisans.Artisan',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='products',
        help_text='The artisan who created this product.'
    )

    attributes = models.ManyToManyField(AttributeValue, through="ProductAttributeValue", blank=True)
    related_products = models.ManyToManyField("self", blank=True)

    # Unified product type handling
    product_type = models.ForeignKey(ProductType, null=True, blank=True, on_delete=models.SET_NULL)

    # SEO & scheduled publishing
    meta_title = models.CharField(max_length=255, blank=True)
    meta_description = models.CharField(max_length=500, blank=True)
    meta_keywords = models.CharField(max_length=500, blank=True)

    # Scheduled publishing window
    publish_from = models.DateTimeField(null=True, blank=True)
    publish_until = models.DateTimeField(null=True, blank=True)

    # Sustainability & transparency
    carbon_footprint_kg = models.FloatField(null=True, blank=True, help_text="Estimated carbon footprint in kg CO2e")
    recycled_content_percentage = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    sustainability_score = models.FloatField(null=True, blank=True, db_index=True)
    ethical_sourcing_notes = models.TextField(blank=True)
    material_breakdown = models.JSONField(null=True, blank=True, help_text="JSON describing materials and percentages")
    eco_certifications = models.ManyToManyField('EcoCertification', blank=True, related_name='products')

    # Mobile / Voice
    is_mobile_optimized = models.BooleanField(default=False, db_index=True)
    voice_keywords = models.JSONField(null=True, blank=True)

    # AR / Visual Search
    is_ar_compatible = models.BooleanField(
        default=False, db_index=True,
        help_text="Product-level flag indicating presence of AR-ready 3D assets (USDZ/GLB).",
    )
    image_embedding = models.JSONField(
        null=True, blank=True,
        help_text="Optional visual embedding vector (list of floats) used for visual similarity lookups; populate via ML jobs in `ai_ml` app.",
    )
    embedding_updated_at = models.DateTimeField(null=True, blank=True)

    # Flags
    is_active = models.BooleanField(default=True, db_index=True)
    is_featured = models.BooleanField(default=False, db_index=True)
    is_bestseller = models.BooleanField(default=False, db_index=True)
    is_new_arrival = models.BooleanField(default=False, db_index=True)
    can_be_customized = models.BooleanField(default=False, help_text="Can this product be customized by customers?")

    # Soft delete
    is_deleted = models.BooleanField(default=False, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    # Counters / metrics
    views_count = models.BigIntegerField(default=0, db_index=True)
    sales_count = models.BigIntegerField(default=0, db_index=True)
    reviews_count = models.IntegerField(default=0, db_index=True)
    rating_count = models.IntegerField(default=0, db_index=True)
    average_rating = models.FloatField(default=0.0, db_index=True)
    reports_count = models.IntegerField(default=0, db_index=True)
    cart_count = models.IntegerField(default=0, db_index=True)
    wishlist_count = models.IntegerField(default=0, db_index=True)

    objects = ProductManager()

    class Meta:
        constraints = [
            models.UniqueConstraint(Lower("slug"), name="unique_product_slug_ci"),
        ]
        indexes = [
            models.Index(fields=["slug"]),
            models.Index(fields=["is_active"]),
            models.Index(fields=["primary_category"]),
            models.Index(fields=["views_count"]),
            models.Index(fields=["sales_count"]),
            models.Index(fields=["is_ar_compatible"]),
            models.Index(fields=["sustainability_score"]),
        ]

    def __str__(self):
        return self.name

    def clean(self):
        # price validations
        if self.sale_price is not None and self.sale_price >= self.price:
            raise ValidationError("Sale price must be lower than regular price")

    def soft_delete(self):
        # decrement category product_count if not already deleted
        if not self.is_deleted:
            cat_ids = list(self.categories.values_list("id", flat=True))
            if cat_ids:
                _update_category_product_counts(cat_ids, -1)
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.save(update_fields=["is_deleted", "deleted_at"])

    def restore(self):
        """Restore a soft-deleted product and increment category counts if active."""
        if self.is_deleted:
            self.is_deleted = False
            self.deleted_at = None
            self.save(update_fields=["is_deleted", "deleted_at"])
            if self.is_active:
                cat_ids = list(self.categories.values_list("id", flat=True))
                if cat_ids:
                    _update_category_product_counts(cat_ids, 1)

    def available_stock(self):
        """Return available stock for listing: if variants exist use their sum/availability, otherwise product stock."""
        if self.variants.exists():
            return sum(v.stock_quantity for v in self.variants.all())
        return self.stock_quantity

    # Counter helpers (atomic using F expressions)
    def increment_views(self, delta=1):
        Product.objects.filter(pk=self.pk).update(views_count=F("views_count") + delta)
        self.refresh_from_db(fields=["views_count"])
        return self.views_count

    def increment_sales(self, delta=1):
        Product.objects.filter(pk=self.pk).update(sales_count=F("sales_count") + delta)
        self.refresh_from_db(fields=["sales_count"])
        return self.sales_count

    def increment_reviews(self, delta=1):
        Product.objects.filter(pk=self.pk).update(reviews_count=F("reviews_count") + delta)
        self.refresh_from_db(fields=["reviews_count"])
        return self.reviews_count

    def increment_reports(self, delta=1):
        Product.objects.filter(pk=self.pk).update(reports_count=F("reports_count") + delta)
        self.refresh_from_db(fields=["reports_count"])
        return self.reports_count

    def increment_cart(self, delta=1):
        Product.objects.filter(pk=self.pk).update(cart_count=F("cart_count") + delta)
        self.refresh_from_db(fields=["cart_count"])
        return self.cart_count

    def increment_wishlist(self, delta=1):
        Product.objects.filter(pk=self.pk).update(wishlist_count=F("wishlist_count") + delta)
        self.refresh_from_db(fields=["wishlist_count"])
        return self.wishlist_count

    def is_in_stock(self):
        if self.allow_backorder:
            return True
        if self.variants.exists():
            return any(v.stock_quantity > 0 for v in self.variants.all())
        return self.stock_quantity > 0

    @property
    def current_price(self):
        return self.sale_price if (self.sale_price is not None and self.sale_price < self.price) else self.price

    @property
    def discount_percentage(self):
        if self.sale_price and self.price and self.price > 0:
            return (self.price - self.sale_price) / self.price * 100
        return Decimal(0)

    @property
    def is_on_sale(self):
        return bool(self.sale_price and self.sale_price < self.price)

    @property
    def is_low_stock(self):
        return self.stock_quantity <= self.low_stock_threshold

    @property
    def is_published(self):
        """Returns True if product is active and within the scheduled publishing window (if any)."""
        now = timezone.now()
        if not self.is_active:
            return False
        if getattr(self, "publish_from", None) and self.publish_from > now:
            return False
        if getattr(self, "publish_until", None) and self.publish_until < now:
            return False
        return True

    # ------------------ Sustainability helpers ------------------
    def compute_sustainability_score(self, save=False):
        """Compute a heuristic sustainability score (0.0 - 1.0) using available data:
        - recycled_content_percentage (higher is better)
        - carbon_footprint_kg (lower is better)
        - number of eco_certifications (bonus)

        This is intentionally simple and conservative; the `sustainability` app or an
        external service could provide a more accurate calculation later.
        """
        # Normalized recycled content (0..1)
        recycled_pct = 0.0
        if self.recycled_content_percentage is not None:
            try:
                recycled_pct = float(self.recycled_content_percentage) / 100.0
            except Exception:
                recycled_pct = 0.0
        # Carbon footprint mapping: assume sensible scale where 0 kg => best (1.0), 100+ kg => worst (0.0)
        carbon_score = 1.0
        if self.carbon_footprint_kg is not None:
            carbon = float(self.carbon_footprint_kg)
            carbon_score = max(0.0, 1.0 - min(carbon / 100.0, 1.0))
        # Certifications bonus
        cert_count = self.eco_certifications.count() if hasattr(self, "eco_certifications") else 0
        cert_bonus = min(cert_count * 0.1, 0.2)

        # Weighted aggregation
        score = (0.6 * recycled_pct) + (0.3 * carbon_score) + cert_bonus
        score = max(0.0, min(1.0, score))
        if save:
            self.sustainability_score = score
            self.save(update_fields=["sustainability_score", "updated_at"])
        return score

    # ------------------ Schema.org helpers ------------------
    def to_schema(self):
        """Return a minimal schema.org Product representation suitable for JSON-LD."""
        schema = {
            "@type": "Product",
            "name": self.name,
            "sku": self.sku,
            "description": self.short_description or self.description,
            "url": self.get_absolute_url() if hasattr(self, "get_absolute_url") else None,
            "offers": {
                "@type": "Offer",
                "price": str(self.current_price),
                "priceCurrency": self.currency,
                "availability": "http://schema.org/InStock" if self.is_in_stock() else "http://schema.org/OutOfStock",
            },
        }
        if self.primary_category:
            schema["category"] = self.primary_category.name
        if self.images.exists():
            schema["image"] = [i.image.url for i in self.images.all() if getattr(i, "image", None)]
        return schema


class ProductAttributeValue(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    value = models.ForeignKey(AttributeValue, on_delete=models.CASCADE)

    class Meta:
        unique_together = ("product", "value")
        indexes = [models.Index(fields=["product"]), models.Index(fields=["value"])]

    def clean(self):
        # Ensure attribute/value adheres to allowed facets of the product's primary category
        if not getattr(self.product, "primary_category", None):
            return
        category_id = self.product.primary_category_id
        facet_slug = self.value.attribute.slug
        facet = _get_category_facet_by_slug(category_id, facet_slug)
        if not facet:
            raise ValidationError({"value": f"Attribute '{facet_slug}' not allowed for category {self.product.primary_category.name}"})
        if facet.type == "choice" and facet.values:
            if str(self.value.value) not in [str(v) for v in facet.values]:
                raise ValidationError({"value": f"Value '{self.value.value}' is not allowed for facet '{facet.slug}'"})

    def save(self, *args, **kwargs):
        self.clean()
        return super().save(*args, **kwargs)

class ProductVariant(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, related_name="variants", on_delete=models.CASCADE)
    sku = models.CharField(max_length=80, blank=True, null=True)
    price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    stock_quantity = models.IntegerField(default=0)
    # variant-specific option values via through model
    option_values = models.ManyToManyField("OptionValue", through="VariantOptionValue", related_name="variants", blank=True)
    is_default = models.BooleanField(default=False)

    class Meta:
        indexes = [models.Index(fields=["sku"])]
        constraints = [
            models.UniqueConstraint(fields=["product"], condition=Q(is_default=True), name="unique_default_variant_per_product"),
        ]

    def __str__(self):
        return f"Variant {self.sku or self.id} of {self.product.name}"

    def clean(self):
        # Ensure only one default variant per product at save time (additional to DB constraint)
        if self.is_default:
            qs = ProductVariant.objects.filter(product=self.product, is_default=True)
            if self.pk:
                qs = qs.exclude(pk=self.pk)
            if qs.exists():
                raise ValidationError("Only one default variant allowed per product")

    @property
    def current_price(self):
        return self.price if self.price is not None else self.product.current_price


class VariantOptionValue(models.Model):
    """Through model linking Variant <-> OptionValue; enforces option uniqueness per variant."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    variant = models.ForeignKey(ProductVariant, related_name="variant_option_values", on_delete=models.CASCADE)
    option_value = models.ForeignKey('OptionValue', on_delete=models.CASCADE)

    class Meta:
        unique_together = ("variant", "option_value")


class ProductImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, related_name="images", on_delete=models.CASCADE)
    image = models.ImageField(upload_to="catalog/product_images/")
    alt_text = models.CharField(max_length=255, blank=True)

    # Visual search embedding (optional) — small vectors or references to an external store
    image_embedding = models.JSONField(null=True, blank=True, help_text="Optional image embedding vector for visual search; list of floats.")
    embedding_updated_at = models.DateTimeField(null=True, blank=True)

    is_primary = models.BooleanField(default=False)
    ordering = models.PositiveSmallIntegerField(default=0)

    class Meta:
        ordering = ["ordering"]
        indexes = [models.Index(fields=["embedding_updated_at"])]

    def __str__(self):
        return f"Image {self.id} of {self.product.name}"


class ShippingMaterial(models.Model):
    """Shipping packaging specs used when calculating shipping costs.

    Fields:
    - packaging_weight: additional package weight in kilograms
    - length/width/height: package dimensions in centimeters (cm)
    - units_per_package: typical number of items per package
    - dimensional_weight_divisor: standard divisor for dimensional weight (cm^3/divisor => kg)

    Helper methods provided to compute packaging volume, dimensional weight, and
    a recommended shipping weight for a given item weight and quantity.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    eco_score = models.IntegerField(null=True, blank=True, help_text="Higher is more eco-friendly")
    notes = models.TextField(blank=True)

    # Physical packaging dimensions (cm) and additional package weight (kg)
    packaging_weight = models.DecimalField(max_digits=8, decimal_places=3, default=0, help_text="Extra packaging weight in kilograms")
    length = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True, help_text="Length of package in centimeters")
    width = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True, help_text="Width of package in centimeters")
    height = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True, help_text="Height of package in centimeters")
    units_per_package = models.PositiveIntegerField(default=1, help_text="Number of items this packaging typically holds")
    dimensional_weight_divisor = models.IntegerField(default=5000, help_text="Divisor for dimensional weight calculation (cm^3 / divisor = kg)")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["name"])]

    def packaging_volume_cm3(self):
        """Return volume in cubic centimeters or None if dims are incomplete."""
        if self.length and self.width and self.height:
            return float(self.length) * float(self.width) * float(self.height)
        return None

    def dimensional_weight(self):
        """Return dimensional weight in kilograms based on volume/divisor, or None."""
        vol = self.packaging_volume_cm3()
        if vol:
            return float(vol) / float(self.dimensional_weight_divisor)
        return None

    def shipping_weight_for(self, item_weight_kg, quantity=1):
        """Compute shipping weight (kg) for given item weight and quantity.

        Uses the greater of gross weight (items + packaging) and dimensional weight if available.
        """
        gross = float(item_weight_kg or 0) * quantity + float(self.packaging_weight or 0)
        dweight = self.dimensional_weight()
        if dweight:
            return max(gross, round(dweight, 3))
        return round(gross, 3)

    def __str__(self):
        return self.name




class Badge(models.Model):
    """Badges are admin-driven visual/promotional markers.

    Targeting is explicit: choose one of target_product, target_category or target_tag (mutually exclusive).
    Badges are evaluated dynamically (no per-product assignment table) to keep behavior predictable and simple.
    """  
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    slug = models.SlugField(max_length=120, unique=True)
    css_class = models.CharField(max_length=100, blank=True)
    start = models.DateTimeField(null=True, blank=True)
    end = models.DateTimeField(null=True, blank=True)
    priority = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    # explicit targets (mutually exclusive)
    # target_product references Product; use string to avoid forward reference issues
    target_product = models.ForeignKey("catalog.Product", null=True, blank=True, on_delete=models.CASCADE)
    target_category = models.ForeignKey(Category, null=True, blank=True, on_delete=models.CASCADE)
    target_tag = models.ForeignKey(Tag, null=True, blank=True, on_delete=models.CASCADE)

    # Legacy: optional free-form target value (human-readable) - not used for logic
    target_raw = models.CharField(max_length=255, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["slug"]), models.Index(fields=["is_active", "start", "end"])]

    def __str__(self):
        return self.name

    def clean(self):
        # ensure only one target field is set
        targets = [bool(self.target_product), bool(self.target_category), bool(self.target_tag)]
        if sum(1 for t in targets if t) > 1:
            raise ValidationError("Only one target can be set among target_product, target_category or target_tag")

    @property
    def is_current(self):
        now = timezone.now()
        if not self.is_active:
            return False
        if self.start and self.start > now:
            return False
        if self.end and self.end < now:
            return False
        return True

    def applies_to(self, product):
        """Evaluate whether badge should display for a product (dynamic evaluation)."""
        if not self.is_current:
            return False
        if self.target_product:
            return self.target_product_id == product.id
        if self.target_category:
            return product.categories.filter(path__startswith=self.target_category.path).exists()
        if self.target_tag:
            return product.tags.filter(pk=self.target_tag_id).exists()
        # no target => global
        return True
    # NOTE: ProductBadge has been removed in favor of dynamic badge evaluation.
    # If you require materialized badge assignments for performance, implement a scheduled job
    # that populates a materialized table and a ProductBadge model with careful sync logic.


class Spotlight(models.Model):
    PLACEMENT_CHOICES = (("home", "Home"), ("category", "Category"), ("collection", "Collection"))

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=120)
    placement = models.CharField(max_length=30, choices=PLACEMENT_CHOICES, default="home")
    product = models.ForeignKey(Product, null=True, blank=True, on_delete=models.CASCADE, related_name="spotlights")
    category = models.ForeignKey(Category, null=True, blank=True, on_delete=models.CASCADE, related_name="spotlights")
    start = models.DateTimeField(null=True, blank=True)
    end = models.DateTimeField(null=True, blank=True)
    priority = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    @property
    def is_current(self):
        now = timezone.now()
        if not self.is_active:
            return False
        if self.start and self.start > now:
            return False
        if self.end and self.end < now:
            return False
        return True


# ---------------------- NEW / ADVANCED FEATURES ----------------------
# 1) Collections / Curated Lists
class Collection(models.Model):
    """Collections support manual and rule-based inclusion of products.

    - Manual collections: use `products` m2m with ordering.
    - Rule-based: store rules in JSON and evaluate using a job or on-demand.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to="catalog/collection_images/", null=True, blank=True)
    ordering = models.IntegerField(default=0)
    visible_from = models.DateTimeField(null=True, blank=True)
    visible_until = models.DateTimeField(null=True, blank=True)
    is_visible = models.BooleanField(default=True, db_index=True)

    # Manual products
    products = models.ManyToManyField(Product, through="CollectionItem", related_name="collections", blank=True)

    # Automatic rules (JSON) to be processed by a background worker (e.g. {"facet": "color", "value": "Red"})
    rules = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ["ordering", "name"]

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        try:
            return reverse("collection-detail", args=[self.slug])
        except Exception:
            return f"/collections/{self.slug}/"

    def get_products(self, limit=None):
        # Manual first, then rule-based if not enough items
        qs = Product.objects.filter(is_active=True).prefetch_related("images")
        manual = qs.filter(collections__id=self.id)
        if manual.exists() or not self.rules:
            data = manual.order_by("-is_featured")
        else:
            # Very simple rule application (supporting only attribute equals)
            rules = self.rules or {}
            if "facet" in rules and "value" in rules:
                data = qs.filter(attributes__value__iexact=rules["value"]).distinct()
            else:
                data = qs
        if limit:
            return data[:limit]
        return data


class CollectionItem(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    ordering = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = ("collection", "product")
        ordering = ["ordering"]

    def __str__(self):
        return f"{self.collection} - {self.product}"


# 2) Reviews and Ratings
class Review(models.Model):
    MODERATION_CHOICES = (("pending", "Pending"), ("approved", "Approved"), ("rejected", "Rejected"))

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="reviews")
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="reviews")
    rating = models.PositiveSmallIntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)])  # 1-5
    title = models.CharField(max_length=200, blank=True)
    body = models.TextField(blank=True)
    verified_purchase = models.BooleanField(default=False)
    helpful_votes = models.IntegerField(default=0)
    moderation_status = models.CharField(max_length=20, choices=MODERATION_CHOICES, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [models.Index(fields=["product"]), models.Index(fields=["user"])]
        ordering = ["-created_at"]

    def __str__(self):
        return f"Review {self.rating} for {self.product} by {self.user}"


class ReviewImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    review = models.ForeignKey(Review, on_delete=models.CASCADE, related_name="images")
    image = models.ImageField(upload_to="catalog/review_images/")

    def __str__(self):
        return f"Image for {self.review}"


# 3) ProductPrice for multi-currency support
class Currency(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    code = models.CharField(max_length=5, unique=True)
    symbol = models.CharField(max_length=5, blank=True)
    name = models.CharField(max_length=50, blank=True)
    rate_to_default = models.DecimalField(max_digits=18, decimal_places=8, default=1)

    def __str__(self):
        return self.code


class ProductPrice(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="prices")
    currency = models.ForeignKey(Currency, on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=12, decimal_places=2)

    class Meta:
        unique_together = ("product", "currency")
        indexes = [models.Index(fields=["currency"])]

    def __str__(self):
        return f"{self.product.name} - {self.currency.code} {self.price}"


# 4) Bundles / Kits
class Bundle(models.Model):
    STRATEGY_CHOICES = (("fixed", "Fixed Price"), ("component_sum", "Sum of Components"))

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to="catalog/bundle_images/", null=True, blank=True)
    strategy = models.CharField(max_length=20, choices=STRATEGY_CHOICES, default="fixed")
    fixed_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    products = models.ManyToManyField(Product, through="BundleItem")
    is_active = models.BooleanField(default=True)
    # Optional link to a Product record for unified cart/checkout handling
    product = models.OneToOneField(Product, null=True, blank=True, on_delete=models.SET_NULL, related_name="as_bundle")

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        try:
            return reverse("bundle-detail", args=[self.slug])
        except Exception:
            return f"/bundles/{self.slug}/"

    def price(self):
        if self.strategy == "fixed" and self.fixed_price is not None:
            return self.fixed_price
        total = 0
        for item in self.bundle_items.select_related("product").all():
            total += item.product.current_price * item.quantity
        return total


class BundleItem(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    bundle = models.ForeignKey(Bundle, related_name="bundle_items", on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)

    class Meta:
        unique_together = ("bundle", "product")


# Digital assets & downloads (for digital goods)
class DigitalAsset(models.Model):
    """Attachable digital asset for downloadable products or licenses.

    - `file` stores the downloadable asset
    - `license_key` optional per-asset license
    - `allowed_downloads` and `downloads_count` control per-asset usage
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, related_name="assets_digital", on_delete=models.CASCADE)
    file = models.FileField(upload_to="catalog/digital_assets/")
    name = models.CharField(max_length=255, blank=True)
    license_key = models.CharField(max_length=255, blank=True)
    allowed_downloads = models.IntegerField(null=True, blank=True, help_text="Max downloads allowed for this asset")
    downloads_count = models.IntegerField(default=0)
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["product"])]

    def __str__(self):
        return self.name or f"Digital asset for {self.product.name}"

    def generate_license(self):
        tpl = getattr(self.product, "digital", None)
        template = tpl.license_key_template if tpl else ""
        if template:
            key = template.replace("{uuid}", str(uuid.uuid4()))
            return key
        return self.license_key


# 5) Inventory & Reservations
class StockHistory(models.Model):
    REASON_CHOICES = (("sale", "Sale"), ("return", "Return"), ("manual", "Manual Adjustment"), ("reservation", "Reservation"))

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="stock_history")
    variant = models.ForeignKey(ProductVariant, on_delete=models.CASCADE, null=True, blank=True)
    change = models.IntegerField()
    reason = models.CharField(max_length=20, choices=REASON_CHOICES)
    note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Stock {self.change} for {self.product}"


class Reservation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    variant = models.ForeignKey(ProductVariant, on_delete=models.CASCADE, null=True, blank=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    quantity = models.PositiveIntegerField(default=1)
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["expires_at"])]


class StockAlert(models.Model):
    """Alert created when stock falls below product/variant threshold for admins/automation to act on."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="stock_alerts")
    variant = models.ForeignKey(ProductVariant, on_delete=models.CASCADE, null=True, blank=True)
    threshold = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    notified = models.BooleanField(default=False)

    class Meta:
        indexes = [models.Index(fields=["created_at"]), models.Index(fields=["notified"])]


# 6) Advanced Variant & Option System
class Option(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    slug = models.SlugField(max_length=100)

    def __str__(self):
        return self.name


class OptionValue(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    option = models.ForeignKey(Option, related_name="values", on_delete=models.CASCADE)
    value = models.CharField(max_length=100)

    class Meta:
        unique_together = ("option", "value")

    def __str__(self):
        return f"{self.option.name}: {self.value}"


class VariantImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    variant = models.ForeignKey(ProductVariant, related_name="variant_images", on_delete=models.CASCADE)
    image = models.ImageField(upload_to="catalog/variant_images/")
    ordering = models.PositiveSmallIntegerField(default=0)

    class Meta:
        ordering = ["ordering"]


# 7) Wishlists & Compare
class Wishlist(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="wishlists")
    products = models.ManyToManyField(Product, related_name="wishlists")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Wishlist of {self.user}"


class CompareList(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="compares")
    products = models.ManyToManyField(Product, related_name="compares")
    created_at = models.DateTimeField(auto_now_add=True)


# 8) SEO helpers
class SEOMixin:
    def canonical_url(self):
        # use get_absolute_url if available
        try:
            return self.get_absolute_url()
        except Exception:
            return None

    def to_json_ld(self):
        # rudimentary JSON-LD for product
        if isinstance(self, Product):
            return {
                "@context": "https://schema.org/",
                "@type": "Product",
                "name": self.name,
                "image": [img.image.url for img in self.images.all()[:3]],
                "description": self.short_description or self.description,
                "sku": self.sku,
                "offers": {"price": str(self.current_price), "priceCurrency": self.currency},
            }
        return {}


# 9) Impressions & Events
class ProductImpression(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="impressions")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    session_key = models.CharField(max_length=200, null=True, blank=True)
    occurred_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["occurred_at"])]

    def __str__(self):
        return f"Impression of {self.product} at {self.occurred_at}"

# End advanced section


class ProductMakingOf(models.Model):
    """
    Represents a step in the making-of process for a product.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, related_name='making_of_steps', on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='catalog/making_of/', blank=True, null=True)
    video_url = models.URLField(blank=True, help_text="URL to a video of this step (e.g., YouTube, Vimeo).")
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['order']
        verbose_name = 'Making Of Step'
        verbose_name_plural = 'Making Of Steps'

    def __str__(self):
        return f"Step {self.order} for {self.product.name}"


class Product3DAsset(models.Model):
    """3D assets associated with a product; restored from accidental overwrite.

    Fields:
    - file: the 3D model file (glb/usdz)
    - poster_image: optional poster/thumbnail
    - is_primary: mark one primary asset per product
    - ordering: display ordering
    - metadata: free-form JSON for processing info
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, related_name="assets_3d", on_delete=models.CASCADE)
    file = models.FileField(upload_to="catalog/3d_assets/")
    file_type = models.CharField(max_length=10, choices=(("glb", "glb"), ("usdz", "usdz")), default="glb")
    validated = models.BooleanField(default=False)
    poster_image = models.ImageField(upload_to="catalog/3d_posters/", null=True, blank=True)
    poster_alt = models.CharField(max_length=255, blank=True)
    is_primary = models.BooleanField(default=False)

    # AR/Viewer fields
    is_ar_compatible = models.BooleanField(default=False, db_index=True)
    ar_quicklook_url = models.URLField(blank=True, help_text="Optional AR Quick Look/Viewer URL")
    viewer_meta = models.JSONField(null=True, blank=True, help_text="Viewer config (scale, orientation, lighting)")

    ordering = models.PositiveIntegerField(default=0)
    metadata = models.JSONField(null=True, blank=True)

    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["ordering"]
        indexes = [models.Index(fields=["product"])]

    def __str__(self):
        return f"3D asset {self.id} for {self.product.name}"

    def save(self, *args, **kwargs):
        # TODO: schedule validation/compression tasks via background worker
        super().save(*args, **kwargs)


class CustomerPhotoManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(status='approved')


class CustomerPhoto(models.Model):
    """
    Customer-uploaded photos of products in use.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending Review'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, related_name='customer_photos', on_delete=models.CASCADE)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='customer_photos'
    )
    image = models.ImageField(upload_to='catalog/customer_photos/')
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = models.Manager() # The default manager
    approved = CustomerPhotoManager() # Our custom manager for approved photos

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Customer Photo'
        verbose_name_plural = 'Customer Photos'

    def __str__(self):
        return f"Photo for {self.product.name} by {self.user.get_full_name() if self.user else 'Anonymous'}"


class ProductQuestionManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(status='approved')


class ProductAnswerManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(status='approved')


class ProductQuestion(models.Model):
    """
    A question asked by a user about a product.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending Review'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, related_name='questions', on_delete=models.CASCADE)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='product_questions'
    )
    question_text = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = models.Manager() # The default manager
    approved = ProductQuestionManager() # Our custom manager for approved questions

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Product Question'
        verbose_name_plural = 'Product Questions'

    def __str__(self):
        return f"Question for {self.product.name} by {self.user.get_full_name() if self.user else 'Anonymous'}"


class ProductAnswer(models.Model):
    """
    An answer to a product question.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending Review'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question = models.ForeignKey(ProductQuestion, related_name='answers', on_delete=models.CASCADE)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='product_answers'
    )
    answer_text = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = models.Manager() # The default manager
    approved = ProductAnswerManager() # Our custom manager for approved answers

    class Meta:
        ordering = ['created_at']
        verbose_name = 'Product Answer'
        verbose_name_plural = 'Product Answers'

    def __str__(self):
        return f"Answer to '{self.question.question_text[:50]}...' by {self.user.get_full_name() if self.user else 'Anonymous'}"


# Facet system
class Facet(models.Model):
    TYPE_CHOICES = ("choice", "range", "boolean", "text")

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    slug = models.SlugField(max_length=100, unique=True)
    type = models.CharField(max_length=20, choices=[(t, t) for t in TYPE_CHOICES], default="choice")
    values = models.JSONField(blank=True, null=True, help_text="Optional list of values for choice facets")

    class Meta:
        indexes = [models.Index(fields=["slug"])]

    def __str__(self):
        return self.name


class CategoryFacet(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name="category_facets")
    facet = models.ForeignKey(Facet, on_delete=models.CASCADE, related_name="category_facets")

    class Meta:
        unique_together = ("category", "facet")
        indexes = [models.Index(fields=["category"]), models.Index(fields=["facet"])]

    def __str__(self):
        return f"Facet {self.facet} for {self.category}"


class EcoCertification(models.Model):
    """Certifications that can be attached to products (e.g., GOTS, Fairtrade)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    issuer = models.CharField(max_length=200, blank=True)
    metadata = models.JSONField(null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=["slug"])]

    def __str__(self):
        return self.name


