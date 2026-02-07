"""
Catalog API Serializers
"""
from rest_framework import serializers
from decimal import Decimal

from apps.catalog.models import (
    Category, Product, ProductImage, ProductVariant, Tag, Attribute, AttributeValue,
    Collection, CollectionItem, Review, ReviewImage, Badge, Spotlight, Bundle, BundleItem,
    Facet, CategoryFacet, ShippingMaterial, Product3DAsset, Option, OptionValue,
    Currency, ProductPrice, EcoCertification, CustomerPhoto,
    ProductQuestion, ProductAnswer
)
from apps.i18n.services import CurrencyService, CurrencyConversionService
from apps.i18n.api.serializers import PriceConversionMixin


# =============================================================================
# Base Serializers
# =============================================================================

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ('id', 'name')


class AttributeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Attribute
        fields = ('id', 'name', 'slug')


class AttributeValueSerializer(serializers.ModelSerializer):
    attribute = AttributeSerializer(read_only=True)
    
    class Meta:
        model = AttributeValue
        fields = ('id', 'attribute', 'value')


class OptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Option
        fields = ('id', 'name', 'slug')


class OptionValueSerializer(serializers.ModelSerializer):
    option = OptionSerializer(read_only=True)
    
    class Meta:
        model = OptionValue
        fields = ('id', 'option', 'value')


# =============================================================================
# Category Serializers
# =============================================================================

class CategoryListSerializer(serializers.ModelSerializer):
    """Lightweight category serializer for lists."""
    
    class Meta:
        model = Category
        fields = ('id', 'name', 'slug', 'depth', 'path', 'image', 'icon', 'product_count')


class CategorySerializer(serializers.ModelSerializer):
    """Full category serializer with children."""
    children = serializers.SerializerMethodField()
    breadcrumbs = serializers.SerializerMethodField()
    
    class Meta:
        model = Category
        fields = (
            'id', 'name', 'slug', 'parent', 'path', 'depth',
            'is_visible', 'meta_title', 'meta_description', 'meta_keywords',
            'image', 'icon', 'aspect_ratio', 'product_count',
            'children', 'breadcrumbs', 'created_at', 'updated_at'
        )
    
    def get_children(self, obj):
        children = obj.children.filter(is_deleted=False, is_visible=True)
        return CategoryListSerializer(children, many=True, context=self.context).data
    
    def get_breadcrumbs(self, obj):
        ancestors = obj.get_ancestors(include_self=True)
        return [{'id': str(c.id), 'name': c.name, 'slug': c.slug} for c in ancestors]


class CategoryTreeSerializer(serializers.Serializer):
    """Serializer for category tree structure."""
    id = serializers.CharField()
    name = serializers.CharField()
    slug = serializers.CharField()
    depth = serializers.IntegerField()
    path = serializers.CharField()
    image = serializers.CharField(allow_null=True)
    icon = serializers.CharField(allow_blank=True)
    product_count = serializers.IntegerField()
    children = serializers.ListField(child=serializers.DictField())


# =============================================================================
# Product Serializers
# =============================================================================

class ProductImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductImage
        fields = ('id', 'image', 'alt_text', 'is_primary', 'ordering')


class Product3DAssetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product3DAsset
        fields = (
            'id', 'file', 'file_type', 'validated', 'poster_image', 'poster_alt',
            'is_primary', 'is_ar_compatible', 'ar_quicklook_url', 'ordering'
        )


class ProductVariantSerializer(serializers.ModelSerializer):
    option_values = OptionValueSerializer(many=True, read_only=True)
    current_price = serializers.DecimalField(max_digits=12, decimal_places=2, read_only=True)
    
    class Meta:
        model = ProductVariant
        fields = (
            'id', 'sku', 'price', 'stock_quantity', 'is_default',
            'option_values', 'current_price'
        )


class ShippingMaterialSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShippingMaterial
        fields = ('id', 'name', 'eco_score', 'notes', 'packaging_weight')


class EcoCertificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = EcoCertification
        fields = ('id', 'name', 'slug', 'issuer')


class BadgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Badge
        fields = ('id', 'name', 'slug', 'css_class', 'start', 'end', 'priority')


class ProductListSerializer(PriceConversionMixin, serializers.ModelSerializer):
    """Lightweight product serializer for lists."""
    primary_image = serializers.SerializerMethodField()
    current_price = serializers.DecimalField(max_digits=12, decimal_places=2, read_only=True)
    discount_percentage = serializers.DecimalField(max_digits=5, decimal_places=2, read_only=True)
    is_on_sale = serializers.BooleanField(read_only=True)
    is_in_stock = serializers.SerializerMethodField()
    primary_category_name = serializers.CharField(source='primary_category.name', read_only=True)
    
    class Meta:
        model = Product
        fields = (
            'id', 'name', 'slug', 'sku', 'short_description',
            'price', 'sale_price', 'current_price', 'currency',
            'discount_percentage', 'is_on_sale', 'is_in_stock',
            'is_featured', 'is_bestseller', 'is_new_arrival',
            'average_rating', 'reviews_count', 'views_count',
            'primary_image', 'primary_category_name'
        )
    
    def get_primary_image(self, obj):
        primary = obj.images.filter(is_primary=True).first() or obj.images.first()
        if primary:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(primary.image.url)
            return primary.image.url
        return None
    
    def get_is_in_stock(self, obj):
        return obj.is_in_stock()
    
    def to_representation(self, instance):
        """Convert prices to user's selected currency."""
        data = super().to_representation(instance)
        return self.convert_price_fields(data)


class ProductDetailSerializer(PriceConversionMixin, serializers.ModelSerializer):
    """Full product serializer with all details."""
    images = ProductImageSerializer(many=True, read_only=True)
    variants = ProductVariantSerializer(many=True, read_only=True)
    categories = CategoryListSerializer(many=True, read_only=True)
    primary_category = CategoryListSerializer(read_only=True)
    tags = TagSerializer(many=True, read_only=True)
    attributes = AttributeValueSerializer(many=True, read_only=True)
    shipping_material = ShippingMaterialSerializer(read_only=True)
    eco_certifications = EcoCertificationSerializer(many=True, read_only=True)
    assets_3d = Product3DAssetSerializer(many=True, read_only=True)
    
    current_price = serializers.DecimalField(max_digits=12, decimal_places=2, read_only=True)
    discount_percentage = serializers.DecimalField(max_digits=5, decimal_places=2, read_only=True)
    is_on_sale = serializers.BooleanField(read_only=True)
    is_in_stock = serializers.SerializerMethodField()
    is_low_stock = serializers.BooleanField(read_only=True)
    available_stock = serializers.SerializerMethodField()
    
    breadcrumbs = serializers.SerializerMethodField()
    schema_org = serializers.SerializerMethodField()
    
    class Meta:
        model = Product
        fields = (
            'id', 'name', 'slug', 'sku', 'short_description', 'description',
            'price', 'sale_price', 'current_price', 'cost', 'currency',
            'discount_percentage', 'is_on_sale', 'is_in_stock', 'is_low_stock',
            'stock_quantity', 'available_stock', 'allow_backorder',
            'weight', 'length', 'width', 'height', 'aspect_ratio',
            'categories', 'primary_category', 'tags', 'attributes',
            'images', 'variants', 'assets_3d',
            'shipping_material', 'eco_certifications',
            'carbon_footprint_kg', 'recycled_content_percentage', 'sustainability_score',
            'is_ar_compatible', 'is_mobile_optimized',
            'meta_title', 'meta_description', 'meta_keywords',
            'is_featured', 'is_bestseller', 'is_new_arrival',
            'average_rating', 'reviews_count', 'rating_count', 'views_count', 'sales_count',
            'breadcrumbs', 'schema_org',
            'created_at', 'updated_at'
        )
    
    def get_is_in_stock(self, obj):
        return obj.is_in_stock()
    
    def get_available_stock(self, obj):
        return obj.available_stock()
    
    def get_breadcrumbs(self, obj):
        if obj.primary_category:
            ancestors = obj.primary_category.get_ancestors(include_self=True)
            return [{'id': str(c.id), 'name': c.name, 'slug': c.slug} for c in ancestors]
        return []
    
    def get_schema_org(self, obj):
        return obj.to_schema()
    
    def to_representation(self, instance):
        """Convert prices to user's selected currency."""
        data = super().to_representation(instance)
        return self.convert_price_fields(data)


class QuickViewProductSerializer(PriceConversionMixin, serializers.ModelSerializer):
    """Lightweight serializer for quick view modal."""
    primary_image = serializers.SerializerMethodField()
    current_price = serializers.DecimalField(max_digits=12, decimal_places=2, read_only=True)
    is_in_stock = serializers.SerializerMethodField()
    badges = serializers.SerializerMethodField()
    
    class Meta:
        model = Product
        fields = (
            'id', 'name', 'slug', 'sku', 'short_description',
            'price', 'sale_price', 'current_price', 'currency',
            'is_in_stock', 'primary_image', 'badges',
            'average_rating', 'reviews_count'
        )
    
    def get_primary_image(self, obj):
        primary = obj.images.filter(is_primary=True).first() or obj.images.first()
        if primary:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(primary.image.url)
            return primary.image.url
        return None
    
    def get_is_in_stock(self, obj):
        return obj.is_in_stock()
    
    def get_badges(self, obj):
        from apps.catalog.services import BadgeService
        badges = BadgeService.get_product_badges(obj)
        return BadgeSerializer(badges, many=True, context=self.context).data
    
    def to_representation(self, instance):
        """Convert prices to user's selected currency."""
        data = super().to_representation(instance)
        return self.convert_price_fields(data)


# =============================================================================
# Review Serializers
# =============================================================================

class ReviewImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReviewImage
        fields = ('id', 'image')


class ReviewSerializer(serializers.ModelSerializer):
    user_name = serializers.SerializerMethodField()
    images = ReviewImageSerializer(many=True, read_only=True)
    
    class Meta:
        model = Review
        fields = (
            'id', 'user_name', 'rating', 'title', 'body',
            'verified_purchase', 'helpful_votes', 'moderation_status',
            'images', 'created_at', 'updated_at'
        )
    
    def get_user_name(self, obj):
        if obj.user:
            return f"{obj.user.first_name} {obj.user.last_name[0]}." if obj.user.last_name else obj.user.first_name
        return "Anonymous"


class CreateReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = Review
        fields = ('rating', 'title', 'body')
    
    def validate_rating(self, value):
        if not 1 <= value <= 5:
            raise serializers.ValidationError("Rating must be between 1 and 5")
        return value


# =============================================================================
# Collection Serializers
# =============================================================================

class CollectionListSerializer(serializers.ModelSerializer):
    product_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Collection
        fields = (
            'id', 'name', 'slug', 'description', 'image',
            'ordering', 'is_visible', 'product_count'
        )
    
    def get_product_count(self, obj):
        return obj.products.count()


class CollectionDetailSerializer(serializers.ModelSerializer):
    products = ProductListSerializer(many=True, read_only=True)
    
    class Meta:
        model = Collection
        fields = (
            'id', 'name', 'slug', 'description', 'image',
            'ordering', 'visible_from', 'visible_until', 'is_visible',
            'products', 'rules'
        )


# =============================================================================
# Bundle Serializers
# =============================================================================

class BundleItemSerializer(serializers.ModelSerializer):
    product = ProductListSerializer(read_only=True)
    
    class Meta:
        model = BundleItem
        fields = ('id', 'product', 'quantity')


class BundleListSerializer(serializers.ModelSerializer):
    price = serializers.SerializerMethodField()
    item_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Bundle
        fields = (
            'id', 'name', 'slug', 'description', 'image',
            'strategy', 'price', 'is_active', 'item_count'
        )
    
    def get_price(self, obj):
        return str(obj.price())
    
    def get_item_count(self, obj):
        return obj.bundle_items.count()


class BundleDetailSerializer(serializers.ModelSerializer):
    items = BundleItemSerializer(source='bundle_items', many=True, read_only=True)
    price = serializers.SerializerMethodField()
    savings = serializers.SerializerMethodField()
    
    class Meta:
        model = Bundle
        fields = (
            'id', 'name', 'slug', 'description', 'image',
            'strategy', 'fixed_price', 'price', 'savings',
            'is_active', 'items'
        )
    
    def get_price(self, obj):
        return str(obj.price())
    
    def get_savings(self, obj):
        from apps.catalog.services import BundleService
        return str(BundleService.calculate_bundle_savings(obj))


# =============================================================================
# Facet Serializers
# =============================================================================

class FacetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Facet
        fields = ('id', 'name', 'slug', 'type', 'values')


class FacetValueSerializer(serializers.Serializer):
    """Serializer for facet value with count."""
    value = serializers.CharField()
    count = serializers.IntegerField()


class FacetWithCountsSerializer(serializers.ModelSerializer):
    """Facet with value counts for filtering UI."""
    value_counts = serializers.ListField(child=FacetValueSerializer())
    
    class Meta:
        model = Facet
        fields = ('id', 'name', 'slug', 'type', 'values', 'value_counts')


# =============================================================================
# Spotlight Serializers
# =============================================================================

class SpotlightSerializer(serializers.ModelSerializer):
    product = ProductListSerializer(read_only=True)
    category = CategoryListSerializer(read_only=True)
    
    class Meta:
        model = Spotlight
        fields = (
            'id', 'name', 'placement', 'product', 'category',
            'start', 'end', 'priority', 'is_active'
        )


# =============================================================================
# Currency Serializers
# =============================================================================

class CurrencySerializer(serializers.ModelSerializer):
    class Meta:
        model = Currency
        fields = ('id', 'code', 'symbol', 'name', 'rate_to_default')


class ProductPriceSerializer(serializers.ModelSerializer):
    currency = CurrencySerializer(read_only=True)
    
    class Meta:
        model = ProductPrice
        fields = ('id', 'currency', 'price')


class CustomerPhotoSerializer(serializers.ModelSerializer):
    user_name = serializers.SerializerMethodField()
    product_name = serializers.CharField(source='product.name', read_only=True)

    class Meta:
        model = CustomerPhoto
        fields = ['id', 'product', 'product_name', 'user', 'user_name', 'image', 'description', 'status', 'created_at']
        read_only_fields = ['user', 'status', 'created_at', 'updated_at', 'product_name']

    def get_user_name(self, obj):
        if obj.user:
            return obj.user.get_full_name()
        return "Anonymous"

    def create(self, validated_data):
        # Set user if authenticated
        user = self.context['request'].user
        if user.is_authenticated:
            validated_data['user'] = user
        return super().create(validated_data)


class ProductQuestionSerializer(serializers.ModelSerializer):
    user_name = serializers.SerializerMethodField()
    answers = serializers.SerializerMethodField()

    class Meta:
        model = ProductQuestion
        fields = ['id', 'product', 'user', 'user_name', 'question_text', 'status', 'created_at', 'answers']
        read_only_fields = ['user', 'status', 'created_at', 'updated_at']

    def get_user_name(self, obj):
        if obj.user:
            return obj.user.get_full_name()
        return "Anonymous"

    def get_answers(self, obj):
        answers = obj.answers.approved().order_by('created_at')
        return ProductAnswerSerializer(answers, many=True, context=self.context).data

    def create(self, validated_data):
        user = self.context['request'].user
        if user.is_authenticated:
            validated_data['user'] = user
        return super().create(validated_data)


class ProductAnswerSerializer(serializers.ModelSerializer):
    user_name = serializers.SerializerMethodField()

    class Meta:
        model = ProductAnswer
        fields = ['id', 'question', 'user', 'user_name', 'answer_text', 'status', 'created_at']
        read_only_fields = ['user', 'status', 'created_at', 'updated_at']

    def get_user_name(self, obj):
        if obj.user:
            return obj.user.get_full_name()
        return "Anonymous"

    def create(self, validated_data):
        user = self.context['request'].user
        if user.is_authenticated:
            validated_data['user'] = user
        return super().create(validated_data)
