"""
Catalog API Views
"""
from rest_framework import viewsets, mixins, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, IsAuthenticatedOrReadOnly, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.generics import CreateAPIView 
from rest_framework.pagination import PageNumberPagination 
from django.shortcuts import get_object_or_404
from django.db.models import Count, Q, Avg

from apps.catalog.models import (
    Category, Product, Collection, Bundle, Review, Badge, Spotlight, Facet, Tag, CustomerPhoto,
    ProductQuestion, ProductAnswer
)
from apps.catalog.services import (
    CategoryService, ProductService, CollectionService, ReviewService,
    BadgeService, SpotlightService, BundleService, InventoryService,
    ProductFilter, ProductFilterService
)

from .serializers import (
    CategorySerializer, CategoryListSerializer, CategoryTreeSerializer,
    ProductListSerializer, ProductDetailSerializer, QuickViewProductSerializer,
    CollectionListSerializer, CollectionDetailSerializer,
    BundleListSerializer, BundleDetailSerializer,
    ReviewSerializer, CreateReviewSerializer,
    FacetSerializer, FacetWithCountsSerializer, FacetValueSerializer,
    SpotlightSerializer, TagSerializer, BadgeSerializer,
    CustomerPhotoSerializer,
    ProductQuestionSerializer, ProductAnswerSerializer
)

from apps.notifications.api.serializers import BackInStockNotificationSerializer 


# =============================================================================
# Pagination
# =============================================================================

class StandardResultsSetPagination(PageNumberPagination):
    page_size = 24
    page_size_query_param = 'page_size'
    max_page_size = 100


class SmallResultsSetPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 50


# =============================================================================
# Category ViewSets
# =============================================================================

class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for categories.
    
    list: Get root categories with nested children
    retrieve: Get single category details
    tree: Get full category tree
    children: Get direct children of a category
    products: Get products in a category
    facets: Get available facets for filtering
    """
    queryset = Category.objects.filter(is_visible=True, is_deleted=False).order_by('path')
    serializer_class = CategorySerializer
    lookup_field = 'slug'
    permission_classes = [AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return CategoryListSerializer
        return CategorySerializer
    
    def list(self, request, *args, **kwargs):
        """Return categories, optionally filtered by parent_id."""
        # Support filtering by parent_id or parent (for consistency)
        parent_id = request.query_params.get('parent_id') or request.query_params.get('parent')
        
        try:
            # If parent_id is 'null' or not provided, get root categories
            if parent_id == 'null' or parent_id is None:
                categories = CategoryService.get_root_categories()
            else:
                # Get children of specified parent
                try:
                    parent = Category.objects.get(id=parent_id, is_visible=True, is_deleted=False)
                    categories = parent.children.filter(is_visible=True, is_deleted=False).order_by('path')
                except Category.DoesNotExist:
                    return Response([])
            
            # Respect pagination if specified
            page_size = request.query_params.get('page_size')
            if page_size:
                try:
                    page_size = int(page_size)
                    categories = categories[:page_size]
                except (ValueError, TypeError):
                    pass
            
            serializer = CategoryListSerializer(categories, many=True, context={'request': request})
            return Response(serializer.data)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.exception(f"Error in category list: {e}")
            return Response({'error': str(e)}, status=500)
    
    @action(detail=False, methods=['get'])
    def tree(self, request):
        """Return full category tree."""
        max_depth = request.query_params.get('max_depth')
        if max_depth:
            max_depth = int(max_depth)
        tree = CategoryService.get_category_tree(max_depth=max_depth)
        return Response(tree)
    
    @action(detail=True, methods=['get'])
    def children(self, request, slug=None):
        """Get direct children of a category."""
        category = self.get_object()
        children = category.children.filter(is_visible=True, is_deleted=False)
        serializer = CategoryListSerializer(children, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def products(self, request, slug=None):
        """Get products in a category."""
        category = self.get_object()
        include_descendants = request.query_params.get('include_descendants', 'true').lower() == 'true'
        
        products = CategoryService.get_category_products(category, include_descendants=include_descendants)
        
        # Apply additional filters
        products = ProductFilterService.apply_attribute_filters(products, request.query_params)
        
        page = self.paginate_queryset(products)
        if page is not None:
            serializer = ProductListSerializer(page, many=True, context={'request': request})
            return self.get_paginated_response(serializer.data)
        
        serializer = ProductListSerializer(products, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def facets(self, request, slug=None):
        """Get facets available for a category with value counts."""
        category = self.get_object()
        facets = CategoryService.get_category_facets(category)
        products = CategoryService.get_category_products(category)
        
        data = []
        for facet in facets:
            item = FacetSerializer(facet, context={'request': request}).data
            if facet.type == 'choice' and facet.values:
                value_counts = []
                for v in facet.values:
                    count = products.filter(attributes__value__iexact=v).distinct().count()
                    if count > 0:
                        value_counts.append({'value': v, 'count': count})
                item['value_counts'] = value_counts
            data.append(item)
        
        return Response(data)
    
    @action(detail=False, methods=['get'])
    def search(self, request):
        """Search categories by name."""
        query = request.query_params.get('q', '')
        if not query:
            return Response([])
        
        categories = CategoryService.search_categories(query, limit=10)
        serializer = CategoryListSerializer(categories, many=True, context={'request': request})
        return Response(serializer.data)


# =============================================================================
# Product ViewSets
# =============================================================================

class ProductViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for products.
    
    list: Get paginated product list with filtering
    retrieve: Get single product details
    quick_view: Get minimal product data for modal
    related: Get related products
    reviews: Get product reviews
    """
    queryset = Product.objects.filter(is_active=True, is_deleted=False)
    serializer_class = ProductListSerializer
    pagination_class = StandardResultsSetPagination
    filterset_class = ProductFilter
    lookup_field = 'slug'
    permission_classes = [AllowAny]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['name', 'description', 'short_description', 'sku']
    ordering_fields = ['price', 'name', 'created_at', 'sales_count', 'views_count', 'average_rating']
    ordering = ['-created_at']
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return ProductDetailSerializer
        elif self.action == 'quick_view':
            return QuickViewProductSerializer
        return ProductListSerializer
    
    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.select_related('primary_category').prefetch_related(
            'images', 'variants', 'categories', 'tags'
        )
        qs = ProductFilterService.apply_attribute_filters(qs, self.request.query_params)
        return qs
    
    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        # Record view
        ProductService.record_view(
            instance,
            user=request.user if request.user.is_authenticated else None,
            session_key=request.session.session_key if hasattr(request, 'session') else None
        )
        serializer = self.get_serializer(instance)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'], url_path='quick-view')
    def quick_view(self, request, slug=None):
        """Get minimal product data for quick view modal."""
        product = self.get_object()
        serializer = QuickViewProductSerializer(product, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def related(self, request, slug=None):
        """Get related products."""
        product = self.get_object()
        limit = int(request.query_params.get('limit', 4))
        related = ProductService.get_related_products(product, limit=limit)
        serializer = ProductListSerializer(related, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def reviews(self, request, slug=None):
        """Get product reviews."""
        product = self.get_object()
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 10))
        
        reviews_data = ReviewService.get_product_reviews(product, page=page, page_size=page_size)
        serializer = ReviewSerializer(reviews_data['reviews'], many=True, context={'request': request})
        
        return Response({
            'reviews': serializer.data,
            'summary': ReviewService.get_review_summary(product),
            'total': reviews_data['total'],
            'page': reviews_data['page'],
            'total_pages': reviews_data['total_pages'],
        })
    
    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def add_review(self, request, slug=None):
        """Add a review to a product."""
        product = self.get_object()
        serializer = CreateReviewSerializer(data=request.data)
        
        if serializer.is_valid():
            try:
                review = ReviewService.create_review(
                    product=product,
                    user=request.user,
                    **serializer.validated_data
                )
                return Response(
                    ReviewSerializer(review, context={'request': request}).data,
                    status=status.HTTP_201_CREATED
                )
            except ValueError as e:
                return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'], permission_classes=[AllowAny], url_path='request-back-in-stock')
    def request_back_in_stock(self, request, slug=None):
        """
        Allows a user to request a notification when a product or variant is back in stock.
        """
        product = self.get_object()
        variant_id = request.data.get('variant_id')
        email = request.data.get('email')

        if not request.user.is_authenticated and not email:
            return Response({'detail': 'Email is required for unauthenticated users.'}, status=status.HTTP_400_BAD_REQUEST)

        if variant_id:
            variant = get_object_or_404(ProductVariant, id=variant_id, product=product)
        else:
            variant = None
        
        # Check if already requested
        if request.user.is_authenticated:
            existing_request = BackInStockNotification.objects.filter(product=product, variant=variant, user=request.user).exists()
        else:
            existing_request = BackInStockNotification.objects.filter(product=product, variant=variant, email=email).exists()

        if existing_request:
            return Response({'detail': 'You have already requested a notification for this product/variant.'}, status=status.HTTP_409_CONFLICT) # 409 Conflict

        serializer = BackInStockNotificationSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(
                product=product,
                variant=variant,
                user=request.user if request.user.is_authenticated else None,
                email=email if not request.user.is_authenticated else ''
            )
            return Response({'detail': 'We will notify you when this product is back in stock!'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['get'])
    def featured(self, request):
        """Get featured products."""
        limit = int(request.query_params.get('limit', 8))
        products = ProductService.get_featured_products(limit=limit)
        serializer = ProductListSerializer(products, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'], url_path='new-arrivals')
    def new_arrivals(self, request):
        """Get new arrival products."""
        limit = int(request.query_params.get('limit', 8))
        products = ProductService.get_new_arrivals(limit=limit)
        serializer = ProductListSerializer(products, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def bestsellers(self, request):
        """Get bestselling products."""
        limit = int(request.query_params.get('limit', 8))
        products = ProductService.get_bestsellers(limit=limit)
        serializer = ProductListSerializer(products, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'], url_path='on-sale')
    def on_sale(self, request):
        """Get products on sale."""
        limit = int(request.query_params.get('limit', 8))
        products = ProductService.get_on_sale_products(limit=limit)
        serializer = ProductListSerializer(products, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'], url_path='by-category')
    def by_category(self, request):
        """Get products by category slug or ID."""
        slug_or_id = request.query_params.get('category')
        if not slug_or_id:
            return Response({'detail': 'category parameter is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            category = Category.objects.get(Q(id=slug_or_id) | Q(slug=slug_or_id))
        except Category.DoesNotExist:
            return Response({'detail': 'Category not found'}, status=status.HTTP_404_NOT_FOUND)
        
        products = CategoryService.get_category_products(category)
        page = self.paginate_queryset(products)
        
        if page is not None:
            serializer = ProductListSerializer(page, many=True, context={'request': request})
            return self.get_paginated_response(serializer.data)
        
        serializer = ProductListSerializer(products, many=True, context={'request': request})
        return Response(serializer.data)


# =============================================================================
# Collection ViewSets
# =============================================================================

class CollectionViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for collections."""
    queryset = Collection.objects.filter(is_visible=True)
    serializer_class = CollectionListSerializer
    lookup_field = 'slug'
    permission_classes = [AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return CollectionDetailSerializer
        return CollectionListSerializer
    
    def get_queryset(self):
        return CollectionService.get_active_collections()
    
    @action(detail=True, methods=['get'])
    def products(self, request, slug=None):
        """Get products in a collection."""
        collection = self.get_object()
        limit = request.query_params.get('limit')
        if limit:
            limit = int(limit)
        
        products = CollectionService.get_collection_products(collection, limit=limit)
        serializer = ProductListSerializer(products, many=True, context={'request': request})
        return Response(serializer.data)


# =============================================================================
# Bundle ViewSets
# =============================================================================

class BundleViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for bundles."""
    queryset = Bundle.objects.filter(is_active=True)
    serializer_class = BundleListSerializer
    lookup_field = 'slug'
    permission_classes = [AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return BundleDetailSerializer
        return BundleListSerializer
    
    def get_queryset(self):
        return BundleService.get_active_bundles()


# =============================================================================
# Tag ViewSets
# =============================================================================

class TagViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for tags."""
    queryset = Tag.objects.all()
    serializer_class = TagSerializer
    permission_classes = [AllowAny]
    
    @action(detail=False, methods=['get'])
    def popular(self, request):
        """Get popular tags with product counts."""
        limit = int(request.query_params.get('limit', 20))
        tags = Tag.objects.annotate(
            product_count=Count('products', filter=Q(products__is_active=True, products__is_deleted=False))
        ).filter(product_count__gt=0).order_by('-product_count')[:limit]
        
        data = [{'id': str(t.id), 'name': t.name, 'count': t.product_count} for t in tags]
        return Response(data)


# =============================================================================
# Spotlight & Badge ViewSets
# =============================================================================

class SpotlightViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for spotlights."""
    queryset = Spotlight.objects.filter(is_active=True)
    serializer_class = SpotlightSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        placement = self.request.query_params.get('placement', 'home')
        return SpotlightService.get_active_spotlights(placement=placement, limit=10)


class BadgeViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for badges."""
    queryset = Badge.objects.filter(is_active=True)
    serializer_class = BadgeSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        return BadgeService.get_active_badges()


# =============================================================================
# Search View
# =============================================================================

class SearchAPIView(APIView):
    """Search API endpoint."""
    permission_classes = [AllowAny]
    
    def get(self, request):
        query = request.query_params.get('q', '').strip()
        
        if not query:
            return Response({'products': [], 'categories': []})
        
        # Search products
        products = ProductService.search_products(query, limit=20)
        products_data = ProductListSerializer(products, many=True, context={'request': request}).data
        
        # Search categories
        categories = CategoryService.search_categories(query, limit=5)
        categories_data = CategoryListSerializer(categories, many=True, context={'request': request}).data
        
        return Response({
            'products': products_data,
            'categories': categories_data,
            'query': query
        })


# =============================================================================
# Homepage Data View
# =============================================================================

class HomepageDataView(APIView):
    """Get all data needed for homepage in one request."""
    permission_classes = [AllowAny]
    
    def get(self, request):
        return Response({
            'featured_products': ProductListSerializer(
                ProductService.get_featured_products(limit=8),
                many=True, context={'request': request}
            ).data,
            'new_arrivals': ProductListSerializer(
                ProductService.get_new_arrivals(limit=8),
                many=True, context={'request': request}
            ).data,
            'bestsellers': ProductListSerializer(
                ProductService.get_bestsellers(limit=8),
                many=True, context={'request': request}
            ).data,
            'on_sale': ProductListSerializer(
                ProductService.get_on_sale_products(limit=8),
                many=True, context={'request': request}
            ).data,
            'featured_categories': CategoryListSerializer(
                CategoryService.get_featured_categories(limit=6),
                many=True, context={'request': request}
            ).data,
            'spotlights': SpotlightSerializer(
                SpotlightService.get_active_spotlights(placement='home', limit=5),
                many=True, context={'request': request}
            ).data,
            'collections': CollectionListSerializer(
                CollectionService.get_active_collections(limit=4),
                many=True, context={'request': request}
            ).data,
        })


class CustomerPhotoUploadView(CreateAPIView):
    """
    API endpoint for customers to upload photos of products.
    """
    queryset = CustomerPhoto.objects.all()
    serializer_class = CustomerPhotoSerializer
    permission_classes = [IsAuthenticatedOrReadOnly] # Allow anonymous to upload, status will be pending
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        # The serializer's create method handles setting the user if authenticated
        serializer.save()

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response({
            'success': True,
            'message': 'Your photo has been uploaded and is pending review!',
            'data': serializer.data
        }, status=status.HTTP_201_CREATED, headers=headers)


class ProductQuestionListView(mixins.ListModelMixin, mixins.CreateModelMixin, viewsets.GenericViewSet):
    """
    API endpoint for listing and creating product questions.
    """
    serializer_class = ProductQuestionSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]

    def get_queryset(self):
        product_id = self.kwargs['product_pk']
        return ProductQuestion.approved.filter(product_id=product_id)

    def perform_create(self, serializer):
        product = get_object_or_404(Product, pk=self.kwargs['product_pk'])
        serializer.save(product=product, user=self.request.user if self.request.user.is_authenticated else None)

    @action(detail=False, methods=['post'], permission_classes=[IsAuthenticated])
    def ask_question(self, request, product_pk=None):
        product = get_object_or_404(Product, pk=product_pk)
        serializer = self.get_serializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        return Response({
            'success': True,
            'message': 'Your question has been submitted and is pending review!',
            'data': serializer.data
        }, status=status.HTTP_201_CREATED)


class ProductAnswerCreateView(mixins.CreateModelMixin, viewsets.GenericViewSet):
    """
    API endpoint for creating product answers.
    Only staff or the original question asker can answer a question?
    For now, any authenticated user can answer. Answers are pending moderation.
    """
    serializer_class = ProductAnswerSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        question = get_object_or_404(ProductQuestion, pk=self.kwargs['question_pk'])
        serializer.save(question=question, user=self.request.user)

    @action(detail=False, methods=['post'], permission_classes=[IsAuthenticated])
    def add_answer(self, request, question_pk=None):
        question = get_object_or_404(ProductQuestion, pk=question_pk)
        serializer = self.get_serializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        return Response({
            'success': True,
            'message': 'Your answer has been submitted and is pending review!',
            'data': serializer.data
        }, status=status.HTTP_201_CREATED)
