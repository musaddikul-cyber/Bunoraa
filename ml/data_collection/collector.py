"""
Comprehensive Data Collector

Collects all possible user and product data for ML training.
Operates silently in the background without affecting user experience.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from decimal import Decimal

try:
    from django.conf import settings
    from django.core.cache import cache
    from django.db import models, transaction
    from django.utils import timezone
    from django.contrib.gis.geoip2 import GeoIP2
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("bunoraa.ml.data_collection")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UserInteraction:
    """Represents a single user interaction event."""
    
    # Session & User Info
    session_id: str
    user_id: Optional[int] = None
    anonymous_id: Optional[str] = None
    
    # Event Details
    event_type: str = ""  # page_view, product_view, click, scroll, add_to_cart, etc.
    event_timestamp: datetime = field(default_factory=datetime.now)
    
    # Page Context
    page_url: str = ""
    page_type: str = ""  # home, category, product, cart, checkout, etc.
    referrer_url: str = ""
    referrer_domain: str = ""
    utm_source: str = ""
    utm_medium: str = ""
    utm_campaign: str = ""
    
    # Device & Browser
    device_type: str = ""  # mobile, tablet, desktop
    browser: str = ""
    browser_version: str = ""
    os: str = ""
    os_version: str = ""
    screen_width: int = 0
    screen_height: int = 0
    viewport_width: int = 0
    viewport_height: int = 0
    device_pixel_ratio: float = 1.0
    
    # Location (from IP)
    ip_address: str = ""
    country: str = ""
    country_code: str = ""
    city: str = ""
    region: str = ""
    postal_code: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    timezone_name: str = ""
    
    # Time Metrics
    time_on_page_seconds: float = 0.0
    time_on_element_seconds: float = 0.0
    scroll_depth_percent: float = 0.0
    active_time_seconds: float = 0.0  # Time user was actively engaged
    idle_time_seconds: float = 0.0
    
    # Theme & Preferences
    theme_used: str = "light"
    language: str = "bn"
    currency: str = "BDT"
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductInteraction:
    """Product-specific interaction data."""
    
    # Product Identification
    product_id: int = 0
    product_slug: str = ""
    product_sku: str = ""
    
    # Product Attributes
    product_title: str = ""
    title_char_count: int = 0
    title_word_count: int = 0
    
    product_description: str = ""
    description_char_count: int = 0
    description_word_count: int = 0
    description_sentence_count: int = 0
    
    # Product Categorization
    category_id: int = 0
    category_name: str = ""
    category_hierarchy: List[int] = field(default_factory=list)
    
    # Product Properties
    price: Decimal = Decimal("0.00")
    original_price: Decimal = Decimal("0.00")
    discount_amount: Decimal = Decimal("0.00")
    discount_percent: float = 0.0
    has_discount: bool = False
    
    color: str = ""
    material: str = ""
    size: str = ""
    brand: str = ""
    
    # Availability & Status
    in_stock: bool = True
    stock_quantity: int = 0
    is_new_arrival: bool = False
    is_best_seller: bool = False
    is_spotlight: bool = False
    is_featured: bool = False
    is_preorder: bool = False
    
    # Product Features
    has_variants: bool = False
    variant_count: int = 0
    has_images: bool = True
    image_count: int = 0
    has_video: bool = False
    has_reviews: bool = False
    review_count: int = 0
    average_rating: float = 0.0
    
    # Interaction Details
    clicked_image: bool = False
    clicked_variant: bool = False
    selected_variant_id: Optional[int] = None
    viewed_reviews: bool = False
    time_viewing_seconds: float = 0.0
    scroll_to_description: bool = False
    scroll_to_reviews: bool = False
    
    # Source Context
    source_page: str = ""  # Where user came from
    source_type: str = ""  # recommendation, search, category, direct
    search_query: str = ""  # If from search
    recommendation_source: str = ""  # If from recommendation
    position_in_list: int = 0  # Position if from list view


@dataclass
class ConversionEvent:
    """Conversion and cart-related events."""
    
    session_id: str = ""
    user_id: Optional[int] = None
    
    # Cart Events
    added_to_cart: bool = False
    removed_from_cart: bool = False
    cart_quantity: int = 0
    cart_item_count: int = 0
    cart_total: Decimal = Decimal("0.00")
    
    # Wishlist Events
    added_to_wishlist: bool = False
    removed_from_wishlist: bool = False
    wishlist_item_count: int = 0
    
    # Checkout Events
    started_checkout: bool = False
    completed_checkout: bool = False
    checkout_step: int = 0
    checkout_abandoned: bool = False
    checkout_abandon_step: int = 0
    
    # Order Details (if converted)
    order_id: Optional[int] = None
    order_total: Decimal = Decimal("0.00")
    order_item_count: int = 0
    payment_method: str = ""
    shipping_method: str = ""
    shipping_location: str = ""
    
    # Promotions
    used_coupon: bool = False
    coupon_code: str = ""
    coupon_discount: Decimal = Decimal("0.00")
    is_gift: bool = False
    has_subscription: bool = False
    
    # VAT & Shipping
    vat_percent: float = 0.0
    vat_amount: Decimal = Decimal("0.00")
    shipping_cost: Decimal = Decimal("0.00")
    
    # Sharing & Social
    shared_product: bool = False
    share_platform: str = ""
    
    event_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UserProfile:
    """Comprehensive user profile data."""
    
    user_id: int = 0
    
    # Demographics
    username: str = ""
    email: str = ""
    phone: str = ""
    gender: str = ""
    age: Optional[int] = None
    date_of_birth: Optional[datetime] = None
    
    # Account Info
    account_created: Optional[datetime] = None
    account_age_days: int = 0
    is_verified: bool = False
    email_verified: bool = False
    phone_verified: bool = False
    
    # Trust & Engagement
    trust_score: float = 0.0
    loyalty_tier: str = ""
    total_orders: int = 0
    total_spent: Decimal = Decimal("0.00")
    average_order_value: Decimal = Decimal("0.00")
    
    # Activity Metrics
    total_sessions: int = 0
    total_page_views: int = 0
    total_product_views: int = 0
    last_activity: Optional[datetime] = None
    days_since_last_order: int = 0
    
    # Preferences
    preferred_categories: List[int] = field(default_factory=list)
    preferred_brands: List[str] = field(default_factory=list)
    preferred_price_range: Tuple[Decimal, Decimal] = (Decimal("0"), Decimal("0"))
    preferred_payment_method: str = ""
    preferred_shipping_method: str = ""
    
    # Location
    default_city: str = ""
    default_region: str = ""
    default_country: str = ""
    shipping_addresses_count: int = 0
    
    # Device Preferences
    primary_device: str = ""
    theme_preference: str = ""
    language_preference: str = ""
    currency_preference: str = ""
    
    # Engagement Scores
    recency_score: float = 0.0
    frequency_score: float = 0.0
    monetary_score: float = 0.0
    rfm_segment: str = ""


# =============================================================================
# DATA COLLECTOR
# =============================================================================

class DataCollector:
    """
    Comprehensive data collector for ML training.
    
    Silently collects all user behavior and product interaction data.
    Stores data in Redis for batch processing and database for persistence.
    """
    
    def __init__(self):
        self.redis_client = None
        self.geoip = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Redis and GeoIP."""
        if REDIS_AVAILABLE:
            try:
                redis_url = getattr(settings, 'ML_REDIS_URL', 'redis://localhost:6379/1')
                self.redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
        
        if DJANGO_AVAILABLE:
            try:
                self.geoip = GeoIP2()
            except Exception as e:
                logger.debug(f"GeoIP2 not available: {e}")
    
    def _get_location_from_ip(self, ip_address: str) -> Dict[str, Any]:
        """Get location data from IP address."""
        location = {
            "country": "",
            "country_code": "",
            "city": "",
            "region": "",
            "postal_code": "",
            "latitude": 0.0,
            "longitude": 0.0,
            "timezone": "",
        }
        
        if not self.geoip or not ip_address:
            return location
        
        try:
            # Skip local/private IPs
            if ip_address.startswith(('127.', '192.168.', '10.', '172.')):
                return location
            
            geo_data = self.geoip.city(ip_address)
            location.update({
                "country": geo_data.get("country_name", ""),
                "country_code": geo_data.get("country_code", ""),
                "city": geo_data.get("city", ""),
                "region": geo_data.get("region", ""),
                "postal_code": geo_data.get("postal_code", ""),
                "latitude": geo_data.get("latitude", 0.0),
                "longitude": geo_data.get("longitude", 0.0),
                "timezone": geo_data.get("time_zone", ""),
            })
        except Exception as e:
            logger.debug(f"GeoIP lookup failed for {ip_address}: {e}")
        
        return location
    
    def _parse_user_agent(self, user_agent: str) -> Dict[str, str]:
        """Parse user agent string."""
        result = {
            "device_type": "desktop",
            "browser": "",
            "browser_version": "",
            "os": "",
            "os_version": "",
        }
        
        try:
            from user_agents import parse
            ua = parse(user_agent)
            
            if ua.is_mobile:
                result["device_type"] = "mobile"
            elif ua.is_tablet:
                result["device_type"] = "tablet"
            
            result["browser"] = ua.browser.family
            result["browser_version"] = ua.browser.version_string
            result["os"] = ua.os.family
            result["os_version"] = ua.os.version_string
            
        except ImportError:
            # user_agents package not installed - use basic parsing
            if user_agent:
                ua_lower = user_agent.lower()
                if 'mobile' in ua_lower or 'android' in ua_lower or 'iphone' in ua_lower:
                    result["device_type"] = "mobile"
                elif 'tablet' in ua_lower or 'ipad' in ua_lower:
                    result["device_type"] = "tablet"
        except Exception as e:
            logger.debug(f"User agent parsing failed: {e}")
        
        return result
    
    def _extract_referrer_info(self, referrer: str) -> Dict[str, str]:
        """Extract referrer domain and UTM parameters."""
        from urllib.parse import urlparse, parse_qs
        
        result = {
            "referrer_domain": "",
            "utm_source": "",
            "utm_medium": "",
            "utm_campaign": "",
        }
        
        if not referrer:
            return result
        
        try:
            parsed = urlparse(referrer)
            result["referrer_domain"] = parsed.netloc
            
            # Extract UTM parameters
            params = parse_qs(parsed.query)
            result["utm_source"] = params.get("utm_source", [""])[0]
            result["utm_medium"] = params.get("utm_medium", [""])[0]
            result["utm_campaign"] = params.get("utm_campaign", [""])[0]
            
        except Exception as e:
            logger.debug(f"Referrer parsing failed: {e}")
        
        return result
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        import re
        if not text:
            return 0
        # Match sentence endings
        sentences = re.split(r'[.!?।]+', text)
        return len([s for s in sentences if s.strip()])
    
    def collect_user_interaction(
        self,
        request,
        event_type: str,
        page_type: str = "",
        metadata: Dict[str, Any] = None,
    ) -> UserInteraction:
        """
        Collect user interaction data from a request.
        
        Args:
            request: Django request object
            event_type: Type of event (page_view, click, etc.)
            page_type: Type of page (home, product, category, etc.)
            metadata: Additional event metadata
        
        Returns:
            UserInteraction dataclass
        """
        # Get or create session ID
        session_id = request.session.session_key or ""
        if not session_id:
            request.session.create()
            session_id = request.session.session_key
        
        # Get user ID
        user_id = None
        if hasattr(request, 'user') and request.user.is_authenticated:
            user_id = request.user.id
        
        # Anonymous ID from cookie
        anonymous_id = request.COOKIES.get('_bunoraa_aid', '')
        
        # Parse user agent
        user_agent = request.META.get('HTTP_USER_AGENT', '')
        ua_info = self._parse_user_agent(user_agent)
        
        # Get location from IP
        ip_address = self._get_client_ip(request)
        location = self._get_location_from_ip(ip_address)
        
        # Parse referrer
        referrer = request.META.get('HTTP_REFERER', '')
        referrer_info = self._extract_referrer_info(referrer)
        
        # Get theme from cookie/session
        theme = request.COOKIES.get('theme', 'light')
        
        # Create interaction record
        interaction = UserInteraction(
            session_id=session_id,
            user_id=user_id,
            anonymous_id=anonymous_id,
            event_type=event_type,
            event_timestamp=timezone.now() if DJANGO_AVAILABLE else datetime.now(),
            page_url=request.build_absolute_uri(),
            page_type=page_type,
            referrer_url=referrer,
            referrer_domain=referrer_info["referrer_domain"],
            utm_source=referrer_info["utm_source"],
            utm_medium=referrer_info["utm_medium"],
            utm_campaign=referrer_info["utm_campaign"],
            device_type=ua_info["device_type"],
            browser=ua_info["browser"],
            browser_version=ua_info["browser_version"],
            os=ua_info["os"],
            os_version=ua_info["os_version"],
            ip_address=ip_address,
            country=location["country"],
            country_code=location["country_code"],
            city=location["city"],
            region=location["region"],
            postal_code=location["postal_code"],
            latitude=location["latitude"],
            longitude=location["longitude"],
            timezone_name=location["timezone"],
            theme_used=theme,
            language=request.LANGUAGE_CODE if hasattr(request, 'LANGUAGE_CODE') else 'bn',
            currency=request.session.get('currency', 'BDT'),
            metadata=metadata or {},
        )
        
        # Store interaction
        self._store_interaction(interaction)
        
        return interaction
    
    def collect_product_interaction(
        self,
        request,
        product,
        interaction_type: str = "view",
        source_info: Dict[str, Any] = None,
    ) -> ProductInteraction:
        """
        Collect product interaction data.
        
        Args:
            request: Django request object
            product: Product model instance
            interaction_type: Type of interaction (view, click_image, etc.)
            source_info: Information about where user came from
        
        Returns:
            ProductInteraction dataclass
        """
        source_info = source_info or {}
        
        # Calculate text metrics
        title = getattr(product, 'name', '') or getattr(product, 'title', '')
        description = getattr(product, 'description', '') or ''
        
        # Get product attributes
        category = getattr(product, 'category', None)
        
        # Get price info
        price = Decimal(str(getattr(product, 'price', 0)))
        original_price = Decimal(str(getattr(product, 'original_price', price) or price))
        
        discount_amount = original_price - price if original_price > price else Decimal("0")
        discount_percent = float((discount_amount / original_price * 100)) if original_price > 0 else 0.0
        
        # Get category hierarchy
        category_hierarchy = []
        if category:
            cat = category
            while cat:
                category_hierarchy.insert(0, cat.id)
                cat = getattr(cat, 'parent', None)
        
        # Create product interaction
        interaction = ProductInteraction(
            product_id=product.id,
            product_slug=getattr(product, 'slug', ''),
            product_sku=getattr(product, 'sku', ''),
            product_title=title,
            title_char_count=len(title),
            title_word_count=len(title.split()),
            product_description=description[:500],  # Truncate for storage
            description_char_count=len(description),
            description_word_count=len(description.split()),
            description_sentence_count=self._count_sentences(description),
            category_id=category.id if category else 0,
            category_name=getattr(category, 'name', ''),
            category_hierarchy=category_hierarchy,
            price=price,
            original_price=original_price,
            discount_amount=discount_amount,
            discount_percent=discount_percent,
            has_discount=discount_amount > 0,
            color=getattr(product, 'color', ''),
            material=getattr(product, 'material', ''),
            brand=getattr(product, 'brand', '') or getattr(getattr(product, 'brand', None), 'name', ''),
            in_stock=getattr(product, 'in_stock', True),
            stock_quantity=getattr(product, 'stock_quantity', 0) or getattr(product, 'stock', 0),
            is_new_arrival=getattr(product, 'is_new_arrival', False) or getattr(product, 'is_new', False),
            is_best_seller=getattr(product, 'is_best_seller', False),
            is_spotlight=getattr(product, 'is_spotlight', False) or getattr(product, 'is_featured', False),
            is_featured=getattr(product, 'is_featured', False),
            is_preorder=getattr(product, 'is_preorder', False),
            has_variants=getattr(product, 'has_variants', False),
            variant_count=getattr(product, 'variants', None).count() if hasattr(product, 'variants') and product.variants else 0,
            image_count=getattr(product, 'images', None).count() if hasattr(product, 'images') and product.images else 0,
            has_reviews=getattr(product, 'review_count', 0) > 0,
            review_count=getattr(product, 'review_count', 0),
            average_rating=float(getattr(product, 'average_rating', 0) or 0),
            source_page=source_info.get('source_page', ''),
            source_type=source_info.get('source_type', 'direct'),
            search_query=source_info.get('search_query', ''),
            recommendation_source=source_info.get('recommendation_source', ''),
            position_in_list=source_info.get('position', 0),
        )
        
        # Store interaction
        self._store_product_interaction(interaction, request)
        
        return interaction
    
    def collect_conversion_event(
        self,
        request,
        event_type: str,
        data: Dict[str, Any] = None,
    ) -> ConversionEvent:
        """
        Collect conversion-related events.
        
        Args:
            request: Django request object
            event_type: Type of conversion event
            data: Event-specific data
        
        Returns:
            ConversionEvent dataclass
        """
        data = data or {}
        
        session_id = request.session.session_key or ""
        user_id = None
        if hasattr(request, 'user') and request.user.is_authenticated:
            user_id = request.user.id
        
        event = ConversionEvent(
            session_id=session_id,
            user_id=user_id,
            event_timestamp=timezone.now() if DJANGO_AVAILABLE else datetime.now(),
        )
        
        # Set event-specific fields
        if event_type == "add_to_cart":
            event.added_to_cart = True
            event.cart_quantity = data.get('quantity', 1)
            event.cart_item_count = data.get('cart_item_count', 0)
            event.cart_total = Decimal(str(data.get('cart_total', 0)))
        
        elif event_type == "remove_from_cart":
            event.removed_from_cart = True
            event.cart_item_count = data.get('cart_item_count', 0)
        
        elif event_type == "add_to_wishlist":
            event.added_to_wishlist = True
            event.wishlist_item_count = data.get('wishlist_item_count', 0)
        
        elif event_type == "start_checkout":
            event.started_checkout = True
            event.checkout_step = 1
            event.cart_total = Decimal(str(data.get('cart_total', 0)))
        
        elif event_type == "complete_checkout":
            event.completed_checkout = True
            event.order_id = data.get('order_id')
            event.order_total = Decimal(str(data.get('order_total', 0)))
            event.order_item_count = data.get('item_count', 0)
            event.payment_method = data.get('payment_method', '')
            event.shipping_method = data.get('shipping_method', '')
            event.shipping_location = data.get('shipping_location', '')
            event.vat_percent = data.get('vat_percent', 0)
            event.vat_amount = Decimal(str(data.get('vat_amount', 0)))
            event.shipping_cost = Decimal(str(data.get('shipping_cost', 0)))
            event.is_gift = data.get('is_gift', False)
        
        elif event_type == "abandon_checkout":
            event.checkout_abandoned = True
            event.checkout_abandon_step = data.get('step', 0)
        
        elif event_type == "apply_coupon":
            event.used_coupon = True
            event.coupon_code = data.get('coupon_code', '')
            event.coupon_discount = Decimal(str(data.get('discount', 0)))
        
        elif event_type == "share_product":
            event.shared_product = True
            event.share_platform = data.get('platform', '')
        
        # Store event
        self._store_conversion_event(event)
        
        return event
    
    def collect_user_profile(self, user) -> UserProfile:
        """
        Collect comprehensive user profile data.
        
        Args:
            user: Django user model instance
        
        Returns:
            UserProfile dataclass
        """
        now = timezone.now() if DJANGO_AVAILABLE else datetime.now()
        
        # Calculate account age
        created = getattr(user, 'date_joined', now)
        account_age = (now - created).days if created else 0
        
        # Get order stats
        orders = []
        total_spent = Decimal("0")
        if hasattr(user, 'orders'):
            orders = list(user.orders.filter(status__in=['completed', 'delivered']))
            total_spent = sum(Decimal(str(o.total)) for o in orders)
        
        # Calculate days since last order
        last_order = orders[0] if orders else None
        days_since_order = (now - last_order.created_at).days if last_order else 999
        
        # Get preferences from behavior
        preferred_categories = self._get_preferred_categories(user.id)
        preferred_brands = self._get_preferred_brands(user.id)
        
        profile = UserProfile(
            user_id=user.id,
            username=user.username,
            email=user.email,
            phone=getattr(user, 'phone', '') or getattr(user, 'phone_number', ''),
            gender=getattr(user, 'gender', ''),
            age=self._calculate_age(getattr(user, 'date_of_birth', None)),
            date_of_birth=getattr(user, 'date_of_birth', None),
            account_created=created,
            account_age_days=account_age,
            is_verified=getattr(user, 'is_verified', False),
            email_verified=getattr(user, 'email_verified', False) or user.is_active,
            phone_verified=getattr(user, 'phone_verified', False),
            trust_score=self._calculate_trust_score(user),
            loyalty_tier=getattr(user, 'loyalty_tier', ''),
            total_orders=len(orders),
            total_spent=total_spent,
            average_order_value=total_spent / len(orders) if orders else Decimal("0"),
            days_since_last_order=days_since_order,
            preferred_categories=preferred_categories,
            preferred_brands=preferred_brands,
            primary_device=self._get_primary_device(user.id),
            theme_preference=getattr(user, 'theme_preference', 'light'),
            language_preference=getattr(user, 'language', 'bn'),
            currency_preference=getattr(user, 'currency', 'BDT'),
        )
        
        # Calculate RFM scores
        profile.recency_score = self._calculate_recency_score(days_since_order)
        profile.frequency_score = self._calculate_frequency_score(len(orders))
        profile.monetary_score = self._calculate_monetary_score(total_spent)
        profile.rfm_segment = self._get_rfm_segment(
            profile.recency_score,
            profile.frequency_score,
            profile.monetary_score
        )
        
        return profile
    
    def _get_client_ip(self, request) -> str:
        """Get client IP address from request."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR', '')
        return ip
    
    def _calculate_age(self, dob) -> Optional[int]:
        """Calculate age from date of birth."""
        if not dob:
            return None
        today = datetime.now().date() if isinstance(dob, datetime) else datetime.now().date()
        dob_date = dob.date() if isinstance(dob, datetime) else dob
        return today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
    
    def _calculate_trust_score(self, user) -> float:
        """Calculate user trust score (0-100)."""
        score = 0.0
        
        # Email verified: +20
        if getattr(user, 'email_verified', False) or user.is_active:
            score += 20
        
        # Phone verified: +20
        if getattr(user, 'phone_verified', False):
            score += 20
        
        # Has completed orders: up to +30
        orders = getattr(user, 'orders', None)
        if orders:
            order_count = orders.filter(status__in=['completed', 'delivered']).count()
            score += min(order_count * 5, 30)
        
        # Account age: up to +20
        created = getattr(user, 'date_joined', None)
        if created:
            now = timezone.now() if DJANGO_AVAILABLE else datetime.now()
            age_days = (now - created).days
            score += min(age_days / 30 * 5, 20)  # Max at ~4 months
        
        # No fraud flags: +10
        if not getattr(user, 'is_flagged', False):
            score += 10
        
        return min(score, 100.0)
    
    def _calculate_recency_score(self, days: int) -> float:
        """Calculate recency score (1-5)."""
        if days <= 7:
            return 5.0
        elif days <= 30:
            return 4.0
        elif days <= 90:
            return 3.0
        elif days <= 180:
            return 2.0
        return 1.0
    
    def _calculate_frequency_score(self, order_count: int) -> float:
        """Calculate frequency score (1-5)."""
        if order_count >= 20:
            return 5.0
        elif order_count >= 10:
            return 4.0
        elif order_count >= 5:
            return 3.0
        elif order_count >= 2:
            return 2.0
        return 1.0
    
    def _calculate_monetary_score(self, total: Decimal) -> float:
        """Calculate monetary score (1-5) in BDT."""
        total_float = float(total)
        if total_float >= 50000:  # 50,000 BDT
            return 5.0
        elif total_float >= 20000:
            return 4.0
        elif total_float >= 10000:
            return 3.0
        elif total_float >= 5000:
            return 2.0
        return 1.0
    
    def _get_rfm_segment(self, r: float, f: float, m: float) -> str:
        """Get RFM customer segment."""
        avg = (r + f + m) / 3
        
        if r >= 4 and f >= 4 and m >= 4:
            return "champions"
        elif r >= 4 and f >= 3:
            return "loyal_customers"
        elif r >= 4 and f <= 2:
            return "potential_loyalists"
        elif r >= 3 and f >= 3:
            return "at_risk"
        elif r <= 2 and f >= 3:
            return "cant_lose"
        elif r <= 2 and f <= 2 and m >= 3:
            return "hibernating"
        elif r <= 2:
            return "lost"
        else:
            return "others"
    
    def _get_preferred_categories(self, user_id: int) -> List[int]:
        """Get user's preferred categories based on behavior."""
        if not self.redis_client:
            return []
        
        try:
            key = f"ml:user:{user_id}:categories"
            categories = self.redis_client.zrevrange(key, 0, 9)
            return [int(c) for c in categories]
        except Exception:
            return []
    
    def _get_preferred_brands(self, user_id: int) -> List[str]:
        """Get user's preferred brands based on behavior."""
        if not self.redis_client:
            return []
        
        try:
            key = f"ml:user:{user_id}:brands"
            brands = self.redis_client.zrevrange(key, 0, 9)
            return [b.decode() if isinstance(b, bytes) else b for b in brands]
        except Exception:
            return []
    
    def _get_primary_device(self, user_id: int) -> str:
        """Get user's primary device type."""
        if not self.redis_client:
            return "desktop"
        
        try:
            key = f"ml:user:{user_id}:devices"
            devices = self.redis_client.zrevrange(key, 0, 0)
            if devices:
                return devices[0].decode() if isinstance(devices[0], bytes) else devices[0]
        except Exception:
            pass
        return "desktop"
    
    # =========================================================================
    # STORAGE METHODS
    # =========================================================================
    
    def _store_interaction(self, interaction: UserInteraction):
        """Store user interaction in Redis and queue for database."""
        if not self.redis_client:
            return
        
        try:
            data = asdict(interaction)
            data['event_timestamp'] = interaction.event_timestamp.isoformat()
            
            # Store in Redis list for batch processing
            key = "ml:interactions:queue"
            self.redis_client.rpush(key, json.dumps(data, default=str))
            
            # Update user activity metrics
            if interaction.user_id:
                self._update_user_metrics(interaction)
            
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
    
    def _store_product_interaction(self, interaction: ProductInteraction, request):
        """Store product interaction data."""
        if not self.redis_client:
            return
        
        try:
            session_id = request.session.session_key or ""
            user_id = None
            if hasattr(request, 'user') and request.user.is_authenticated:
                user_id = request.user.id
            
            data = asdict(interaction)
            data['session_id'] = session_id
            data['user_id'] = user_id
            data['timestamp'] = datetime.now().isoformat()
            
            # Store in Redis list
            key = "ml:product_interactions:queue"
            self.redis_client.rpush(key, json.dumps(data, default=str))
            
            # Update category preferences
            if user_id and interaction.category_id:
                cat_key = f"ml:user:{user_id}:categories"
                self.redis_client.zincrby(cat_key, 1, interaction.category_id)
            
            # Update brand preferences
            if user_id and interaction.brand:
                brand_key = f"ml:user:{user_id}:brands"
                self.redis_client.zincrby(brand_key, 1, interaction.brand)
            
            # Update product view counts
            prod_key = f"ml:product:{interaction.product_id}:views"
            self.redis_client.incr(prod_key)
            
        except Exception as e:
            logger.error(f"Failed to store product interaction: {e}")
    
    def _store_conversion_event(self, event: ConversionEvent):
        """Store conversion event data."""
        if not self.redis_client:
            return
        
        try:
            data = asdict(event)
            data['event_timestamp'] = event.event_timestamp.isoformat()
            
            # Store in Redis list
            key = "ml:conversions:queue"
            self.redis_client.rpush(key, json.dumps(data, default=str))
            
            # Update conversion metrics
            if event.completed_checkout:
                self.redis_client.incr("ml:metrics:total_conversions")
                self.redis_client.incrbyfloat("ml:metrics:total_revenue", float(event.order_total))
            
        except Exception as e:
            logger.error(f"Failed to store conversion event: {e}")
    
    def _update_user_metrics(self, interaction: UserInteraction):
        """Update user activity metrics in Redis."""
        if not self.redis_client or not interaction.user_id:
            return
        
        try:
            user_id = interaction.user_id
            
            # Update device preference
            device_key = f"ml:user:{user_id}:devices"
            self.redis_client.zincrby(device_key, 1, interaction.device_type)
            
            # Update last activity
            activity_key = f"ml:user:{user_id}:last_activity"
            self.redis_client.set(activity_key, interaction.event_timestamp.isoformat())
            
            # Update session count
            session_key = f"ml:user:{user_id}:sessions"
            self.redis_client.sadd(session_key, interaction.session_id)
            
        except Exception as e:
            logger.error(f"Failed to update user metrics: {e}")
    
    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================
    
    def process_queued_data(self, batch_size: int = 1000) -> Dict[str, int]:
        """
        Process queued interaction data and store in database.
        
        Args:
            batch_size: Number of records to process per queue
        
        Returns:
            Dictionary with counts of processed records
        """
        results = {
            "interactions": 0,
            "product_interactions": 0,
            "conversions": 0,
        }
        
        if not self.redis_client:
            return results
        
        # Process user interactions
        results["interactions"] = self._process_queue(
            "ml:interactions:queue",
            self._save_interactions_batch,
            batch_size
        )
        
        # Process product interactions
        results["product_interactions"] = self._process_queue(
            "ml:product_interactions:queue",
            self._save_product_interactions_batch,
            batch_size
        )
        
        # Process conversions
        results["conversions"] = self._process_queue(
            "ml:conversions:queue",
            self._save_conversions_batch,
            batch_size
        )
        
        return results
    
    def _process_queue(self, queue_key: str, save_func, batch_size: int) -> int:
        """Process a Redis queue."""
        processed = 0
        
        try:
            while True:
                # Get batch from queue
                items = self.redis_client.lrange(queue_key, 0, batch_size - 1)
                
                if not items:
                    break
                
                # Parse and save
                records = [json.loads(item) for item in items]
                save_func(records)
                
                # Remove processed items
                self.redis_client.ltrim(queue_key, len(items), -1)
                processed += len(items)
                
                if len(items) < batch_size:
                    break
                    
        except Exception as e:
            logger.error(f"Failed to process queue {queue_key}: {e}")
        
        return processed
    
    def _save_interactions_batch(self, records: List[Dict]):
        """Save user interactions to database."""
        try:
            from apps.analytics.models import UserInteraction as InteractionModel
            
            with transaction.atomic():
                for record in records:
                    InteractionModel.objects.create(
                        session_id=record.get('session_id', ''),
                        user_id=record.get('user_id'),
                        event_type=record.get('event_type', ''),
                        page_url=record.get('page_url', ''),
                        page_type=record.get('page_type', ''),
                        device_type=record.get('device_type', ''),
                        country=record.get('country', ''),
                        city=record.get('city', ''),
                        referrer_domain=record.get('referrer_domain', ''),
                        time_on_page=record.get('time_on_page_seconds', 0),
                        metadata=record.get('metadata', {}),
                    )
        except Exception as e:
            logger.error(f"Failed to save interactions batch: {e}")
    
    def _save_product_interactions_batch(self, records: List[Dict]):
        """Save product interactions to database."""
        try:
            from apps.analytics.models import ProductInteraction as ProdInteractionModel
            
            with transaction.atomic():
                for record in records:
                    ProdInteractionModel.objects.create(
                        session_id=record.get('session_id', ''),
                        user_id=record.get('user_id'),
                        product_id=record.get('product_id'),
                        category_id=record.get('category_id'),
                        source_type=record.get('source_type', ''),
                        time_viewing=record.get('time_viewing_seconds', 0),
                        clicked_image=record.get('clicked_image', False),
                        clicked_variant=record.get('clicked_variant', False),
                    )
        except Exception as e:
            logger.error(f"Failed to save product interactions batch: {e}")
    
    def _save_conversions_batch(self, records: List[Dict]):
        """Save conversion events to database."""
        try:
            from apps.analytics.models import ConversionEvent as ConversionModel
            
            with transaction.atomic():
                for record in records:
                    ConversionModel.objects.create(
                        session_id=record.get('session_id', ''),
                        user_id=record.get('user_id'),
                        order_id=record.get('order_id'),
                        event_type=self._determine_conversion_type(record),
                        order_total=record.get('order_total', 0),
                        coupon_used=record.get('used_coupon', False),
                        coupon_code=record.get('coupon_code', ''),
                    )
        except Exception as e:
            logger.error(f"Failed to save conversions batch: {e}")
    
    def _determine_conversion_type(self, record: Dict) -> str:
        """Determine conversion event type from record."""
        if record.get('completed_checkout'):
            return 'purchase'
        elif record.get('started_checkout'):
            return 'checkout_start'
        elif record.get('checkout_abandoned'):
            return 'checkout_abandon'
        elif record.get('added_to_cart'):
            return 'add_to_cart'
        elif record.get('added_to_wishlist'):
            return 'add_to_wishlist'
        return 'unknown'
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get current queue sizes."""
        stats = {}
        
        if self.redis_client:
            try:
                stats['interactions'] = self.redis_client.llen("ml:interactions:queue")
                stats['product_interactions'] = self.redis_client.llen("ml:product_interactions:queue")
                stats['conversions'] = self.redis_client.llen("ml:conversions:queue")
            except Exception:
                pass
        
        return stats
