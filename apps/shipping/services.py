"""
Shipping Services
Business logic for shipping calculations, carrier integrations, and tracking.
"""
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple
from django.db import transaction
from django.utils import timezone
from django.core.cache import cache

from .models import (
    ShippingZone, ShippingCarrier, ShippingMethod, ShippingRate,
    ShippingRestriction, Shipment, ShipmentEvent, ShippingSettings
)


class ShippingZoneService:
    """Service for shipping zone operations."""
    
    CACHE_KEY_ZONES = 'shipping_zones_active'
    CACHE_TIMEOUT = 3600  # 1 hour
    
    @classmethod
    def get_active_zones(cls) -> List[ShippingZone]:
        """Get all active shipping zones, cached."""
        zones = cache.get(cls.CACHE_KEY_ZONES)
        if zones is None:
            zones = list(
                ShippingZone.objects.filter(is_active=True)
                .order_by('-priority', 'name')
            )
            cache.set(cls.CACHE_KEY_ZONES, zones, cls.CACHE_TIMEOUT)
        return zones
    
    @classmethod
    def find_zone_for_location(
        cls,
        country: str,
        state: Optional[str] = None,
        postal_code: Optional[str] = None,
        city: Optional[str] = None
    ) -> Optional[ShippingZone]:
        """Find the best matching zone for a location."""
        zones = cls.get_active_zones()
        
        # Check each zone by priority
        for zone in zones:
            if zone.matches_location(country=country, state=state, city=city, postal_code=postal_code):
                return zone
        
        # Return default zone if exists
        for zone in zones:
            if zone.is_default:
                return zone
        
        return None
    
    @classmethod
    def clear_cache(cls):
        """Clear zone cache."""
        cache.delete(cls.CACHE_KEY_ZONES)


class ShippingRateService:
    """Service for shipping rate calculations."""
    
    @classmethod
    def get_available_methods(
        cls,
        country: str,
        state: Optional[str] = None,
        postal_code: Optional[str] = None,
        city: Optional[str] = None,
        subtotal: Decimal = Decimal('0'),
        weight: Decimal = Decimal('0'),
        item_count: int = 1,
        product_ids: Optional[List[str]] = None,
        currency_code: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available shipping methods with calculated rates.
        
        Returns list of dicts with method details and calculated rates.
        """
        # Find zone
        zone = ShippingZoneService.find_zone_for_location(country, state, postal_code, city=city)
        if not zone:
            return []
        
        # Get rates for this zone
        rates = ShippingRate.objects.filter(
            zone=zone,
            is_active=True,
            method__is_active=True
        ).select_related('method', 'method__carrier')
        
        # Check restrictions
        restrictions = cls._get_applicable_restrictions(zone, product_ids)
        blocked_methods = set()
        surcharges = {}
        
        for restriction in restrictions:
            if restriction.action == ShippingRestriction.ACTION_BLOCK:
                if restriction.method:
                    blocked_methods.add(restriction.method_id)
            elif restriction.action == ShippingRestriction.ACTION_SURCHARGE:
                if restriction.method:
                    surcharges[restriction.method_id] = surcharges.get(
                        restriction.method_id, Decimal('0')
                    ) + restriction.surcharge_amount
        
        # Calculate rates
        available_methods = []
        settings = ShippingSettings.get_settings()
        
        try:
            from apps.i18n.services import CurrencyService, CurrencyConversionService
            default_currency = CurrencyService.get_default_currency()
            target_currency = CurrencyService.get_currency_by_code(currency_code) if currency_code else None
        except Exception:
            default_currency = None
            target_currency = None
            CurrencyConversionService = None

        for rate in rates:
            method = rate.method
            
            # Skip blocked methods
            if method.id in blocked_methods:
                continue
            
            # Check weight restrictions
            if method.max_weight and weight > method.max_weight:
                continue
            
            # Calculate base rate
            calculated_rate = rate.calculate_rate(subtotal, weight, item_count)
            
            # Add surcharges
            if method.id in surcharges:
                calculated_rate += surcharges[method.id]
            
            # Check global free shipping
            if settings.enable_free_shipping and settings.free_shipping_threshold:
                if subtotal >= settings.free_shipping_threshold:
                    if not settings.free_shipping_countries or country in settings.free_shipping_countries:
                        calculated_rate = Decimal('0.00')
            
            # Prepare currency-aware display (convert to user's currency if requested)
            rate_currency_obj = getattr(rate, 'currency', None) or default_currency
            display_currency = rate_currency_obj
            display_rate = calculated_rate
            if target_currency and rate_currency_obj and target_currency.code != rate_currency_obj.code and CurrencyConversionService:
                try:
                    display_rate = CurrencyConversionService.convert_by_code(
                        calculated_rate, rate_currency_obj.code, target_currency.code, round_result=True
                    )
                    display_currency = target_currency
                except Exception:
                    display_rate = calculated_rate
                    display_currency = rate_currency_obj

            try:
                if display_currency:
                    rate_display = display_currency.format_amount(display_rate) if display_rate > 0 else 'Free'
                    currency_meta = {
                        'code': display_currency.code,
                        'symbol': display_currency.symbol,
                        'decimal_places': display_currency.decimal_places,
                    }
                else:
                    rate_display = f"${display_rate:.2f}" if display_rate > 0 else 'Free'
                    currency_meta = None
            except Exception:
                rate_display = f"${display_rate:.2f}" if display_rate > 0 else 'Free'
                currency_meta = None

            available_methods.append({
                'id': str(method.id),
                'method_id': str(method.id),
                'rate_id': str(rate.id),
                'code': method.code,
                'name': method.name,
                'description': method.description,
                'carrier': {
                    'id': str(method.carrier.id) if method.carrier else None,
                    'name': method.carrier.name if method.carrier else None,
                    'logo': method.carrier.logo.url if method.carrier and method.carrier.logo else None,
                } if method.carrier else None,
                'rate': float(display_rate),
                'rate_display': rate_display,
                'currency': currency_meta,
                'is_free': display_rate == 0,
                'delivery_estimate': method.delivery_estimate,
                'min_days': method.min_delivery_days,
                'max_days': method.max_delivery_days,
                'is_express': method.is_express,
                'requires_signature': method.requires_signature,
            })
        
        # Sort by rate
        available_methods.sort(key=lambda x: x['rate'])
        
        return available_methods
    
    @classmethod
    def _get_applicable_restrictions(
        cls,
        zone: ShippingZone,
        product_ids: Optional[List[str]] = None
    ) -> List[ShippingRestriction]:
        """Get applicable restrictions for zone and products."""
        restrictions = ShippingRestriction.objects.filter(
            is_active=True
        ).filter(
            models.Q(zone=zone) | models.Q(zone__isnull=True)
        )
        
        if product_ids:
            # Filter by product restrictions
            # This would need to check product_ids and category_ids
            pass
        
        return list(restrictions)
    
    @classmethod
    def calculate_shipping(
        cls,
        method_id: str,
        country: str,
        state: Optional[str] = None,
        postal_code: Optional[str] = None,
        subtotal: Decimal = Decimal('0'),
        weight: Decimal = Decimal('0'),
        item_count: int = 1
    ) -> Optional[Decimal]:
        """Calculate shipping cost for a specific method."""
        zone = ShippingZoneService.find_zone_for_location(country, state, postal_code)
        if not zone:
            return None
        
        try:
            rate = ShippingRate.objects.get(
                zone=zone,
                method_id=method_id,
                is_active=True
            )
            return rate.calculate_rate(subtotal, weight, item_count)
        except ShippingRate.DoesNotExist:
            return None


class ShipmentService:
    """Service for shipment management and tracking."""
    
    @classmethod
    @transaction.atomic
    def create_shipment(
        cls,
        order,
        carrier_id: Optional[str] = None,
        method_id: Optional[str] = None,
        tracking_number: Optional[str] = None,
        weight: Optional[Decimal] = None,
        dimensions: Optional[Dict] = None
    ) -> Shipment:
        """Create a new shipment for an order."""
        shipment = Shipment.objects.create(
            order=order,
            carrier_id=carrier_id,
            method_id=method_id,
            tracking_number=tracking_number or '',
            weight=weight,
            dimensions=dimensions or {},
            shipping_cost=order.shipping_cost
        )
        
        # Create initial event
        ShipmentEvent.objects.create(
            shipment=shipment,
            status='created',
            description='Shipment created',
            occurred_at=timezone.now()
        )
        
        return shipment
    
    @classmethod
    @transaction.atomic
    def update_tracking(
        cls,
        shipment: Shipment,
        tracking_number: str,
        carrier_id: Optional[str] = None
    ) -> Shipment:
        """Update shipment tracking information."""
        shipment.tracking_number = tracking_number
        if carrier_id:
            shipment.carrier_id = carrier_id
        
        # Auto-generate tracking URL
        if shipment.carrier:
            shipment.tracking_url = shipment.carrier.get_tracking_url(tracking_number) or ''
        
        shipment.save()
        
        # Add tracking event
        ShipmentEvent.objects.create(
            shipment=shipment,
            status='tracking_added',
            description=f'Tracking number added: {tracking_number}',
            occurred_at=timezone.now()
        )
        
        return shipment
    
    @classmethod
    @transaction.atomic
    def mark_shipped(
        cls,
        shipment: Shipment,
        tracking_number: Optional[str] = None
    ) -> Shipment:
        """Mark shipment as shipped."""
        shipment.status = Shipment.STATUS_IN_TRANSIT
        shipment.shipped_at = timezone.now()
        
        if tracking_number:
            shipment.tracking_number = tracking_number
            if shipment.carrier:
                shipment.tracking_url = shipment.carrier.get_tracking_url(tracking_number) or ''
        
        shipment.save()
        
        # Add event
        ShipmentEvent.objects.create(
            shipment=shipment,
            status='shipped',
            description='Package has been shipped',
            occurred_at=timezone.now()
        )
        
        # Update order status
        from apps.orders.services import OrderService
        OrderService.update_status(shipment.order, 'shipped')
        
        return shipment
    
    @classmethod
    @transaction.atomic
    def mark_delivered(
        cls,
        shipment: Shipment,
        signed_by: Optional[str] = None
    ) -> Shipment:
        """Mark shipment as delivered."""
        shipment.status = Shipment.STATUS_DELIVERED
        shipment.delivered_at = timezone.now()
        
        if signed_by:
            shipment.signed_by = signed_by
        
        shipment.save()
        
        # Add event
        ShipmentEvent.objects.create(
            shipment=shipment,
            status='delivered',
            description=f'Package delivered{" - Signed by: " + signed_by if signed_by else ""}',
            occurred_at=timezone.now()
        )
        
        # Update order status if all shipments delivered
        order = shipment.order
        all_delivered = all(
            s.status == Shipment.STATUS_DELIVERED
            for s in order.shipments.all()
        )
        if all_delivered:
            from apps.orders.services import OrderService
            OrderService.update_status(order, 'delivered')
        
        return shipment
    
    @classmethod
    def add_tracking_event(
        cls,
        shipment: Shipment,
        status: str,
        description: str,
        location: str = '',
        occurred_at=None
    ) -> ShipmentEvent:
        """Add a tracking event to a shipment."""
        return ShipmentEvent.objects.create(
            shipment=shipment,
            status=status,
            description=description,
            location=location,
            occurred_at=occurred_at or timezone.now()
        )
    
    @classmethod
    def get_tracking_history(cls, shipment: Shipment) -> List[Dict]:
        """Get formatted tracking history for a shipment."""
        events = shipment.events.all()
        return [
            {
                'status': event.status,
                'description': event.description,
                'location': event.location,
                'timestamp': event.occurred_at.isoformat(),
            }
            for event in events
        ]


class CarrierIntegrationService:
    """Service for carrier API integrations."""
    
    @classmethod
    def get_real_time_rates(
        cls,
        carrier: ShippingCarrier,
        origin: Dict,
        destination: Dict,
        packages: List[Dict]
    ) -> List[Dict]:
        """Get real-time rates from carrier API."""
        if not carrier.api_enabled:
            return []
        
        # Carrier-specific implementations
        if carrier.code == 'ups':
            return cls._get_ups_rates(carrier, origin, destination, packages)
        elif carrier.code == 'fedex':
            return cls._get_fedex_rates(carrier, origin, destination, packages)
        elif carrier.code == 'usps':
            return cls._get_usps_rates(carrier, origin, destination, packages)
        
        return []
    
    @classmethod
    def _get_ups_rates(cls, carrier, origin, destination, packages):
        """Get rates from UPS API."""
        # Implementation would use UPS API
        # This is a placeholder
        return []
    
    @classmethod
    def _get_fedex_rates(cls, carrier, origin, destination, packages):
        """Get rates from FedEx API."""
        # Implementation would use FedEx API
        return []
    
    @classmethod
    def _get_usps_rates(cls, carrier, origin, destination, packages):
        """Get rates from USPS API."""
        # Implementation would use USPS API
        return []
    
    @classmethod
    def create_shipping_label(
        cls,
        carrier: ShippingCarrier,
        shipment: Shipment,
        origin: Dict,
        destination: Dict,
        package: Dict
    ) -> Optional[Dict]:
        """Create shipping label via carrier API."""
        if not carrier.api_enabled or not carrier.supports_label_generation:
            return None
        
        # Carrier-specific implementations
        # Returns dict with label_url, tracking_number, etc.
        return None
    
    @classmethod
    def track_shipment(
        cls,
        carrier: ShippingCarrier,
        tracking_number: str
    ) -> Optional[Dict]:
        """Get tracking info from carrier API."""
        if not carrier.api_enabled or not carrier.supports_tracking:
            return None
        
        # Implementation would query carrier API
        return None


# Import models for type hints
from django.db import models
