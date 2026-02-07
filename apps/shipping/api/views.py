"""
Shipping API Views
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.views import APIView

from ..models import (
    ShippingZone, ShippingCarrier, ShippingMethod,
    Shipment, ShippingSettings
)
from ..services import ShippingZoneService, ShippingRateService, ShipmentService
from .serializers import (
    ShippingZoneSerializer, ShippingCarrierSerializer, ShippingMethodSerializer,
    ShippingRateCalculationSerializer,
    AvailableShippingMethodSerializer, ShipmentSerializer,
    ShipmentCreateSerializer, TrackingUpdateSerializer,
    ShippingSettingsSerializer
)


class ShippingZoneViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for shipping zones."""
    queryset = ShippingZone.objects.filter(is_active=True)
    serializer_class = ShippingZoneSerializer
    permission_classes = [AllowAny]


class ShippingCarrierViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for shipping carriers."""
    queryset = ShippingCarrier.objects.filter(is_active=True)
    serializer_class = ShippingCarrierSerializer
    permission_classes = [AllowAny]


class ShippingMethodViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for shipping methods."""
    queryset = ShippingMethod.objects.filter(is_active=True)
    serializer_class = ShippingMethodSerializer
    permission_classes = [AllowAny]


class ShippingRateCalculationView(APIView):
    """
    Calculate available shipping rates for a destination.
    
    POST /api/v1/shipping/calculate/
    """
    permission_classes = [AllowAny]
    
    def post(self, request):
        serializer = ShippingRateCalculationSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid request data',
                'data': None,
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        try:
            from apps.commerce.services import CheckoutService

            normalized_country = CheckoutService.normalize_country_code(
                data.get('country')
            )
            if normalized_country:
                data['country'] = normalized_country
        except Exception:
            pass

        # Normalize subtotal to default currency for calculations
        try:
            from decimal import Decimal
            from apps.i18n.services import CurrencyService, CurrencyConversionService
            target_currency = CurrencyService.get_user_currency(request=request)
            target_code = target_currency.code if target_currency else None
            default_currency = CurrencyService.get_default_currency()
            default_code = default_currency.code if default_currency else None
            subtotal = data.get('subtotal', Decimal('0'))
            if target_code and default_code and target_code != default_code:
                subtotal = CurrencyConversionService.convert_by_code(
                    Decimal(str(subtotal)), target_code, default_code, round_result=False
                )
            data['subtotal'] = subtotal
        except Exception:
            target_code = None
        
        # Get available methods with rates
        methods = ShippingRateService.get_available_methods(
            country=data['country'],
            state=data.get('state'),
            postal_code=data.get('postal_code'),
            subtotal=data.get('subtotal', 0),
            weight=data.get('weight', 0),
            item_count=data.get('item_count', 1),
            product_ids=[str(pid) for pid in data.get('product_ids', [])],
            currency_code=target_code
        )
        
        return Response({
            'success': True,
            'message': f'Found {len(methods)} shipping method(s)',
            'data': {
                'methods': methods,
                'zone': self._get_zone_info(data['country'], data.get('state'), data.get('postal_code'))
            }
        })
    
    def _get_zone_info(self, country, state, postal_code):
        zone = ShippingZoneService.find_zone_for_location(country, state, postal_code)
        if zone:
            return {
                'id': str(zone.id),
                'name': zone.name
            }
        return None


class ShipmentViewSet(viewsets.ModelViewSet):
    """API endpoint for shipment management."""
    serializer_class = ShipmentSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            return Shipment.objects.all()
        return Shipment.objects.filter(order__user=user)
    
    def list(self, request):
        """List shipments."""
        queryset = self.get_queryset()
        
        # Filter by order
        order_id = request.query_params.get('order')
        if order_id:
            queryset = queryset.filter(order_id=order_id)
        
        # Filter by status
        status_filter = request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'message': 'Shipments retrieved',
            'data': serializer.data
        })
    
    def retrieve(self, request, pk=None):
        """Get shipment details with tracking history."""
        try:
            shipment = self.get_queryset().get(pk=pk)
        except Shipment.DoesNotExist:
            return Response({
                'success': False,
                'message': 'Shipment not found',
                'data': None
            }, status=status.HTTP_404_NOT_FOUND)
        
        serializer = self.get_serializer(shipment)
        return Response({
            'success': True,
            'message': 'Shipment retrieved',
            'data': serializer.data
        })
    
    def create(self, request):
        """Create a new shipment (admin only)."""
        if not request.user.is_staff:
            return Response({
                'success': False,
                'message': 'Permission denied',
                'data': None
            }, status=status.HTTP_403_FORBIDDEN)
        
        serializer = ShipmentCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid data',
                'data': None,
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        # Get order
        from apps.orders.models import Order
        try:
            order = Order.objects.get(id=data['order_id'])
        except Order.DoesNotExist:
            return Response({
                'success': False,
                'message': 'Order not found',
                'data': None
            }, status=status.HTTP_404_NOT_FOUND)
        
        shipment = ShipmentService.create_shipment(
            order=order,
            carrier_id=str(data.get('carrier_id')) if data.get('carrier_id') else None,
            method_id=str(data.get('method_id')) if data.get('method_id') else None,
            tracking_number=data.get('tracking_number'),
            weight=data.get('weight'),
            dimensions=data.get('dimensions')
        )
        
        return Response({
            'success': True,
            'message': 'Shipment created',
            'data': ShipmentSerializer(shipment).data
        }, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['post'])
    def ship(self, request, pk=None):
        """Mark shipment as shipped."""
        if not request.user.is_staff:
            return Response({
                'success': False,
                'message': 'Permission denied',
                'data': None
            }, status=status.HTTP_403_FORBIDDEN)
        
        try:
            shipment = Shipment.objects.get(pk=pk)
        except Shipment.DoesNotExist:
            return Response({
                'success': False,
                'message': 'Shipment not found',
                'data': None
            }, status=status.HTTP_404_NOT_FOUND)
        
        tracking_number = request.data.get('tracking_number')
        shipment = ShipmentService.mark_shipped(shipment, tracking_number)
        
        return Response({
            'success': True,
            'message': 'Shipment marked as shipped',
            'data': ShipmentSerializer(shipment).data
        })
    
    @action(detail=True, methods=['post'])
    def deliver(self, request, pk=None):
        """Mark shipment as delivered."""
        if not request.user.is_staff:
            return Response({
                'success': False,
                'message': 'Permission denied',
                'data': None
            }, status=status.HTTP_403_FORBIDDEN)
        
        try:
            shipment = Shipment.objects.get(pk=pk)
        except Shipment.DoesNotExist:
            return Response({
                'success': False,
                'message': 'Shipment not found',
                'data': None
            }, status=status.HTTP_404_NOT_FOUND)
        
        signed_by = request.data.get('signed_by')
        shipment = ShipmentService.mark_delivered(shipment, signed_by)
        
        return Response({
            'success': True,
            'message': 'Shipment marked as delivered',
            'data': ShipmentSerializer(shipment).data
        })
    
    @action(detail=True, methods=['post'])
    def add_event(self, request, pk=None):
        """Add tracking event to shipment."""
        if not request.user.is_staff:
            return Response({
                'success': False,
                'message': 'Permission denied',
                'data': None
            }, status=status.HTTP_403_FORBIDDEN)
        
        try:
            shipment = Shipment.objects.get(pk=pk)
        except Shipment.DoesNotExist:
            return Response({
                'success': False,
                'message': 'Shipment not found',
                'data': None
            }, status=status.HTTP_404_NOT_FOUND)
        
        serializer = TrackingUpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid data',
                'data': None,
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        event = ShipmentService.add_tracking_event(
            shipment=shipment,
            status=data['status'],
            description=data['description'],
            location=data.get('location', ''),
            occurred_at=data.get('occurred_at')
        )
        
        return Response({
            'success': True,
            'message': 'Tracking event added',
            'data': ShipmentSerializer(shipment).data
        })


class TrackingView(APIView):
    """
    Public tracking endpoint.
    
    GET /api/v1/shipping/track/{tracking_number}/
    """
    permission_classes = [AllowAny]
    
    def get(self, request, tracking_number):
        try:
            shipment = Shipment.objects.get(tracking_number=tracking_number)
        except Shipment.DoesNotExist:
            return Response({
                'success': False,
                'message': 'Tracking number not found',
                'data': None
            }, status=status.HTTP_404_NOT_FOUND)
        
        tracking_history = ShipmentService.get_tracking_history(shipment)
        
        return Response({
            'success': True,
            'message': 'Tracking info retrieved',
            'data': {
                'tracking_number': shipment.tracking_number,
                'carrier': shipment.carrier.name if shipment.carrier else None,
                'status': shipment.status,
                'status_display': shipment.get_status_display(),
                'shipped_at': shipment.shipped_at,
                'estimated_delivery': shipment.estimated_delivery,
                'delivered_at': shipment.delivered_at,
                'tracking_url': shipment.tracking_url,
                'history': tracking_history
            }
        })


class ShippingSettingsView(APIView):
    """Get public shipping settings."""
    permission_classes = [AllowAny]
    
    def get(self, request):
        settings = ShippingSettings.get_settings()
        serializer = ShippingSettingsSerializer(settings)
        return Response({
            'success': True,
            'message': 'Shipping settings retrieved',
            'data': serializer.data
        })
