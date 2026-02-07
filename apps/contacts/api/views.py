"""
Contacts API Views
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser

from ..models import ContactCategory, StoreLocation, ContactSettings
from ..services import (
    ContactInquiryService, ContactAttachmentService, StoreLocationService
)
from .serializers import (
    ContactCategorySerializer, ContactInquiryCreateSerializer,
    ContactInquirySerializer, ContactInquiryDetailSerializer,
    StoreLocationSerializer, StoreLocationMinimalSerializer, StoreLocationPickupSerializer,
    ContactSettingsPublicSerializer, NearbyLocationRequestSerializer,
    CustomizationRequestSerializer
)


class ContactCategoryViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for contact categories."""
    
    queryset = ContactCategory.objects.filter(is_active=True)
    serializer_class = ContactCategorySerializer
    permission_classes = [AllowAny]
    lookup_field = 'slug'
    
    def list(self, request):
        """List all active contact categories."""
        categories = self.get_queryset().order_by('order')
        serializer = self.get_serializer(categories, many=True)
        return Response({
            'success': True,
            'data': serializer.data
        })


class ContactInquiryView(APIView):
    """View for contact inquiries."""
    
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        """Create a new contact inquiry."""
        serializer = ContactInquiryCreateSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid data',
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        # Get client info
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip_address = x_forwarded_for.split(',')[0].strip()
        else:
            ip_address = request.META.get('REMOTE_ADDR')
        
        user_agent = request.META.get('HTTP_USER_AGENT', '')
        referer = request.META.get('HTTP_REFERER', '')
        
        inquiry = ContactInquiryService.create_inquiry(
            name=data['name'],
            email=data['email'],
            subject=data['subject'],
            message=data['message'],
            category_id=data.get('category_id'),
            phone=data.get('phone', ''),
            company=data.get('company', ''),
            order_number=data.get('order_number', ''),
            user=request.user if request.user.is_authenticated else None,
            ip_address=ip_address,
            user_agent=user_agent,
            source_page=referer
        )
        
        # Handle attachments
        settings = ContactSettings.get_settings()
        if settings.allow_attachments:
            for key in request.FILES:
                file = request.FILES[key]
                attachment, error = ContactAttachmentService.add_attachment(inquiry, file)
                if error:
                    # Log error but don't fail the submission
                    pass
        
        return Response({
            'success': True,
            'message': 'Thank you for contacting us! We will respond shortly.',
            'data': ContactInquirySerializer(inquiry).data
        }, status=status.HTTP_201_CREATED)


class UserInquiriesView(APIView):
    """View for user's inquiries."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get current user's inquiries."""
        inquiries = ContactInquiryService.get_user_inquiries(request.user)
        serializer = ContactInquirySerializer(inquiries, many=True)
        return Response({
            'success': True,
            'data': serializer.data
        })


class InquiryDetailView(APIView):
    """View for inquiry detail."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request, inquiry_id):
        """Get inquiry detail."""
        inquiry = ContactInquiryService.get_inquiry(inquiry_id)
        
        if not inquiry:
            return Response({
                'success': False,
                'message': 'Inquiry not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Check permission
        if inquiry.user != request.user and not request.user.is_staff:
            return Response({
                'success': False,
                'message': 'Access denied'
            }, status=status.HTTP_403_FORBIDDEN)
        
        serializer = ContactInquiryDetailSerializer(inquiry)
        return Response({
            'success': True,
            'data': serializer.data
        })


class StoreLocationViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for store locations."""
    
    queryset = StoreLocation.objects.filter(is_active=True)
    permission_classes = [AllowAny]
    lookup_field = 'slug'
    
    def get_serializer_class(self):
        if self.action == 'list':
            return StoreLocationMinimalSerializer
        return StoreLocationSerializer
    
    def list(self, request):
        """List all active locations."""
        locations = StoreLocationService.get_all_locations()
        serializer = StoreLocationSerializer(locations, many=True)
        return Response({
            'success': True,
            'data': serializer.data
        })
    
    def retrieve(self, request, slug=None):
        """Get location detail."""
        location = StoreLocationService.get_location_by_slug(slug)
        if not location:
            return Response({
                'success': False,
                'message': 'Location not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        serializer = StoreLocationSerializer(location)
        return Response({
            'success': True,
            'data': serializer.data
        })
    
    @action(detail=False, methods=['get'])
    def main(self, request):
        """Get main/headquarters location."""
        location = StoreLocationService.get_main_location()
        if not location:
            return Response({
                'success': False,
                'message': 'No main location configured'
            }, status=status.HTTP_404_NOT_FOUND)
        
        serializer = StoreLocationSerializer(location)
        return Response({
            'success': True,
            'data': serializer.data
        })
    
    @action(detail=False, methods=['get'])
    def pickup(self, request):
        """Get pickup locations."""
        locations = StoreLocationService.get_pickup_locations()
        serializer = StoreLocationPickupSerializer(locations, many=True)
        return Response({
            'success': True,
            'data': serializer.data
        })
    
    @action(detail=False, methods=['get'])
    def returns(self, request):
        """Get returns locations."""
        locations = StoreLocationService.get_returns_locations()
        serializer = StoreLocationMinimalSerializer(locations, many=True)
        return Response({
            'success': True,
            'data': serializer.data
        })


class NearbyLocationsView(APIView):
    """View for finding nearby locations."""
    
    permission_classes = [AllowAny]
    
    def get(self, request):
        """Find locations near given coordinates."""
        serializer = NearbyLocationRequestSerializer(data=request.query_params)
        
        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid coordinates',
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        locations = StoreLocationService.get_nearby_locations(
            latitude=data['latitude'],
            longitude=data['longitude'],
            radius_km=data.get('radius_km', 50)
        )
        
        # Add distance to response
        location_data = []
        for loc in locations:
            loc_serializer = StoreLocationSerializer(loc)
            loc_data = loc_serializer.data
            loc_data['distance_km'] = round(loc.distance, 2)
            location_data.append(loc_data)
        
        return Response({
            'success': True,
            'data': location_data
        })


class ContactSettingsView(APIView):
    """View for public contact settings."""
    
    permission_classes = [AllowAny]
    
    def get(self, request):
        """Get public contact settings."""
        settings = ContactSettings.get_settings()
        serializer = ContactSettingsPublicSerializer(settings)
        return Response({
            'success': True,
            'data': serializer.data
        })


class CustomizationRequestView(APIView):
    """View for creating customization requests."""
    
    permission_classes = [AllowAny]
    
    def post(self, request):
        """Create a new customization request."""
        serializer = CustomizationRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid data',
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get client info
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip_address = x_forwarded_for.split(',')[0].strip()
        else:
            ip_address = request.META.get('REMOTE_ADDR')
        
        user_agent = request.META.get('HTTP_USER_AGENT', '')
        
        customization_request = serializer.save(
            user=request.user if request.user.is_authenticated else None,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Here you could add a call to a service to send a notification email
        # to the admin, similar to ContactInquiryService.
        
        return Response({
            'success': True,
            'message': 'Your customization request has been submitted successfully!',
            'data': CustomizationRequestSerializer(customization_request).data
        }, status=status.HTTP_201_CREATED)
