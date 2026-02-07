"""
Internationalization API Views

DRF ViewSets and Views for i18n operations.
"""
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from decimal import Decimal

from ..models import (
    Language, Currency, ExchangeRate, ExchangeRateHistory,
    Timezone, Country, Division, District, Upazila,
    TranslationNamespace, TranslationKey, Translation, ContentTranslation,
    UserLocalePreference
)
from ..services import (
    LanguageService, CurrencyService, ExchangeRateService,
    TimezoneService, GeoService, CurrencyConversionService,
    UserPreferenceService
)
from .serializers import (
    LanguageSerializer, LanguageListSerializer,
    CurrencySerializer, CurrencyListSerializer,
    ExchangeRateSerializer, ExchangeRateHistorySerializer,
    TimezoneSerializer, TimezoneListSerializer,
    CountrySerializer, CountryListSerializer,
    DivisionSerializer, DivisionListSerializer,
    DistrictSerializer, DistrictListSerializer,
    UpazilaSerializer,
    TranslationNamespaceSerializer, TranslationKeySerializer,
    TranslationSerializer, ContentTranslationSerializer,
    UserLocalePreferenceSerializer, UserLocalePreferenceUpdateSerializer,
    CurrencyConversionRequestSerializer, CurrencyConversionResponseSerializer
)


# =============================================================================
# Language ViewSet
# =============================================================================

class LanguageViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Language model."""
    
    queryset = Language.objects.filter(is_active=True)
    permission_classes = [permissions.AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return LanguageListSerializer
        return LanguageSerializer
    
    @action(detail=False, methods=['get'])
    def default(self, request):
        """Get the default language."""
        language = LanguageService.get_default_language()
        if language:
            return Response(LanguageSerializer(language).data)
        return Response({'error': 'No default language found'}, status=404)
    
    @action(detail=False, methods=['get'])
    def detect(self, request):
        """Detect language from request."""
        language = LanguageService.detect_language(request)
        if language:
            return Response(LanguageSerializer(language).data)
        return Response({'detected': False})
    
    @action(detail=True, methods=['post'])
    def set(self, request, pk=None):
        """Set language preference."""
        language = self.get_object()
        
        # Set in session
        request.session['language'] = language.code
        request.session['django_language'] = language.code
        
        # Update user preference if authenticated
        if request.user.is_authenticated:
            LanguageService.set_user_language(request.user, language.code)
        
        return Response({'success': True, 'code': language.code})


# =============================================================================
# Currency ViewSet
# =============================================================================

class CurrencyViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Currency model."""
    
    queryset = Currency.objects.filter(is_active=True)
    permission_classes = [permissions.AllowAny]
    lookup_field = 'code'
    
    def get_serializer_class(self):
        if self.action == 'list':
            return CurrencyListSerializer
        return CurrencySerializer
    
    @action(detail=False, methods=['get'])
    def default(self, request):
        """Get the default currency."""
        currency = CurrencyService.get_default_currency()
        if currency:
            return Response(CurrencySerializer(currency).data)
        return Response({'error': 'No default currency found'}, status=404)
    
    @action(detail=False, methods=['get'])
    def detect(self, request):
        """Detect currency from request."""
        currency = CurrencyService.detect_currency(request)
        if currency:
            return Response(CurrencySerializer(currency).data)
        return Response({'detected': False})
    
    @action(detail=True, methods=['post'])
    def set(self, request, code=None):
        """Set currency preference."""
        currency = self.get_object()
        
        # Set in session
        request.session['currency_code'] = currency.code
        
        # Update user preference if authenticated
        if request.user.is_authenticated:
            CurrencyService.set_user_currency(request.user, currency.code)
        
        return Response({'success': True, 'code': currency.code})


# =============================================================================
# Exchange Rate ViewSet
# =============================================================================

class ExchangeRateViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for ExchangeRate model."""
    
    queryset = ExchangeRate.objects.filter(is_active=True)
    serializer_class = ExchangeRateSerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        from_code = self.request.query_params.get('from')
        to_code = self.request.query_params.get('to')
        
        if from_code:
            queryset = queryset.filter(from_currency__code=from_code.upper())
        if to_code:
            queryset = queryset.filter(to_currency__code=to_code.upper())
        
        return queryset.select_related('from_currency', 'to_currency')
    
    @action(detail=False, methods=['get'])
    def history(self, request):
        """Get exchange rate history."""
        from_code = request.query_params.get('from')
        to_code = request.query_params.get('to')
        days = int(request.query_params.get('days', 30))
        
        if not from_code or not to_code:
            return Response(
                {'error': 'Both from and to currency codes are required'},
                status=400
            )
        
        from_currency = CurrencyService.get_currency_by_code(from_code)
        to_currency = CurrencyService.get_currency_by_code(to_code)
        
        if not from_currency or not to_currency:
            return Response({'error': 'Invalid currency code'}, status=400)
        
        history = ExchangeRateService.get_rate_history(from_currency, to_currency, days)
        serializer = ExchangeRateHistorySerializer(history, many=True)
        
        return Response({
            'from': from_code,
            'to': to_code,
            'days': days,
            'history': serializer.data
        })


# =============================================================================
# Currency Conversion View
# =============================================================================

class CurrencyConversionView(APIView):
    """API view for currency conversion."""
    
    permission_classes = [permissions.AllowAny]
    
    def get(self, request):
        """Convert currency (GET with query params)."""
        try:
            amount = Decimal(request.query_params.get('amount', '0'))
            from_code = request.query_params.get('from', '').upper()
            to_code = request.query_params.get('to', '').upper()
            
            return self._convert(amount, from_code, to_code)
            
        except Exception as e:
            return Response({'error': str(e)}, status=400)
    
    def post(self, request):
        """Convert currency (POST with body)."""
        serializer = CurrencyConversionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        return self._convert(
            serializer.validated_data['amount'],
            serializer.validated_data['from_currency'].upper(),
            serializer.validated_data['to_currency'].upper(),
            serializer.validated_data.get('round_result', True)
        )
    
    def _convert(self, amount, from_code, to_code, round_result=True):
        """Perform the conversion."""
        if not from_code or not to_code:
            return Response(
                {'error': 'Both from and to currency codes are required'},
                status=400
            )
        
        from_currency = CurrencyService.get_currency_by_code(from_code)
        to_currency = CurrencyService.get_currency_by_code(to_code)
        
        if not from_currency:
            return Response({'error': f'Currency not found: {from_code}'}, status=400)
        if not to_currency:
            return Response({'error': f'Currency not found: {to_code}'}, status=400)
        
        try:
            rate = ExchangeRateService.get_exchange_rate(from_currency, to_currency)
            if rate is None:
                return Response(
                    {'error': f'No exchange rate available for {from_code} to {to_code}'},
                    status=400
                )
            
            converted = CurrencyConversionService.convert(
                amount, from_currency, to_currency, round_result
            )
            
            return Response({
                'original_amount': str(amount),
                'original_currency': from_code,
                'original_formatted': from_currency.format_amount(amount),
                'converted_amount': str(converted),
                'converted_currency': to_code,
                'converted_formatted': to_currency.format_amount(converted),
                'exchange_rate': str(rate)
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=400)


# =============================================================================
# Timezone ViewSet
# =============================================================================

class TimezoneViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Timezone model."""
    
    queryset = Timezone.objects.filter(is_active=True)
    permission_classes = [permissions.AllowAny]
    lookup_field = 'name'
    
    def get_serializer_class(self):
        if self.action == 'list':
            return TimezoneListSerializer
        return TimezoneSerializer
    
    @action(detail=False, methods=['get'])
    def common(self, request):
        """Get common timezones."""
        timezones = TimezoneService.get_common_timezones()
        serializer = TimezoneListSerializer(timezones, many=True)
        return Response(serializer.data)


# =============================================================================
# Country ViewSet
# =============================================================================

class CountryViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Country model."""
    
    queryset = Country.objects.filter(is_active=True)
    permission_classes = [permissions.AllowAny]
    lookup_field = 'code'
    
    def get_serializer_class(self):
        if self.action == 'list':
            return CountryListSerializer
        return CountrySerializer
    
    @action(detail=False, methods=['get'])
    def shipping(self, request):
        """Get countries where shipping is available."""
        countries = GeoService.get_shipping_countries()
        serializer = CountryListSerializer(countries, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def detect(self, request):
        """Detect country from IP."""
        country = GeoService.detect_country(request)
        if country:
            return Response(CountrySerializer(country).data)
        return Response({'detected': False})


# =============================================================================
# Geographic ViewSets
# =============================================================================

class DivisionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Division model."""
    
    queryset = Division.objects.filter(is_active=True)
    permission_classes = [permissions.AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return DivisionListSerializer
        return DivisionSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        country_code = self.request.query_params.get('country')
        if country_code:
            queryset = queryset.filter(country__code=country_code.upper())
        
        return queryset.select_related('country').order_by('sort_order', 'name')


class DistrictViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for District model."""
    
    queryset = District.objects.filter(is_active=True)
    permission_classes = [permissions.AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return DistrictListSerializer
        return DistrictSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        division_id = self.request.query_params.get('division')
        if division_id:
            queryset = queryset.filter(division_id=division_id)
        
        return queryset.select_related('division').order_by('sort_order', 'name')


class UpazilaViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Upazila model."""
    
    queryset = Upazila.objects.filter(is_active=True)
    serializer_class = UpazilaSerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        district_id = self.request.query_params.get('district')
        if district_id:
            queryset = queryset.filter(district_id=district_id)
        
        return queryset.select_related('district').order_by('sort_order', 'name')


# =============================================================================
# Translation ViewSets
# =============================================================================

class TranslationNamespaceViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for TranslationNamespace model."""
    
    queryset = TranslationNamespace.objects.all()
    serializer_class = TranslationNamespaceSerializer
    permission_classes = [permissions.AllowAny]
    lookup_field = 'name'


class TranslationKeyViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for TranslationKey model."""
    
    queryset = TranslationKey.objects.all()
    serializer_class = TranslationKeySerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        namespace = self.request.query_params.get('namespace')
        if namespace:
            queryset = queryset.filter(namespace__name=namespace)
        
        return queryset.select_related('namespace')


class TranslationViewSet(viewsets.ModelViewSet):
    """ViewSet for Translation model."""
    
    queryset = Translation.objects.all()
    serializer_class = TranslationSerializer
    
    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            return [permissions.AllowAny()]
        return [permissions.IsAuthenticated()]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        language = self.request.query_params.get('language')
        namespace = self.request.query_params.get('namespace')
        status_filter = self.request.query_params.get('status')
        
        if language:
            queryset = queryset.filter(language__code=language)
        if namespace:
            queryset = queryset.filter(key__namespace__name=namespace)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        return queryset.select_related('key', 'language', 'translated_by')
    
    @action(detail=False, methods=['get'])
    def for_namespace(self, request):
        """Get all translations for a namespace and language."""
        namespace = request.query_params.get('namespace')
        language = request.query_params.get('language')
        
        if not namespace or not language:
            return Response(
                {'error': 'Both namespace and language are required'},
                status=400
            )
        
        translations = Translation.objects.filter(
            key__namespace__name=namespace,
            language__code=language,
            status='approved'
        ).select_related('key').values_list('key__key', 'translated_text')
        
        return Response({
            'namespace': namespace,
            'language': language,
            'translations': dict(translations)
        })


class ContentTranslationViewSet(viewsets.ModelViewSet):
    """ViewSet for ContentTranslation model."""
    
    queryset = ContentTranslation.objects.all()
    serializer_class = ContentTranslationSerializer
    
    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            return [permissions.AllowAny()]
        return [permissions.IsAuthenticated()]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        content_type = self.request.query_params.get('content_type')
        content_id = self.request.query_params.get('content_id')
        language = self.request.query_params.get('language')
        
        if content_type:
            queryset = queryset.filter(content_type=content_type)
        if content_id:
            queryset = queryset.filter(content_id=content_id)
        if language:
            queryset = queryset.filter(language__code=language)
        
        return queryset.select_related('language', 'translated_by')


# =============================================================================
# User Locale Preference View
# =============================================================================

class UserLocalePreferenceView(APIView):
    """API view for user locale preferences."""
    
    permission_classes = [permissions.AllowAny]
    
    def get(self, request):
        """Get current user's locale preferences."""
        if request.user.is_authenticated:
            pref = UserPreferenceService.get_or_create_preference(request.user)
            serializer = UserLocalePreferenceSerializer(pref)
            return Response(serializer.data)
        else:
            # Return session-based preferences for anonymous users
            return Response({
                'language': request.session.get('language'),
                'language_name': request.session.get('language'),
                'currency_code': request.session.get('currency_code'),
                'currency': None,
                'timezone': request.session.get('timezone'),
                'timezone_name': request.session.get('timezone'),
                'auto_detect_language': request.session.get('auto_detect_language', True),
                'auto_detect_currency': request.session.get('auto_detect_currency', True),
            })
    
    def post(self, request):
        """Create or update user's locale preferences (alias for PUT for convenience)."""
        return self.put(request)
    
    def put(self, request):
        """Update current user's locale preferences."""
        serializer = UserLocalePreferenceUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        validated_data = serializer.validated_data.copy()
        
        # Handle auto_detect alias -> auto_detect_currency
        if 'auto_detect' in validated_data:
            auto_detect_value = validated_data.pop('auto_detect')
            # Only set auto_detect_currency if not explicitly provided
            if 'auto_detect_currency' not in validated_data:
                validated_data['auto_detect_currency'] = auto_detect_value
        
        # When setting currency_code manually, disable auto-detect by default
        if 'currency_code' in validated_data and validated_data['currency_code']:
            if 'auto_detect_currency' not in validated_data:
                validated_data['auto_detect_currency'] = False
        
        response = Response()
        
        # Update based on authentication status
        if request.user.is_authenticated:
            pref = UserPreferenceService.update_preference(
                request.user, **validated_data
            )
            
            # Update session
            if pref.language:
                request.session['language'] = pref.language.code
                request.session['django_language'] = pref.language.code
            if pref.currency:
                request.session['currency_code'] = pref.currency.code
            if pref.timezone:
                request.session['timezone'] = pref.timezone.name
            
            # Return success with preference data
            response_data = UserLocalePreferenceSerializer(pref).data
            response_data['success'] = True
            response.data = response_data
            
            # Also set cookie for currency (persists across sessions)
            if pref.currency:
                response.set_cookie(
                    'currency',
                    pref.currency.code,
                    max_age=365 * 24 * 60 * 60,  # 1 year
                    httponly=False,
                    samesite='Lax'
                )
        else:
            # For anonymous users, only set in session and cookie
            response_data = {'success': True}
            
            if 'language_code' in validated_data or 'language' in validated_data:
                lang_code = validated_data.get('language_code') or validated_data.get('language')
                if lang_code:
                    request.session['language'] = lang_code
                    request.session['django_language'] = lang_code
                    response_data['language'] = lang_code
            
            if 'currency_code' in validated_data or 'currency' in validated_data:
                curr_code = validated_data.get('currency_code') or validated_data.get('currency')
                if curr_code:
                    request.session['currency_code'] = curr_code
                    response_data['currency_code'] = curr_code
                    # Set cookie for currency persistence
                    response.set_cookie(
                        'currency',
                        curr_code,
                        max_age=365 * 24 * 60 * 60,  # 1 year
                        httponly=False,
                        samesite='Lax'
                    )
            
            if 'timezone' in validated_data:
                tz = validated_data.get('timezone')
                if tz:
                    request.session['timezone'] = tz
                    response_data['timezone'] = tz
            
            response.data = response_data
        
        return response
    
    def patch(self, request):
        """Partial update of user's locale preferences."""
        return self.put(request)
