"""
ML API Views

Django REST Framework views for ML services.
Includes tracking, recommendations, search, and predictions.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings

try:
    from rest_framework.views import APIView
    from rest_framework.response import Response
    from rest_framework import status
    from rest_framework.permissions import IsAuthenticated, AllowAny
    from rest_framework.decorators import api_view, permission_classes
    DRF_AVAILABLE = True
except ImportError:
    DRF_AVAILABLE = False
    # Mock classes
    class APIView:
        pass
    class Response:
        pass
    class IsAuthenticated:
        pass
    class AllowAny:
        pass

logger = logging.getLogger("bunoraa.ml.api")


# =============================================================================
# TRACKING APIs (Django Views for frontend tracking)
# =============================================================================

@method_decorator(csrf_exempt, name='dispatch')
class MLTrackingAPIView(View):
    """
    Main endpoint for receiving tracking events from the frontend.
    
    POST /api/ml/track/
    
    Receives batched events from the JavaScript tracking library.
    """
    
    def post(self, request, *args, **kwargs):
        """Handle tracking event batch."""
        try:
            # Parse request body
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
            
            events = data.get('events', [])
            meta = data.get('meta', {})
            
            if not events:
                return JsonResponse({'status': 'ok', 'processed': 0})
            
            # Add server-side metadata
            server_meta = {
                'received_at': datetime.now().isoformat(),
                'ip_address': self._get_client_ip(request),
                'user_agent': request.META.get('HTTP_USER_AGENT', ''),
                'user_id': request.user.id if request.user.is_authenticated else None,
            }
            
            # Process events
            processed_count = self._process_events(events, meta, server_meta)
            
            return JsonResponse({
                'status': 'ok',
                'processed': processed_count,
                'batch_id': meta.get('batch_id'),
            })
            
        except Exception as e:
            logger.exception("Error processing tracking events")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    def _get_client_ip(self, request):
        """Extract client IP from request."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', '')
    
    def _process_events(self, events, meta, server_meta):
        """Process and queue events for ML training."""
        try:
            from ..data_collection.collector import DataCollector
            from ..data_collection.events import EventTracker, EventType
            
            collector = DataCollector()
            tracker = EventTracker()
            
            processed = 0
            
            for event in events:
                try:
                    event_type = event.get('event_type', 'unknown')
                    
                    # Add server metadata
                    event['server_meta'] = server_meta
                    
                    # Route to appropriate handler
                    if event_type == 'page_view':
                        self._handle_page_view(event, tracker)
                    elif event_type == 'product_view':
                        self._handle_product_view(event, tracker)
                    elif event_type == 'product_click':
                        self._handle_product_click(event, tracker)
                    elif event_type == 'add_to_cart':
                        self._handle_add_to_cart(event, tracker)
                    elif event_type == 'remove_from_cart':
                        self._handle_remove_from_cart(event, tracker)
                    elif event_type == 'add_to_wishlist':
                        self._handle_wishlist(event, tracker)
                    elif event_type == 'search':
                        self._handle_search(event, tracker)
                    elif event_type == 'checkout':
                        self._handle_checkout(event, tracker)
                    elif event_type == 'purchase':
                        self._handle_purchase(event, tracker)
                    elif event_type == 'page_exit':
                        self._handle_page_exit(event, tracker)
                    elif event_type == 'heartbeat':
                        self._handle_heartbeat(event, tracker)
                    elif event_type == 'click':
                        self._handle_click(event, tracker)
                    else:
                        self._handle_generic_event(event, tracker)
                    
                    processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing event {event.get('event_type')}: {e}")
            
            return processed
            
        except ImportError as e:
            logger.warning(f"ML tracking modules not available: {e}")
            self._store_raw_events(events, meta, server_meta)
            return len(events)
    
    def _store_raw_events(self, events, meta, server_meta):
        """Store raw events to Redis when collectors are not available."""
        try:
            import redis
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            r = redis.from_url(redis_url)
            
            for event in events:
                event['server_meta'] = server_meta
                r.lpush('ml:raw_events', json.dumps(event))
            
            r.ltrim('ml:raw_events', 0, 99999)
            
        except Exception as e:
            logger.error(f"Error storing raw events: {e}")
    
    def _handle_page_view(self, event, tracker):
        from ..data_collection.events import EventType
        tracker.track(
            event_type=EventType.PAGE_VIEW,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={
                'page_url': event.get('page_url'),
                'page_type': event.get('page_data', {}).get('type', 'other'),
                'referrer': event.get('referrer'),
                'utm': event.get('utm'),
            }
        )
    
    def _handle_product_view(self, event, tracker):
        from ..data_collection.events import EventType
        tracker.track(
            event_type=EventType.PRODUCT_VIEW,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={
                'product_id': event.get('product_id'),
                'product_name': event.get('product_name'),
                'category': event.get('product_category'),
                'price': event.get('product_price'),
                'source_page': event.get('source_page'),
                'position': event.get('position'),
            }
        )
    
    def _handle_product_click(self, event, tracker):
        from ..data_collection.events import EventType
        tracker.track(
            event_type=EventType.PRODUCT_CLICK,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={
                'product_id': event.get('product_id'),
                'interaction_type': 'click',
                'element_type': event.get('element_type'),
                'position': event.get('position'),
            }
        )
    
    def _handle_add_to_cart(self, event, tracker):
        from ..data_collection.events import EventType
        tracker.track(
            event_type=EventType.ADD_TO_CART,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={
                'product_id': event.get('product_id'),
                'action': 'add',
                'quantity': event.get('quantity', 1),
                'variant': event.get('variant'),
                'cart_value': event.get('cart_value'),
            }
        )
    
    def _handle_remove_from_cart(self, event, tracker):
        from ..data_collection.events import EventType
        tracker.track(
            event_type=EventType.REMOVE_FROM_CART,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={
                'product_id': event.get('product_id'),
                'action': 'remove',
                'quantity': event.get('quantity', 1),
            }
        )
    
    def _handle_wishlist(self, event, tracker):
        from ..data_collection.events import EventType
        tracker.track(
            event_type=EventType.ADD_TO_WISHLIST,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={
                'product_id': event.get('product_id'),
                'action': 'add',
            }
        )
    
    def _handle_search(self, event, tracker):
        from ..data_collection.events import EventType
        results_count = event.get('results_count', 0)
        event_type = EventType.SEARCH if results_count > 0 else EventType.SEARCH_NO_RESULTS
        tracker.track(
            event_type=event_type,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={
                'query': event.get('query', ''),
                'results_count': results_count,
                'filters': event.get('filters'),
            }
        )
    
    def _handle_checkout(self, event, tracker):
        from ..data_collection.events import EventType
        step = event.get('step', 1)
        event_type = EventType.START_CHECKOUT if step == 1 else EventType.CHECKOUT_STEP
        tracker.track(
            event_type=event_type,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={
                'step': step,
                'step_name': event.get('step_name'),
                'cart_value': event.get('cart_value'),
                'items_count': event.get('items_count'),
            }
        )
    
    def _handle_purchase(self, event, tracker):
        from ..data_collection.events import EventType
        tracker.track(
            event_type=EventType.PURCHASE,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={'order_id': event.get('order_id'), 'order_value': event.get('order_value'),
                  'items_count': event.get('items_count'), 'coupon': event.get('coupon')},
        )
    
    def _handle_page_exit(self, event, tracker):
        from ..data_collection.events import EventType
        page_context = event.get('page_context', {})
        tracker.track(
            event_type=EventType.PAGE_EXIT,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={'time_on_page': event.get('time_on_page', page_context.get('time_on_page', 0)),
                  'active_time': event.get('active_time', page_context.get('active_time', 0)),
                  'scroll_depth': event.get('scroll_depth', page_context.get('scroll_depth', 0)),
                  'clicks': event.get('clicks', page_context.get('clicks', 0)),
                  'is_bounce': event.get('is_bounce', False)},
        )
    
    def _handle_heartbeat(self, event, tracker):
        from ..data_collection.events import EventType
        tracker.track(
            event_type=EventType.SESSION_HEARTBEAT,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={'time_on_page': event.get('time_on_page', 0), 'active_time': event.get('active_time', 0),
                  'scroll_depth': event.get('scroll_depth', 0), 'is_idle': event.get('is_idle', False)},
        )
    
    def _handle_click(self, event, tracker):
        from ..data_collection.events import EventType
        tracker.track(
            event_type=EventType.CLICK,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data={'element_type': event.get('element_type'), 'element_id': event.get('element_id'),
                  'href': event.get('href'), 'position': event.get('position')},
        )
    
    def _handle_generic_event(self, event, tracker):
        from ..data_collection.events import EventType
        event_type_str = event.get('event_type', 'unknown').upper().replace('-', '_').replace(' ', '_')
        try:
            event_type = EventType[event_type_str]
        except KeyError:
            event_type = EventType.CUSTOM
        tracker.track(
            event_type=event_type,
            user_id=event.get('user_id') or event.get('server_meta', {}).get('user_id'),
            session_id=event.get('session_id'),
            data=event,
        )


@method_decorator(csrf_exempt, name='dispatch')
class MLPredictionsAPIView(View):
    """
    Endpoint for ML predictions (demand, fraud, churn).
    
    POST /api/ml/predict/
    """
    
    def post(self, request, *args, **kwargs):
        """Get ML predictions."""
        try:
            data = json.loads(request.body)
            prediction_type = data.get('type', 'demand')
            
            from ..services.ml_service import MLService
            service = MLService()
            
            if prediction_type == 'demand':
                product_id = data.get('product_id')
                horizon = data.get('horizon', 30)
                prediction = service.predict_demand(product_id=product_id, horizon=horizon)
            elif prediction_type == 'fraud':
                order_data = data.get('order', {})
                prediction = service.predict_fraud(order_data)
            elif prediction_type == 'churn':
                user_id = data.get('user_id')
                prediction = service.predict_churn(user_id)
            else:
                return JsonResponse({'status': 'error', 'message': f'Unknown prediction type: {prediction_type}'}, status=400)
            
            return JsonResponse({'status': 'ok', 'type': prediction_type, 'prediction': prediction})
            
        except Exception as e:
            logger.exception("Error getting predictions")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


# =============================================================================
# RECOMMENDATION APIs
# =============================================================================

class PersonalizedRecommendationsView(APIView):
    """Get personalized recommendations for a user."""
    
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            from ..services.recommendation_service import RecommendationService
            
            user_id = request.user.id
            num_items = int(request.query_params.get("limit", 20))
            category_id = request.query_params.get("category_id")
            
            if category_id:
                category_id = int(category_id)
            
            service = RecommendationService()
            recommendations = service.get_personalized_recommendations(
                user_id=user_id,
                num_items=num_items,
                category_id=category_id,
                context={
                    "device": request.headers.get("User-Agent"),
                    "platform": request.query_params.get("platform", "web"),
                }
            )
            
            return Response({
                "success": True,
                "recommendations": recommendations,
                "count": len(recommendations),
            })
        except Exception as e:
            logger.exception("Error getting personalized recommendations")
            # Return empty recommendations with success=True to avoid frontend errors
            return Response({
                "success": True,
                "recommendations": [],
                "count": 0,
                "fallback": True,
                "message": "Personalized recommendations temporarily unavailable"
            })


class SimilarProductsView(APIView):
    """Get products similar to a given product."""
    
    permission_classes = [AllowAny]
    
    def get(self, request, product_id: int):
        from ..services.recommendation_service import RecommendationService
        
        num_items = int(request.query_params.get("limit", 10))
        similarity_type = request.query_params.get("type", "hybrid")
        
        service = RecommendationService()
        similar = service.get_similar_products(
            product_id=product_id,
            num_items=num_items,
            similarity_type=similarity_type
        )
        
        return Response({
            "success": True,
            "product_id": product_id,
            "similar_products": similar,
            "count": len(similar),
        })


class VisuallySimilarProductsView(APIView):
    """Get visually similar products to a given product."""
    
    permission_classes = [AllowAny]
    
    def get(self, request, product_id: int):
        from ..services.recommendation_service import RecommendationService
        
        num_items = int(request.query_params.get("limit", 10))
        
        service = RecommendationService()
        similar = service.get_visually_similar_products(
            product_id=product_id,
            num_items=num_items
        )
        
        return Response({
            "success": True,
            "product_id": product_id,
            "visually_similar_products": similar,
            "count": len(similar),
        })


class FrequentlyBoughtTogetherView(APIView):
    """Get products frequently bought together."""
    
    permission_classes = [AllowAny]
    
    def get(self, request, product_id: int):
        from ..services.recommendation_service import RecommendationService
        
        num_items = int(request.query_params.get("limit", 5))
        
        service = RecommendationService()
        fbt = service.get_frequently_bought_together(
            product_id=product_id,
            num_items=num_items
        )
        
        return Response({
            "success": True,
            "product_id": product_id,
            "frequently_bought_together": fbt,
        })


class CartRecommendationsView(APIView):
    """Get recommendations based on cart contents."""
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        from ..services.recommendation_service import RecommendationService
        
        user_id = request.user.id
        cart_product_ids = request.data.get("product_ids", [])
        num_items = int(request.data.get("limit", 5))
        
        service = RecommendationService()
        recommendations = service.get_cart_recommendations(
            user_id=user_id,
            cart_product_ids=cart_product_ids,
            num_items=num_items
        )
        
        return Response({
            "success": True,
            "recommendations": recommendations,
        })


class PopularProductsView(APIView):
    """Get popular/trending products."""
    
    permission_classes = [AllowAny]
    
    def get(self, request):
        from ..services.recommendation_service import RecommendationService
        
        num_items = int(request.query_params.get("limit", 20))
        category_id = request.query_params.get("category_id")
        time_window = int(request.query_params.get("days", 7))
        
        if category_id:
            category_id = int(category_id)
        
        service = RecommendationService()
        popular = service.get_popular_items(
            num_items=num_items,
            category_id=category_id,
            time_window_days=time_window
        )
        
        return Response({
            "success": True,
            "products": popular,
            "count": len(popular),
        })


# =============================================================================
# SEARCH APIs
# =============================================================================

class SemanticSearchView(APIView):
    """Semantic product search."""
    
    permission_classes = [AllowAny]
    
    def get(self, request):
        from ..services.search_service import SearchService
        
        query = request.query_params.get("q", "")
        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 20))
        sort_by = request.query_params.get("sort", "relevance")
        
        # Build filters
        filters = {}
        if request.query_params.get("category"):
            filters["category_id"] = int(request.query_params.get("category"))
        if request.query_params.get("min_price"):
            filters["min_price"] = float(request.query_params.get("min_price"))
        if request.query_params.get("max_price"):
            filters["max_price"] = float(request.query_params.get("max_price"))
        if request.query_params.get("in_stock") == "true":
            filters["in_stock"] = True
        
        user_id = None
        if request.user.is_authenticated:
            user_id = request.user.id
        
        service = SearchService()
        results = service.search(
            query=query,
            filters=filters if filters else None,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            user_id=user_id
        )
        
        return Response({
            "success": True,
            **results,
        })


class AutocompleteView(APIView):
    """Search autocomplete."""
    
    permission_classes = [AllowAny]
    
    def get(self, request):
        from ..services.search_service import SearchService
        
        query = request.query_params.get("q", "")
        num_suggestions = int(request.query_params.get("limit", 5))
        
        if len(query) < 2:
            return Response({"suggestions": []})
        
        service = SearchService()
        suggestions = service.autocomplete(query, num_suggestions)
        
        return Response({
            "success": True,
            "suggestions": suggestions,
        })


class VisualSearchView(APIView):
    """Search products by image."""
    
    permission_classes = [AllowAny]
    
    def post(self, request):
        from ..services.search_service import SearchService
        
        if "image" not in request.FILES:
            return Response(
                {"error": "No image provided"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        image_file = request.FILES["image"]
        image_data = image_file.read()
        
        num_results = int(request.data.get("limit", 20))
        
        service = SearchService()
        results = service.visual_search(image_data, num_results)
        
        return Response({
            "success": True,
            "products": results,
            "count": len(results),
        })


# =============================================================================
# PERSONALIZATION APIs
# =============================================================================

class PersonalizedHomepageView(APIView):
    """Get personalized homepage content."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        from ..services.personalization_service import PersonalizationService
        
        user_id = request.user.id
        
        service = PersonalizationService()
        homepage = service.personalize_homepage(
            user_id=user_id,
            context={
                "device": request.headers.get("User-Agent"),
            }
        )
        
        return Response({
            "success": True,
            **homepage,
        })


class UserProfileView(APIView):
    """Get user personalization profile."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        from ..services.personalization_service import PersonalizationService
        
        user_id = request.user.id
        
        service = PersonalizationService()
        profile = service.get_user_profile(user_id)
        
        return Response({
            "success": True,
            "profile": profile,
        })


class NextBestActionView(APIView):
    """Get next best action for user."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        from ..services.personalization_service import PersonalizationService
        
        user_id = request.user.id
        
        service = PersonalizationService()
        action = service.get_next_best_action(user_id)
        
        return Response({
            "success": True,
            "action": action,
        })


# =============================================================================
# ANALYTICS APIs
# =============================================================================

class DemandForecastView(APIView):
    """Get demand forecasts for products."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        from ..services.analytics_service import AnalyticsService
        
        product_ids = request.query_params.getlist("product_ids")
        if product_ids:
            product_ids = [int(pid) for pid in product_ids]
        else:
            product_ids = None
        
        horizon = int(request.query_params.get("days", 14))
        
        service = AnalyticsService()
        forecasts = service.get_demand_forecast(
            product_ids=product_ids,
            horizon_days=horizon
        )
        
        return Response({
            "success": True,
            "forecasts": forecasts,
            "horizon_days": horizon,
        })


class PriceRecommendationsView(APIView):
    """Get price optimization recommendations."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        from ..services.analytics_service import AnalyticsService
        
        product_ids = request.query_params.getlist("product_ids")
        if product_ids:
            product_ids = [int(pid) for pid in product_ids]
        else:
            product_ids = None
        
        goal = request.query_params.get("goal", "revenue")
        
        service = AnalyticsService()
        recommendations = service.get_price_recommendations(
            product_ids=product_ids,
            optimization_goal=goal
        )
        
        return Response({
            "success": True,
            "recommendations": recommendations,
        })


class CustomerSegmentsView(APIView):
    """Get customer segments."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        from ..services.analytics_service import AnalyticsService
        
        num_segments = int(request.query_params.get("segments", 5))
        
        service = AnalyticsService()
        segments = service.segment_customers(num_segments)
        
        return Response({
            "success": True,
            "segments": segments,
        })


class ProductInsightsView(APIView):
    """Get ML-powered product insights."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request, product_id: int):
        from ..services.analytics_service import AnalyticsService
        
        service = AnalyticsService()
        insights = service.get_product_insights(product_id)
        
        return Response({
            "success": True,
            "product_id": product_id,
            "insights": insights,
        })


class AnalyticsDashboardView(APIView):
    """Get analytics dashboard metrics."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        from ..services.analytics_service import AnalyticsService
        
        service = AnalyticsService()
        metrics = service.get_dashboard_metrics()
        
        return Response({
            "success": True,
            "metrics": metrics,
        })


# =============================================================================
# FRAUD APIs
# =============================================================================

class OrderRiskAssessmentView(APIView):
    """Assess fraud risk for an order."""
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        from ..services.fraud_service import FraudService
        
        order_data = request.data.get("order", {})
        user_data = request.data.get("user", {})
        
        # Add request context
        order_data["ip_address"] = request.META.get("REMOTE_ADDR")
        order_data["device_fingerprint"] = request.data.get("device_fingerprint")
        
        service = FraudService()
        risk = service.assess_order_risk(order_data, user_data)
        
        return Response({
            "success": True,
            "risk_score": risk.score,
            "risk_level": risk.level,
            "factors": risk.factors,
            "recommendation": risk.recommendation,
            "is_blocked": risk.is_blocked,
            "needs_review": risk.needs_review,
        })


class UserRiskAssessmentView(APIView):
    """Assess fraud risk for a user."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request, user_id: int):
        from ..services.fraud_service import FraudService
        
        service = FraudService()
        risk = service.assess_user_risk(user_id)
        
        return Response({
            "success": True,
            "user_id": user_id,
            "risk_score": risk.score,
            "risk_level": risk.level,
            "factors": risk.factors,
        })


class FraudDashboardView(APIView):
    """Get fraud detection dashboard data."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        from ..services.fraud_service import FraudService
        
        service = FraudService()
        data = service.get_fraud_dashboard_data()
        
        return Response({
            "success": True,
            "data": data,
        })


# =============================================================================
# CHURN APIs
# =============================================================================

class ChurnPredictionView(APIView):
    """Predict churn risk for a user."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request, user_id: int):
        from ..services.personalization_service import PersonalizationService
        
        service = PersonalizationService()
        prediction = service.predict_churn_risk(user_id)
        
        return Response({
            "success": True,
            **prediction,
        })


# =============================================================================
# TRAINING APIs (Admin only)
# =============================================================================

class TriggerTrainingView(APIView):
    """Trigger model training."""
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        # Check admin permission
        if not request.user.is_staff:
            return Response(
                {"error": "Admin access required"},
                status=status.HTTP_403_FORBIDDEN
            )
        
        model_type = request.data.get("model_type", "all")
        
        from ..training.tasks import (
            train_recommendation_model,
            train_embedding_models,
            train_demand_forecaster,
            train_fraud_detector,
            train_churn_predictor,
            train_search_model,
        )
        
        tasks = []
        
        if model_type in ("all", "recommendation"):
            tasks.append(train_recommendation_model.delay("ncf"))
        
        if model_type in ("all", "embeddings"):
            tasks.append(train_embedding_models.delay())
        
        if model_type in ("all", "forecasting"):
            tasks.append(train_demand_forecaster.delay())
        
        if model_type in ("all", "fraud"):
            tasks.append(train_fraud_detector.delay())
        
        if model_type in ("all", "churn"):
            tasks.append(train_churn_predictor.delay())
        
        if model_type in ("all", "search"):
            tasks.append(train_search_model.delay())
        
        return Response({
            "success": True,
            "message": f"Training triggered for {model_type}",
            "tasks_queued": len(tasks),
        })


class ModelHealthView(APIView):
    """Check health of deployed models."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        from ..training.tasks import model_health_check
        
        # Run sync for immediate response
        try:
            health = model_health_check()
            return Response({
                "success": True,
                "health": health,
            })
        except Exception as e:
            return Response({
                "success": False,
                "error": str(e),
            })
