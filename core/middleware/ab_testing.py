"""
A/B Testing Middleware and Utilities
"""
import random
import hashlib
from typing import Dict, Optional, List
from django.conf import settings
from django.core.cache import cache


class ABTestMiddleware:
    """
    Middleware for A/B testing.
    Assigns users to test variants and tracks them.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.cookie_name = getattr(settings, 'AB_TESTING_COOKIE_NAME', 'bunoraa_ab')
        self.cookie_age = getattr(settings, 'AB_TESTING_COOKIE_AGE', 86400 * 30)
    
    def __call__(self, request):
        if not getattr(settings, 'AB_TESTING_ENABLED', True):
            return self.get_response(request)
        
        # Get or create AB test assignments
        ab_cookie = request.COOKIES.get(self.cookie_name, '')
        assignments = self._parse_assignments(ab_cookie)
        
        # Check active tests and assign variants
        active_tests = self._get_active_tests()
        new_assignments = False
        
        for test in active_tests:
            if test['name'] not in assignments:
                variant = self._assign_variant(request, test)
                assignments[test['name']] = variant
                new_assignments = True
        
        # Store assignments in request
        request.ab_tests = assignments
        
        # Get response
        response = self.get_response(request)
        
        # Set cookie if new assignments
        if new_assignments:
            response.set_cookie(
                self.cookie_name,
                self._serialize_assignments(assignments),
                max_age=self.cookie_age,
                httponly=True,
                samesite='Lax',
                secure=not settings.DEBUG
            )
        
        return response
    
    def _get_active_tests(self) -> List[Dict]:
        """Get active A/B tests from cache or database."""
        cache_key = 'ab_tests_active'
        tests = cache.get(cache_key)
        
        if tests is None:
            try:
                from apps.analytics.models import ABTest
                tests = list(ABTest.objects.filter(
                    is_active=True
                ).values('name', 'variants', 'weights'))
            except Exception:
                # Fallback to default tests
                tests = [
                    {
                        'name': 'homepage_hero',
                        'variants': ['control', 'variant_a', 'variant_b'],
                        'weights': [0.34, 0.33, 0.33],
                    },
                    {
                        'name': 'checkout_flow',
                        'variants': ['control', 'simplified'],
                        'weights': [0.5, 0.5],
                    },
                ]
            
            cache.set(cache_key, tests, timeout=300)
        
        return tests
    
    def _assign_variant(self, request, test: Dict) -> str:
        """Assign a variant to the user."""
        variants = test.get('variants', ['control', 'variant'])
        weights = test.get('weights')
        
        if weights and len(weights) == len(variants):
            # Weighted random selection
            return random.choices(variants, weights=weights)[0]
        else:
            # Equal distribution
            return random.choice(variants)
    
    def _parse_assignments(self, cookie: str) -> Dict[str, str]:
        """Parse cookie value into assignments dict."""
        if not cookie:
            return {}
        
        try:
            assignments = {}
            for pair in cookie.split('|'):
                if ':' in pair:
                    name, variant = pair.split(':', 1)
                    assignments[name] = variant
            return assignments
        except Exception:
            return {}
    
    def _serialize_assignments(self, assignments: Dict[str, str]) -> str:
        """Serialize assignments dict to cookie value."""
        return '|'.join(f'{k}:{v}' for k, v in assignments.items())


class ABTestContextProcessor:
    """Context processor to add A/B test data to templates."""
    
    def __call__(self, request):
        return {
            'ab_tests': getattr(request, 'ab_tests', {}),
        }


def ab_test_context(request):
    """Context processor function for A/B tests."""
    return {
        'ab_tests': getattr(request, 'ab_tests', {}),
    }


def get_variant(request, test_name: str, default: str = 'control') -> str:
    """
    Get the variant for a specific test.
    
    Usage in views:
        variant = get_variant(request, 'homepage_hero')
        if variant == 'variant_a':
            # Show variant A
    """
    ab_tests = getattr(request, 'ab_tests', {})
    return ab_tests.get(test_name, default)


def track_conversion(request, test_name: str, conversion_type: str = 'goal'):
    """
    Track a conversion for A/B test.
    
    Usage:
        track_conversion(request, 'checkout_flow', 'purchase')
    """
    try:
        from apps.analytics.models import ABTestConversion
        
        ab_tests = getattr(request, 'ab_tests', {})
        variant = ab_tests.get(test_name)
        
        if variant:
            ABTestConversion.objects.create(
                test_name=test_name,
                variant=variant,
                conversion_type=conversion_type,
                user=request.user if request.user.is_authenticated else None,
                session_key=request.session.session_key,
            )
    except Exception:
        pass  # Don't break the flow if tracking fails
