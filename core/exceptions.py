"""
Custom exception handler for consistent API responses
"""
from rest_framework.views import exception_handler
from rest_framework.exceptions import (
    APIException,
    ValidationError,
    NotFound,
    PermissionDenied,
    AuthenticationFailed,
    NotAuthenticated,
    Throttled,
)
from rest_framework import status
from rest_framework.authentication import SessionAuthentication
from django.http import Http404
from django.core.exceptions import ObjectDoesNotExist, PermissionDenied as DjangoPermissionDenied
import logging

logger = logging.getLogger('bunoraa')


def custom_exception_handler(exc, context):
    """
    Custom exception handler that returns consistent API response format:
    {
        "success": false,
        "message": "Error message",
        "data": null,
        "meta": {"errors": [...]}
    }
    """
    # Call REST framework's default exception handler first
    response = exception_handler(exc, context)
    
    if response is not None:
        custom_response_data = {
            'success': False,
            'message': get_error_message(exc),
            'data': None,
            'meta': {
                'errors': get_error_details(exc, response.data),
                'status_code': response.status_code,
            }
        }
        response.data = custom_response_data
        
        # Log the error
        logger.error(f"API Error: {exc.__class__.__name__} - {str(exc)}", exc_info=True)
    
    return response


# CSRF failure handler: return structured JSON and set a fresh CSRF cookie so clients can retry
from django.middleware.csrf import get_token
from django.http import JsonResponse


def csrf_failure(request, reason=""):
    try:
        user = getattr(request, 'user', None)
        user_info = f'user_id={user.id}' if getattr(user, 'is_authenticated', False) else 'anonymous'
    except Exception:
        user_info = 'unknown'
    logger.warning('CSRF failure: %s (%s) for path %s', reason, user_info, getattr(request, 'path', ''))

    token = get_token(request)
    resp = JsonResponse({
        'success': False,
        'message': 'CSRF validation failed',
        'data': None,
        'meta': {
            'reason': str(reason),
            'new_csrf_token': token
        }
    }, status=403)
    # ensure cookie is set so client can pick it up
    resp.set_cookie('csrftoken', token, samesite='Lax', httponly=False)
    return resp


def get_error_message(exc):
    """Get a user-friendly error message based on exception type."""
    if isinstance(exc, ValidationError):
        return "Validation failed. Please check your input."
    elif isinstance(exc, NotFound) or isinstance(exc, Http404):
        return "The requested resource was not found."
    elif isinstance(exc, PermissionDenied) or isinstance(exc, DjangoPermissionDenied):
        return "You do not have permission to perform this action."
    elif isinstance(exc, NotAuthenticated):
        return "Authentication credentials were not provided."
    elif isinstance(exc, AuthenticationFailed):
        return "Invalid authentication credentials."
    elif isinstance(exc, Throttled):
        return f"Request was throttled. Try again in {exc.wait} seconds."
    elif hasattr(exc, 'detail'):
        if isinstance(exc.detail, str):
            return exc.detail
        elif isinstance(exc.detail, list):
            return exc.detail[0] if exc.detail else "An error occurred."
        elif isinstance(exc.detail, dict):
            return next(iter(exc.detail.values()), ["An error occurred."])[0]
    return "An unexpected error occurred."


def get_error_details(exc, response_data):
    """Extract detailed error information."""
    if isinstance(exc, ValidationError):
        if isinstance(response_data, dict):
            errors = []
            for field, messages in response_data.items():
                if isinstance(messages, list):
                    for msg in messages:
                        errors.append({'field': field, 'message': str(msg)})
                else:
                    errors.append({'field': field, 'message': str(messages)})
            return errors
        elif isinstance(response_data, list):
            return [{'field': 'non_field_errors', 'message': str(msg)} for msg in response_data]
    return [{'message': str(response_data)}]


class BunoraaAPIException(APIException):
    """Base exception for Bunoraa API errors."""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'An error occurred.'
    default_code = 'error'


class InvalidInputException(BunoraaAPIException):
    """Exception for invalid input data."""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'Invalid input provided.'
    default_code = 'invalid_input'


class ResourceNotFoundException(BunoraaAPIException):
    """Exception when a resource is not found."""
    status_code = status.HTTP_404_NOT_FOUND
    default_detail = 'Resource not found.'
    default_code = 'not_found'


class ConflictException(BunoraaAPIException):
    """Exception for resource conflicts."""
    status_code = status.HTTP_409_CONFLICT
    default_detail = 'Resource conflict.'
    default_code = 'conflict'


class InsufficientStockException(BunoraaAPIException):
    """Exception when product stock is insufficient."""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'Insufficient stock available.'
    default_code = 'insufficient_stock'


class PaymentException(BunoraaAPIException):
    """Exception for payment processing errors."""
    status_code = status.HTTP_402_PAYMENT_REQUIRED
    default_detail = 'Payment processing failed.'
    default_code = 'payment_failed'


class CartException(BunoraaAPIException):
    """Exception for cart-related errors."""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'Cart operation failed.'
    default_code = 'cart_error'


class OrderException(BunoraaAPIException):
    """Exception for order-related errors."""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'Order operation failed.'
    default_code = 'order_error'


class CSRFExemptSessionAuthentication(SessionAuthentication):
    """
    Custom SessionAuthentication that skips CSRF validation.
    
    This is safe because:
    1. API endpoints should use JWT tokens for authentication
    2. SessionAuthentication is a fallback for browser-based clients
    3. CSRF tokens are properly validated at the middleware level for form submissions
    4. JWT tokens inherently protect against CSRF attacks
    
    Use this class in views that accept both SessionAuthentication and JWTAuthentication
    to allow authenticated session-based clients (like web apps) to make API calls
    without CSRF token validation errors.
    """
    
    def enforce_csrf_checks(self, request):
        """
        Override to disable CSRF checks for API endpoints.
        
        The CSRF middleware will still run and validate tokens for form submissions,
        but DRF's SessionAuthentication won't enforce additional CSRF checks here.
        """
        # Return False to skip CSRF validation in SessionAuthentication
        return False
