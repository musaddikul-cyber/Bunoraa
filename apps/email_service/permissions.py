"""
Email Service Permissions
=========================

Permission classes for the email service API.
"""

from rest_framework import permissions
from .models import APIKey


class HasAPIKeyPermission(permissions.BasePermission):
    """
    Permission class that checks if the request has a valid API key
    with the required permission level.
    """
    
    message = 'Valid API key with appropriate permissions required.'
    
    def has_permission(self, request, view):
        # Check if authenticated via API key
        if not hasattr(request, 'auth') or not isinstance(request.auth, APIKey):
            return False
        
        api_key = request.auth
        
        # Full access can do anything
        if api_key.permission == APIKey.Permission.FULL_ACCESS:
            return True
        
        # Check specific permissions based on view
        view_name = getattr(view, 'basename', '') or view.__class__.__name__.lower()
        action = getattr(view, 'action', request.method.lower())
        
        # Mail send permission
        if view_name == 'mailsend' or 'send' in view_name:
            return api_key.permission == APIKey.Permission.MAIL_SEND
        
        # Template management
        if 'template' in view_name:
            if action in ['list', 'retrieve']:
                return api_key.permission in [
                    APIKey.Permission.TEMPLATES,
                    APIKey.Permission.READ_ONLY
                ]
            return api_key.permission == APIKey.Permission.TEMPLATES
        
        # Read-only access
        if action in ['list', 'retrieve', 'get']:
            return api_key.permission == APIKey.Permission.READ_ONLY
        
        return False


class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Object-level permission to only allow owners of an object to edit it.
    """
    
    def has_object_permission(self, request, view, obj):
        # Read permissions are allowed for any request
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions only for the owner
        return obj.user == request.user


class CanManageAPIKeys(permissions.BasePermission):
    """
    Permission to manage API keys.
    Only authenticated users can manage their own API keys.
    """
    
    def has_permission(self, request, view):
        return request.user and request.user.is_authenticated
    
    def has_object_permission(self, request, view, obj):
        return obj.user == request.user


class CanManageDomains(permissions.BasePermission):
    """
    Permission to manage sender domains.
    """
    
    def has_permission(self, request, view):
        return request.user and request.user.is_authenticated
    
    def has_object_permission(self, request, view, obj):
        return obj.user == request.user


class CanViewStats(permissions.BasePermission):
    """
    Permission to view email statistics.
    """
    
    def has_permission(self, request, view):
        # Check for API key with read access
        if hasattr(request, 'auth') and isinstance(request.auth, APIKey):
            return request.auth.permission in [
                APIKey.Permission.FULL_ACCESS,
                APIKey.Permission.READ_ONLY
            ]
        
        # Or authenticated user
        return request.user and request.user.is_authenticated
