"""
Schema helpers for API documentation defaults.
"""
from rest_framework import serializers


class DefaultAPIViewSerializer(serializers.Serializer):
    """Fallback serializer for APIViews without an explicit serializer_class."""
    pass

