"""
Currency Template Tags

This is an alias module for backward compatibility.
Currency tags are now part of the i18n app's template tags.

Usage in templates:
    {% load currency_tags %}
    
    or
    
    {% load i18n_tags %}
"""
from apps.i18n.templatetags import (
    register,
    format_currency,
    convert_currency,
    price,
    currency_symbol,
    format_price,
)

__all__ = [
    'register',
    'format_currency',
    'convert_currency',
    'price',
    'currency_symbol',
    'format_price',
]
