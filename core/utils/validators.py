"""
Custom validators
"""
import re
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator


phone_validator = RegexValidator(
    regex=r'^\+?1?\d{9,15}$',
    message="Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed."
)


def validate_positive_decimal(value):
    """Validate that the value is a positive decimal."""
    if value < 0:
        raise ValidationError('Value must be positive.')


def validate_percentage(value):
    """Validate that the value is between 0 and 100."""
    if value < 0 or value > 100:
        raise ValidationError('Value must be between 0 and 100.')


def validate_image_extension(value):
    """Validate that the uploaded file is an image."""
    import os
    ext = os.path.splitext(value.name)[1].lower()
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']
    if ext not in valid_extensions:
        raise ValidationError(
            f'Unsupported file extension. Allowed: {", ".join(valid_extensions)}'
        )


def validate_file_size(value, max_size_mb=5):
    """Validate that the uploaded file is within size limit."""
    max_size = max_size_mb * 1024 * 1024  # Convert to bytes
    if value.size > max_size:
        raise ValidationError(f'File size must be under {max_size_mb}MB.')


def validate_sku(value):
    """Validate SKU format."""
    if not re.match(r'^[A-Z0-9\-]+$', value):
        raise ValidationError(
            'SKU must contain only uppercase letters, numbers, and hyphens.'
        )


def validate_slug(value):
    """Validate slug format."""
    if not re.match(r'^[a-z0-9]+(?:-[a-z0-9]+)*$', value):
        raise ValidationError(
            'Slug must contain only lowercase letters, numbers, and hyphens.'
        )


def validate_color_hex(value):
    """Validate hex color code."""
    if not re.match(r'^#[0-9A-Fa-f]{6}$', value):
        raise ValidationError(
            'Color must be a valid hex code (e.g., #FF5733).'
        )


def validate_json(value):
    """Validate JSON string."""
    import json
    try:
        json.loads(value)
    except (json.JSONDecodeError, TypeError):
        raise ValidationError('Value must be valid JSON.')


def validate_url(value):
    """Validate URL format."""
    from django.core.validators import URLValidator
    validator = URLValidator()
    try:
        validator(value)
    except ValidationError:
        raise ValidationError('Enter a valid URL.')
