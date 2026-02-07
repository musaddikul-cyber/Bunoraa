"""
Helper utilities
"""
import re
import uuid
from decimal import Decimal
from django.utils.text import slugify
from django.utils import timezone


def generate_unique_slug(model_class, value, slug_field='slug'):
    """
    Generate a unique slug for a model instance.
    """
    base_slug = slugify(value)
    slug = base_slug
    counter = 1
    
    while model_class.objects.filter(**{slug_field: slug}).exists():
        slug = f"{base_slug}-{counter}"
        counter += 1
    
    return slug


def generate_sku(prefix='SKU'):
    """
    Generate a unique SKU.
    """
    unique_id = uuid.uuid4().hex[:8].upper()
    return f"{prefix}-{unique_id}"


def generate_order_number():
    """
    Generate a unique order number.
    """
    timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
    unique_id = uuid.uuid4().hex[:6].upper()
    return f"ORD-{timestamp}-{unique_id}"


def format_price(amount, currency='USD'):
    """
    Format a price amount for display.
    """
    if isinstance(amount, str):
        amount = Decimal(amount)
    
    currency_symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'BDT': '৳',
    }
    
    symbol = currency_symbols.get(currency, currency)
    return f"{symbol}{amount:,.2f}"


def truncate_text(text, max_length=100, suffix='...'):
    """
    Truncate text to a maximum length.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix


def sanitize_filename(filename):
    """
    Sanitize a filename for safe storage.
    """
    # Remove any path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Remove special characters
    filename = re.sub(r'[^\w\s\-.]', '', filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    return filename


def get_upload_path(instance, filename, folder='uploads'):
    """
    Generate upload path for file uploads.
    """
    filename = sanitize_filename(filename)
    date_path = timezone.now().strftime('%Y/%m/%d')
    unique_id = uuid.uuid4().hex[:8]
    name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
    
    if ext:
        new_filename = f"{name}_{unique_id}.{ext}"
    else:
        new_filename = f"{name}_{unique_id}"
    
    return f"{folder}/{date_path}/{new_filename}"


def calculate_percentage(value, total):
    """
    Calculate percentage with zero division handling.
    """
    if total == 0:
        return Decimal('0.00')
    return Decimal(str((value / total) * 100)).quantize(Decimal('0.01'))


def mask_email(email):
    """
    Mask an email address for display.
    """
    if '@' not in email:
        return email
    
    local, domain = email.split('@')
    if len(local) <= 2:
        masked_local = '*' * len(local)
    else:
        masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
    
    return f"{masked_local}@{domain}"


def mask_phone(phone):
    """
    Mask a phone number for display.
    """
    if len(phone) <= 4:
        return '*' * len(phone)
    return '*' * (len(phone) - 4) + phone[-4:]
