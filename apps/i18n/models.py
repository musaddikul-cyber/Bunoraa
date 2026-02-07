"""
Internationalization Models

Comprehensive models for multi-language, multi-currency, and localization support.
"""
import uuid
import hashlib
from decimal import Decimal
from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
from django.core.cache import cache


# =============================================================================
# Language Models
# =============================================================================

class Language(models.Model):
    """Available languages for the site."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    code = models.CharField(
        max_length=10,
        unique=True,
        db_index=True,
        help_text="Language code (e.g., en, bn, hi)"
    )
    name = models.CharField(max_length=100, help_text="Language name in English")
    native_name = models.CharField(max_length=100, help_text="Language name in native script")
    
    # Display settings
    is_rtl = models.BooleanField(default=False, help_text="Right-to-left language")
    flag_code = models.CharField(max_length=10, blank=True, help_text="Country code for flag icon")
    font_family = models.CharField(
        max_length=200,
        blank=True,
        help_text="CSS font-family for this language"
    )
    
    # Locale settings
    locale_code = models.CharField(
        max_length=20,
        blank=True,
        help_text="Full locale code (e.g., en_US, bn_BD)"
    )
    
    # Status
    is_active = models.BooleanField(default=True, db_index=True)
    is_default = models.BooleanField(default=False)
    
    # Completeness tracking
    translation_progress = models.PositiveIntegerField(
        default=0,
        validators=[MaxValueValidator(100)],
        help_text="Percentage of translations complete (0-100)"
    )
    total_strings = models.PositiveIntegerField(default=0)
    translated_strings = models.PositiveIntegerField(default=0)
    
    # Metadata
    sort_order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['sort_order', 'name']
        verbose_name = 'Language'
        verbose_name_plural = 'Languages'
        indexes = [
            models.Index(fields=['code', 'is_active']),
            models.Index(fields=['is_default']),
        ]
    
    def __str__(self):
        return f"{self.native_name} ({self.code})"
    
    def save(self, *args, **kwargs):
        # Ensure only one default
        if self.is_default:
            Language.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)
        # Update translation progress
        if self.total_strings > 0:
            self.translation_progress = int((self.translated_strings / self.total_strings) * 100)
        super().save(*args, **kwargs)
        # Clear cache
        cache.delete_many(['active_languages', 'default_language', f'language_{self.code}'])


# =============================================================================
# Currency Models
# =============================================================================

class Currency(models.Model):
    """Currency for multi-currency support."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    code = models.CharField(
        max_length=3,
        unique=True,
        db_index=True,
        help_text="ISO 4217 currency code (e.g., USD, BDT)"
    )
    name = models.CharField(max_length=100, help_text="Currency name")
    symbol = models.CharField(max_length=10, help_text="Currency symbol (e.g., $, ৳)")
    native_symbol = models.CharField(
        max_length=10,
        blank=True,
        help_text="Native symbol if different"
    )
    
    # Display settings
    decimal_places = models.PositiveSmallIntegerField(default=2)
    symbol_position = models.CharField(
        max_length=20,
        choices=[
            ('before', 'Before amount ($100)'),
            ('before_space', 'Before with space ($ 100)'),
            ('after', 'After amount (100$)'),
            ('after_space', 'After with space (100 $)'),
        ],
        default='before'
    )
    thousand_separator = models.CharField(max_length=5, default=',')
    decimal_separator = models.CharField(max_length=5, default='.')
    
    # Number system
    number_system = models.CharField(
        max_length=20,
        choices=[
            ('western', 'Western (1,234.56)'),
            ('indian', 'Indian (1,23,456.78)'),
            ('bengali', 'Bengali numerals'),
        ],
        default='western'
    )
    
    # Exchange rate info
    is_base_currency = models.BooleanField(
        default=False,
        help_text="Use as base for exchange rate calculations"
    )
    
    # Status
    is_active = models.BooleanField(default=True, db_index=True)
    is_default = models.BooleanField(default=False)
    
    # Sorting
    sort_order = models.PositiveIntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['sort_order', 'code']
        verbose_name = 'Currency'
        verbose_name_plural = 'Currencies'
        indexes = [
            models.Index(fields=['code', 'is_active']),
            models.Index(fields=['is_default']),
        ]
    
    def __str__(self):
        return f"{self.code} - {self.name}"
    
    def save(self, *args, **kwargs):
        # Ensure only one default currency
        if self.is_default:
            Currency.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)
        super().save(*args, **kwargs)
        # Clear cache
        cache.delete_many(['active_currencies', 'default_currency', f'currency_{self.code}'])
    
    def format_amount(self, amount, use_native_symbol=False):
        """Format an amount in this currency."""
        from decimal import ROUND_HALF_UP
        
        amount = Decimal(str(amount)).quantize(
            Decimal(10) ** -self.decimal_places,
            rounding=ROUND_HALF_UP
        )
        
        # Format number based on system
        formatted_number = self._format_number(amount)
        
        # Get symbol
        symbol = self.native_symbol if use_native_symbol and self.native_symbol else self.symbol
        
        # Apply symbol position
        if self.symbol_position == 'before':
            return f"{symbol}{formatted_number}"
        elif self.symbol_position == 'before_space':
            return f"{symbol} {formatted_number}"
        elif self.symbol_position == 'after':
            return f"{formatted_number}{symbol}"
        else:  # after_space
            return f"{formatted_number} {symbol}"
    
    def _format_number(self, amount):
        """Format number based on number system."""
        parts = str(amount).split('.')
        integer_part = parts[0]
        decimal_part = parts[1] if len(parts) > 1 else '0' * self.decimal_places
        
        # Pad decimal part
        decimal_part = decimal_part.ljust(self.decimal_places, '0')[:self.decimal_places]
        
        # Format integer part based on system
        if self.number_system == 'indian':
            formatted_int = self._format_indian_number(integer_part)
        elif self.number_system == 'bengali':
            formatted_int = self._add_separators(integer_part)
            # Convert to Bengali numerals
            bengali_digits = '০১২৩৪৫৬৭৮৯'
            formatted_int = ''.join(bengali_digits[int(d)] if d.isdigit() else d for d in formatted_int)
            decimal_part = ''.join(bengali_digits[int(d)] if d.isdigit() else d for d in decimal_part)
        else:
            formatted_int = self._add_separators(integer_part)
        
        if self.decimal_places > 0:
            return f"{formatted_int}{self.decimal_separator}{decimal_part}"
        return formatted_int
    
    def _add_separators(self, s):
        """Add thousand separators (Western style)."""
        if not self.thousand_separator:
            return s
        result = []
        for i, digit in enumerate(reversed(s)):
            if i > 0 and i % 3 == 0:
                result.append(self.thousand_separator)
            result.append(digit)
        return ''.join(reversed(result))
    
    def _format_indian_number(self, s):
        """Format in Indian numbering system (e.g., 1,00,00,000)."""
        if len(s) <= 3:
            return s
        
        # First 3 digits from right
        result = s[-3:]
        s = s[:-3]
        
        # Remaining in groups of 2
        while s:
            result = s[-2:] + self.thousand_separator + result
            s = s[:-2]
        
        return result


class ExchangeRate(models.Model):
    """Exchange rate between currencies."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    from_currency = models.ForeignKey(
        Currency,
        on_delete=models.CASCADE,
        related_name='rates_from'
    )
    to_currency = models.ForeignKey(
        Currency,
        on_delete=models.CASCADE,
        related_name='rates_to'
    )
    rate = models.DecimalField(
        max_digits=18,
        decimal_places=8,
        validators=[MinValueValidator(Decimal('0.00000001'))],
        help_text="Exchange rate (1 from_currency = rate to_currency)"
    )
    
    # Bid/Ask spread for accurate conversions
    bid_rate = models.DecimalField(
        max_digits=18,
        decimal_places=8,
        null=True,
        blank=True,
        help_text="Buying rate"
    )
    ask_rate = models.DecimalField(
        max_digits=18,
        decimal_places=8,
        null=True,
        blank=True,
        help_text="Selling rate"
    )
    
    # Source tracking
    source = models.CharField(
        max_length=50,
        choices=[
            ('manual', 'Manual'),
            ('openexchange', 'Open Exchange Rates'),
            ('fixer', 'Fixer.io'),
            ('ecb', 'European Central Bank'),
            ('xe', 'XE.com'),
            ('currencylayer', 'CurrencyLayer'),
            ('exchangerate_api', 'ExchangeRate-API'),
        ],
        default='manual'
    )
    
    # Validity
    valid_from = models.DateTimeField(default=timezone.now, db_index=True)
    valid_until = models.DateTimeField(null=True, blank=True, db_index=True)
    is_active = models.BooleanField(default=True, db_index=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-valid_from']
        verbose_name = 'Exchange Rate'
        verbose_name_plural = 'Exchange Rates'
        indexes = [
            models.Index(fields=['from_currency', 'to_currency', 'is_active']),
            models.Index(fields=['valid_from', 'valid_until']),
        ]
        constraints = [
            models.CheckConstraint(
                check=~models.Q(from_currency=models.F('to_currency')),
                name='different_currencies'
            ),
        ]
    
    def __str__(self):
        return f"{self.from_currency.code} → {self.to_currency.code}: {self.rate}"
    
    @property
    def is_valid(self):
        """Check if rate is currently valid."""
        now = timezone.now()
        if not self.is_active:
            return False
        if now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True
    
    @property
    def inverse_rate(self):
        """Get the inverse rate."""
        if self.rate:
            return Decimal('1') / self.rate
        return None
    
    @property
    def spread_percentage(self):
        """Calculate bid-ask spread percentage."""
        if self.bid_rate and self.ask_rate and self.ask_rate > 0:
            return ((self.ask_rate - self.bid_rate) / self.ask_rate) * 100
        return None


class ExchangeRateHistory(models.Model):
    """Historical exchange rates for analytics and charting."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    from_currency = models.ForeignKey(
        Currency,
        on_delete=models.CASCADE,
        related_name='history_from'
    )
    to_currency = models.ForeignKey(
        Currency,
        on_delete=models.CASCADE,
        related_name='history_to'
    )
    
    # OHLC data for charting
    rate = models.DecimalField(max_digits=18, decimal_places=8, help_text="Closing rate")
    open_rate = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True)
    high_rate = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True)
    low_rate = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True)
    
    date = models.DateField(db_index=True)
    source = models.CharField(max_length=50)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-date']
        unique_together = ['from_currency', 'to_currency', 'date']
        verbose_name = 'Exchange Rate History'
        verbose_name_plural = 'Exchange Rate History'
        indexes = [
            models.Index(fields=['from_currency', 'to_currency', 'date']),
        ]
    
    def __str__(self):
        return f"{self.from_currency.code} → {self.to_currency.code}: {self.rate} ({self.date})"


# =============================================================================
# Timezone Models
# =============================================================================

class Timezone(models.Model):
    """Available timezones."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(
        max_length=100,
        unique=True,
        help_text="IANA timezone name (e.g., Asia/Dhaka)"
    )
    display_name = models.CharField(max_length=200, help_text="User-friendly name")
    offset = models.CharField(max_length=10, help_text="UTC offset (e.g., +06:00)")
    offset_minutes = models.IntegerField(default=0, help_text="Offset in minutes for sorting")
    
    # DST info
    has_dst = models.BooleanField(default=False, help_text="Has daylight saving time")
    dst_offset = models.CharField(max_length=10, blank=True, help_text="DST offset")
    
    is_active = models.BooleanField(default=True)
    is_common = models.BooleanField(default=False, help_text="Show in common timezones list")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['offset_minutes', 'name']
        verbose_name = 'Timezone'
        verbose_name_plural = 'Timezones'
    
    def __str__(self):
        return f"{self.display_name} ({self.offset})"


# =============================================================================
# Geographic Models (Bangladesh-specific hierarchy)
# =============================================================================

class Country(models.Model):
    """Countries for localization and shipping."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    code = models.CharField(
        max_length=2,
        unique=True,
        db_index=True,
        help_text="ISO 3166-1 alpha-2 code"
    )
    code_alpha3 = models.CharField(max_length=3, blank=True, help_text="ISO 3166-1 alpha-3 code")
    code_numeric = models.CharField(max_length=3, blank=True, help_text="ISO 3166-1 numeric code")
    name = models.CharField(max_length=100)
    native_name = models.CharField(max_length=100, blank=True)
    
    # Localization defaults
    default_language = models.ForeignKey(
        Language,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='default_for_countries'
    )
    default_currency = models.ForeignKey(
        Currency,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='default_for_countries'
    )
    default_timezone = models.ForeignKey(
        Timezone,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='default_for_countries'
    )
    
    # Phone
    phone_code = models.CharField(max_length=10, blank=True, help_text="International dialing code")
    phone_format = models.CharField(
        max_length=50,
        blank=True,
        help_text="Phone format pattern (e.g., +880 XXXX-XXXXXX)"
    )
    
    # Address format
    address_format = models.TextField(
        blank=True,
        help_text="Address format template"
    )
    postal_code_format = models.CharField(
        max_length=50,
        blank=True,
        help_text="Postal code regex pattern"
    )
    
    # VAT/Tax
    vat_number_format = models.CharField(max_length=50, blank=True)
    default_tax_rate = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=Decimal('0'),
        help_text="Default VAT/GST rate"
    )
    
    # Geography
    continent = models.CharField(
        max_length=50,
        choices=[
            ('africa', 'Africa'),
            ('antarctica', 'Antarctica'),
            ('asia', 'Asia'),
            ('europe', 'Europe'),
            ('north_america', 'North America'),
            ('oceania', 'Oceania'),
            ('south_america', 'South America'),
        ],
        blank=True
    )
    region = models.CharField(max_length=100, blank=True, help_text="Sub-region (e.g., South Asia)")
    
    # Status
    is_active = models.BooleanField(default=True)
    is_shipping_available = models.BooleanField(default=True)
    is_billing_allowed = models.BooleanField(default=True)
    
    sort_order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['sort_order', 'name']
        verbose_name = 'Country'
        verbose_name_plural = 'Countries'
    
    def __str__(self):
        return self.name


class Division(models.Model):
    """
    Division/State/Province within a country.
    For Bangladesh: 8 divisions (Dhaka, Chittagong, Rajshahi, Khulna, Barisal, Sylhet, Rangpur, Mymensingh)
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    country = models.ForeignKey(
        Country,
        on_delete=models.CASCADE,
        related_name='divisions'
    )
    code = models.CharField(max_length=10, help_text="Division code (e.g., DHK)")
    name = models.CharField(max_length=100)
    native_name = models.CharField(max_length=100, blank=True)
    
    # Type
    division_type = models.CharField(
        max_length=30,
        choices=[
            ('division', 'Division'),
            ('state', 'State'),
            ('province', 'Province'),
            ('region', 'Region'),
            ('territory', 'Territory'),
        ],
        default='division'
    )
    
    # Coordinates for mapping
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    
    # Shipping
    is_active = models.BooleanField(default=True)
    is_shipping_available = models.BooleanField(default=True)
    shipping_zone = models.CharField(max_length=50, blank=True)
    
    sort_order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['sort_order', 'name']
        unique_together = ['country', 'code']
        verbose_name = 'Division'
        verbose_name_plural = 'Divisions'
    
    def __str__(self):
        return f"{self.name}, {self.country.name}"


class District(models.Model):
    """
    District/City within a Division.
    For Bangladesh: 64 districts
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    division = models.ForeignKey(
        Division,
        on_delete=models.CASCADE,
        related_name='districts'
    )
    code = models.CharField(max_length=10, help_text="District code")
    name = models.CharField(max_length=100)
    native_name = models.CharField(max_length=100, blank=True)
    
    # Coordinates for mapping
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    is_shipping_available = models.BooleanField(default=True)
    
    # Shipping configuration
    shipping_zone = models.CharField(
        max_length=20,
        choices=[
            ('metro', 'Metro/City'),
            ('suburban', 'Suburban'),
            ('rural', 'Rural'),
            ('remote', 'Remote'),
        ],
        default='suburban',
        help_text="Used for shipping rate calculation"
    )
    
    # Delivery estimates
    estimated_delivery_days_min = models.PositiveIntegerField(default=2)
    estimated_delivery_days_max = models.PositiveIntegerField(default=5)
    
    sort_order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['sort_order', 'name']
        unique_together = ['division', 'code']
        verbose_name = 'District'
        verbose_name_plural = 'Districts'
    
    def __str__(self):
        return f"{self.name}, {self.division.name}"
    
    @property
    def country(self):
        return self.division.country


class Upazila(models.Model):
    """
    Upazila/Thana/Subdistrict within a District.
    For Bangladesh: ~500 upazilas
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    district = models.ForeignKey(
        District,
        on_delete=models.CASCADE,
        related_name='upazilas'
    )
    code = models.CharField(max_length=10, help_text="Upazila code")
    name = models.CharField(max_length=100)
    native_name = models.CharField(max_length=100, blank=True)
    
    # Type
    upazila_type = models.CharField(
        max_length=20,
        choices=[
            ('upazila', 'Upazila'),
            ('thana', 'Thana'),
            ('municipality', 'Municipality'),
            ('city_corporation', 'City Corporation'),
            ('cantonment', 'Cantonment'),
        ],
        default='upazila'
    )
    
    # Coordinates
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    is_shipping_available = models.BooleanField(default=True)
    
    # Postal codes for this upazila
    postal_codes = models.JSONField(
        default=list,
        blank=True,
        help_text="List of postal codes for this upazila"
    )
    
    sort_order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['sort_order', 'name']
        unique_together = ['district', 'code']
        verbose_name = 'Upazila'
        verbose_name_plural = 'Upazilas'
    
    def __str__(self):
        return f"{self.name}, {self.district.name}"
    
    @property
    def division(self):
        return self.district.division
    
    @property
    def country(self):
        return self.district.division.country


# =============================================================================
# Translation Models
# =============================================================================

class TranslationNamespace(models.Model):
    """Namespace for organizing translations."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    
    class Meta:
        ordering = ['name']
        verbose_name = 'Translation Namespace'
        verbose_name_plural = 'Translation Namespaces'
    
    def __str__(self):
        return self.name


class TranslationKey(models.Model):
    """Translation key for static strings."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    namespace = models.ForeignKey(
        TranslationNamespace,
        on_delete=models.CASCADE,
        related_name='keys',
        null=True,
        blank=True
    )
    key = models.CharField(max_length=255, help_text="Unique key (e.g., common.save_button)")
    source_text = models.TextField(help_text="Source text in default language")
    context = models.TextField(blank=True, help_text="Context for translators")
    
    # Metadata
    max_length = models.PositiveIntegerField(null=True, blank=True, help_text="Maximum character length")
    is_html = models.BooleanField(default=False, help_text="Contains HTML")
    is_plural = models.BooleanField(default=False, help_text="Has plural forms")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['namespace__name', 'key']
        unique_together = ['namespace', 'key']
        verbose_name = 'Translation Key'
        verbose_name_plural = 'Translation Keys'
    
    def __str__(self):
        if self.namespace:
            return f"{self.namespace.name}:{self.key}"
        return self.key


class Translation(models.Model):
    """Translation for a specific key and language."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    key = models.ForeignKey(
        TranslationKey,
        on_delete=models.CASCADE,
        related_name='translations'
    )
    language = models.ForeignKey(
        Language,
        on_delete=models.CASCADE,
        related_name='translations'
    )
    
    # Translations
    translated_text = models.TextField()
    plural_forms = models.JSONField(
        default=dict,
        blank=True,
        help_text="Plural translations: {0: 'zero', 1: 'one', 2: 'two', 'few': '...', 'many': '...', 'other': '...'}"
    )
    
    # Status
    status = models.CharField(
        max_length=20,
        choices=[
            ('draft', 'Draft'),
            ('pending_review', 'Pending Review'),
            ('approved', 'Approved'),
            ('rejected', 'Rejected'),
        ],
        default='draft'
    )
    is_machine_translated = models.BooleanField(default=False)
    
    # Quality
    quality_score = models.PositiveIntegerField(
        null=True,
        blank=True,
        validators=[MaxValueValidator(100)],
        help_text="Translation quality score (0-100)"
    )
    
    # Audit
    translated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='translations_created'
    )
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='translations_reviewed'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['key__key', 'language__code']
        unique_together = ['key', 'language']
        verbose_name = 'Translation'
        verbose_name_plural = 'Translations'
    
    def __str__(self):
        return f"{self.key.key} ({self.language.code})"


class ContentTranslation(models.Model):
    """Translation for dynamic content (products, categories, etc.)."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Content identification
    content_type = models.CharField(
        max_length=50,
        choices=[
            ('product', 'Product'),
            ('category', 'Category'),
            ('page', 'Page'),
            ('email', 'Email Template'),
            ('notification', 'Notification'),
            ('banner', 'Banner'),
            ('menu', 'Menu Item'),
        ],
        db_index=True
    )
    content_id = models.CharField(max_length=100, db_index=True, help_text="UUID of the content")
    field_name = models.CharField(max_length=100, help_text="Field being translated")
    
    language = models.ForeignKey(Language, on_delete=models.CASCADE, related_name='content_translations')
    
    # Translation
    original_text = models.TextField(blank=True)
    translated_text = models.TextField()
    
    # Status
    is_approved = models.BooleanField(default=False)
    is_machine_translated = models.BooleanField(default=False)
    
    # Audit
    translated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    approved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='approved_content_translations'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['content_type', 'content_id', 'field_name']
        unique_together = ['content_type', 'content_id', 'field_name', 'language']
        verbose_name = 'Content Translation'
        verbose_name_plural = 'Content Translations'
        indexes = [
            models.Index(fields=['content_type', 'content_id', 'language']),
        ]
    
    def __str__(self):
        return f"{self.content_type}.{self.content_id}.{self.field_name} ({self.language.code})"


# =============================================================================
# User Preferences
# =============================================================================

class UserLocalePreference(models.Model):
    """User's locale preferences."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='i18n_preference'
    )
    
    # Preferences
    language = models.ForeignKey(
        Language,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    currency = models.ForeignKey(
        Currency,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    timezone = models.ForeignKey(
        Timezone,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    country = models.ForeignKey(
        Country,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    
    # Date/Time formatting preferences
    date_format = models.CharField(
        max_length=20,
        choices=[
            ('MM/DD/YYYY', 'MM/DD/YYYY'),
            ('DD/MM/YYYY', 'DD/MM/YYYY'),
            ('YYYY-MM-DD', 'YYYY-MM-DD'),
            ('DD.MM.YYYY', 'DD.MM.YYYY'),
            ('YYYY年MM月DD日', 'YYYY年MM月DD日'),
        ],
        default='DD/MM/YYYY'  # Bangladesh default
    )
    time_format = models.CharField(
        max_length=10,
        choices=[
            ('12h', '12-hour'),
            ('24h', '24-hour'),
        ],
        default='12h'
    )
    first_day_of_week = models.PositiveSmallIntegerField(
        default=0,
        choices=[(i, day) for i, day in enumerate(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])],
        help_text="0=Sunday, 1=Monday, etc."
    )
    
    # Measurement preferences
    measurement_system = models.CharField(
        max_length=10,
        choices=[
            ('metric', 'Metric'),
            ('imperial', 'Imperial'),
        ],
        default='metric'
    )
    temperature_unit = models.CharField(
        max_length=10,
        choices=[
            ('celsius', 'Celsius'),
            ('fahrenheit', 'Fahrenheit'),
        ],
        default='celsius'
    )
    
    # Auto-detection
    auto_detect_language = models.BooleanField(default=True)
    auto_detect_currency = models.BooleanField(default=True)
    auto_detect_timezone = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'User Locale Preference'
        verbose_name_plural = 'User Locale Preferences'
    
    def __str__(self):
        return f"Locale for {self.user.email}"


# =============================================================================
# Settings Models (Singletons)
# =============================================================================

class I18nSettings(models.Model):
    """Global i18n settings (singleton)."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Language settings
    default_language = models.ForeignKey(
        Language,
        on_delete=models.PROTECT,
        related_name='+'
    )
    fallback_language = models.ForeignKey(
        Language,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='+',
        help_text="Fallback if translation not available"
    )
    auto_detect_language = models.BooleanField(default=True)
    show_language_selector = models.BooleanField(default=True)
    
    # Currency settings
    default_currency = models.ForeignKey(
        Currency,
        on_delete=models.PROTECT,
        related_name='+'
    )
    auto_detect_currency = models.BooleanField(default=True)
    show_currency_selector = models.BooleanField(default=True)
    show_original_price = models.BooleanField(
        default=False,
        help_text="Show original price alongside converted price"
    )
    
    # Timezone settings
    default_timezone = models.ForeignKey(
        Timezone,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='+'
    )
    auto_detect_timezone = models.BooleanField(default=True)
    show_timezone_selector = models.BooleanField(default=False)
    
    # Exchange rate settings
    auto_update_exchange_rates = models.BooleanField(default=True)
    exchange_rate_update_frequency = models.PositiveIntegerField(
        default=24,
        help_text="Hours between updates"
    )
    exchange_rate_provider = models.CharField(
        max_length=50,
        choices=[
            ('manual', 'Manual'),
            ('openexchange', 'Open Exchange Rates'),
            ('fixer', 'Fixer.io'),
            ('ecb', 'European Central Bank'),
            ('exchangerate_api', 'ExchangeRate-API'),
        ],
        default='manual'
    )
    exchange_rate_api_key = models.CharField(max_length=255, blank=True)
    last_exchange_rate_update = models.DateTimeField(null=True, blank=True)
    
    # Price rounding
    rounding_method = models.CharField(
        max_length=20,
        choices=[
            ('none', 'No rounding'),
            ('nearest_cent', 'Nearest cent'),
            ('nearest_99', 'Nearest .99'),
            ('nearest_95', 'Nearest .95'),
            ('nearest_integer', 'Nearest whole number'),
            ('round_up', 'Always round up'),
        ],
        default='nearest_cent'
    )
    
    # Machine translation
    enable_machine_translation = models.BooleanField(default=False)
    translation_provider = models.CharField(
        max_length=50,
        choices=[
            ('google', 'Google Translate'),
            ('deepl', 'DeepL'),
            ('azure', 'Azure Translator'),
            ('aws', 'Amazon Translate'),
        ],
        blank=True
    )
    translation_api_key = models.CharField(max_length=255, blank=True)
    auto_translate_new_content = models.BooleanField(default=False)
    require_human_review = models.BooleanField(
        default=True,
        help_text="Require human review for machine translations"
    )
    
    # Geo-IP settings
    enable_geo_detection = models.BooleanField(default=True)
    geo_ip_provider = models.CharField(
        max_length=50,
        choices=[
            ('cloudflare', 'Cloudflare Headers'),
            ('maxmind', 'MaxMind GeoIP'),
            ('ipinfo', 'IPinfo'),
        ],
        default='cloudflare'
    )
    geo_ip_api_key = models.CharField(max_length=255, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'I18n Settings'
        verbose_name_plural = 'I18n Settings'
    
    def __str__(self):
        return "I18n Settings"
    
    def save(self, *args, **kwargs):
        if not self.pk and I18nSettings.objects.exists():
            raise ValueError("Only one I18nSettings instance allowed")
        super().save(*args, **kwargs)
    
    @classmethod
    def get_settings(cls):
        """Get or create the settings instance."""
        settings = cls.objects.first()
        if not settings:
            # Create defaults
            language, _ = Language.objects.get_or_create(
                code='bn',
                defaults={
                    'name': 'Bengali',
                    'native_name': 'বাংলা',
                    'is_default': True,
                    'flag_code': 'BD'
                }
            )
            currency, _ = Currency.objects.get_or_create(
                code='BDT',
                defaults={
                    'name': 'Bangladeshi Taka',
                    'symbol': '৳',
                    'is_default': True
                }
            )
            timezone_obj, _ = Timezone.objects.get_or_create(
                name='Asia/Dhaka',
                defaults={
                    'display_name': 'Bangladesh Time',
                    'offset': '+06:00',
                    'offset_minutes': 360
                }
            )
            settings = cls.objects.create(
                default_language=language,
                default_currency=currency,
                default_timezone=timezone_obj
            )
        return settings
