"""
Initial Data for i18n App

Management command to load initial data for languages, currencies, etc.
"""
from django.core.management.base import BaseCommand
from apps.i18n.models import Language, Currency, Timezone, Country


class Command(BaseCommand):
    help = 'Load initial i18n data (languages, currencies, timezones, countries)'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--all',
            action='store_true',
            help='Load all data types',
        )
        parser.add_argument(
            '--languages',
            action='store_true',
            help='Load languages only',
        )
        parser.add_argument(
            '--currencies',
            action='store_true',
            help='Load currencies only',
        )
        parser.add_argument(
            '--timezones',
            action='store_true',
            help='Load timezones only',
        )
        parser.add_argument(
            '--countries',
            action='store_true',
            help='Load countries only',
        )
    
    def handle(self, *args, **options):
        load_all = options['all'] or not any([
            options['languages'], options['currencies'], 
            options['timezones'], options['countries']
        ])
        
        if load_all or options['languages']:
            self._load_languages()
        
        if load_all or options['currencies']:
            self._load_currencies()
        
        if load_all or options['timezones']:
            self._load_timezones()
        
        if load_all or options['countries']:
            self._load_countries()
        
        self.stdout.write(self.style.SUCCESS('Successfully loaded i18n data'))
    
    def _load_languages(self):
        """Load common languages."""
        languages = [
            # code, name, native_name, flag_code, locale_code, is_rtl, is_default
            ('en', 'English', 'English', 'GB', 'en_US', False, False),
            ('bn', 'Bengali', 'বাংলা', 'BD', 'bn_BD', False, True),  # Default for Bangladesh
            ('hi', 'Hindi', 'हिन्दी', 'IN', 'hi_IN', False, False),
            ('ar', 'Arabic', 'العربية', 'SA', 'ar_SA', True, False),
            ('ur', 'Urdu', 'اردو', 'PK', 'ur_PK', True, False),
            ('es', 'Spanish', 'Español', 'ES', 'es_ES', False, False),
            ('fr', 'French', 'Français', 'FR', 'fr_FR', False, False),
            ('de', 'German', 'Deutsch', 'DE', 'de_DE', False, False),
            ('zh', 'Chinese', '中文', 'CN', 'zh_CN', False, False),
            ('ja', 'Japanese', '日本語', 'JP', 'ja_JP', False, False),
        ]
        
        for i, data in enumerate(languages):
            code, name, native, flag, locale, rtl, default = data
            Language.objects.update_or_create(
                code=code,
                defaults={
                    'name': name,
                    'native_name': native,
                    'flag_code': flag,
                    'locale_code': locale,
                    'is_rtl': rtl,
                    'is_default': default,
                    'is_active': True,
                    'sort_order': i
                }
            )
        
        self.stdout.write(f'Loaded {len(languages)} languages')
    
    def _load_currencies(self):
        """Load common currencies."""
        currencies = [
            # code, name, symbol, native_symbol, decimal_places, position, default, number_system
            ('BDT', 'Bangladeshi Taka', '৳', '৳', 2, 'before', True, 'bengali'),
            ('USD', 'US Dollar', '$', '$', 2, 'before', False, 'western'),
            ('EUR', 'Euro', '€', '€', 2, 'before', False, 'western'),
            ('GBP', 'British Pound', '£', '£', 2, 'before', False, 'western'),
            ('INR', 'Indian Rupee', '₹', '₹', 2, 'before', False, 'indian'),
            ('SAR', 'Saudi Riyal', 'SAR', 'ر.س', 2, 'after', False, 'western'),
            ('AED', 'UAE Dirham', 'AED', 'د.إ', 2, 'after', False, 'western'),
            ('JPY', 'Japanese Yen', '¥', '¥', 0, 'before', False, 'western'),
            ('CNY', 'Chinese Yuan', '¥', '¥', 2, 'before', False, 'western'),
            ('SGD', 'Singapore Dollar', 'S$', 'S$', 2, 'before', False, 'western'),
            ('MYR', 'Malaysian Ringgit', 'RM', 'RM', 2, 'before', False, 'western'),
            ('PKR', 'Pakistani Rupee', 'Rs', 'Rs', 2, 'before', False, 'indian'),
            ('NPR', 'Nepalese Rupee', 'Rs', 'रू', 2, 'before', False, 'indian'),
        ]
        
        for i, data in enumerate(currencies):
            code, name, symbol, native, decimal, pos, default, num_sys = data
            Currency.objects.update_or_create(
                code=code,
                defaults={
                    'name': name,
                    'symbol': symbol,
                    'native_symbol': native,
                    'decimal_places': decimal,
                    'symbol_position': pos,
                    'is_default': default,
                    'is_active': True,
                    'number_system': num_sys,
                    'sort_order': i
                }
            )
        
        self.stdout.write(f'Loaded {len(currencies)} currencies')
    
    def _load_timezones(self):
        """Load common timezones."""
        timezones = [
            # name, display_name, offset_str, offset_minutes, is_common, has_dst
            ('Asia/Dhaka', 'Bangladesh Time (BST)', '+06:00', 360, True, False),
            ('Asia/Kolkata', 'India Time (IST)', '+05:30', 330, True, False),
            ('Asia/Karachi', 'Pakistan Time (PKT)', '+05:00', 300, True, False),
            ('UTC', 'Coordinated Universal Time (UTC)', '+00:00', 0, True, False),
            ('Europe/London', 'British Time (GMT/BST)', '+00:00', 0, True, True),
            ('Europe/Paris', 'Central European Time (CET)', '+01:00', 60, True, True),
            ('America/New_York', 'Eastern Time (ET)', '-05:00', -300, True, True),
            ('America/Los_Angeles', 'Pacific Time (PT)', '-08:00', -480, True, True),
            ('Asia/Dubai', 'Gulf Time (GST)', '+04:00', 240, True, False),
            ('Asia/Singapore', 'Singapore Time (SGT)', '+08:00', 480, True, False),
            ('Asia/Tokyo', 'Japan Time (JST)', '+09:00', 540, True, False),
            ('Asia/Shanghai', 'China Time (CST)', '+08:00', 480, True, False),
            ('Australia/Sydney', 'Australian Eastern Time (AEST)', '+10:00', 600, True, True),
        ]
        
        for data in timezones:
            name, display, offset_str, offset_min, common, dst = data
            Timezone.objects.update_or_create(
                name=name,
                defaults={
                    'display_name': display,
                    'offset': offset_str,
                    'offset_minutes': offset_min,
                    'is_common': common,
                    'has_dst': dst,
                    'is_active': True
                }
            )
        
        self.stdout.write(f'Loaded {len(timezones)} timezones')
    
    def _load_countries(self):
        """Load common countries."""
        countries = [
            # code, code_alpha3, name, native_name, phone_code, continent, shipping
            ('BD', 'BGD', 'Bangladesh', 'বাংলাদেশ', '880', 'asia', True),
            ('IN', 'IND', 'India', 'भारत', '91', 'asia', True),
            ('US', 'USA', 'United States', 'United States', '1', 'north_america', False),
            ('GB', 'GBR', 'United Kingdom', 'United Kingdom', '44', 'europe', False),
            ('AE', 'ARE', 'United Arab Emirates', 'الإمارات', '971', 'asia', True),
            ('SA', 'SAU', 'Saudi Arabia', 'السعودية', '966', 'asia', True),
            ('SG', 'SGP', 'Singapore', 'Singapore', '65', 'asia', True),
            ('MY', 'MYS', 'Malaysia', 'Malaysia', '60', 'asia', True),
            ('PK', 'PAK', 'Pakistan', 'پاکستان', '92', 'asia', True),
            ('NP', 'NPL', 'Nepal', 'नेपाल', '977', 'asia', True),
            ('JP', 'JPN', 'Japan', '日本', '81', 'asia', False),
            ('CN', 'CHN', 'China', '中国', '86', 'asia', False),
            ('DE', 'DEU', 'Germany', 'Deutschland', '49', 'europe', False),
            ('FR', 'FRA', 'France', 'France', '33', 'europe', False),
            ('AU', 'AUS', 'Australia', 'Australia', '61', 'oceania', False),
            ('CA', 'CAN', 'Canada', 'Canada', '1', 'north_america', False),
        ]
        
        for data in countries:
            code, code3, name, native, phone, continent, shipping = data
            Country.objects.update_or_create(
                code=code,
                defaults={
                    'code_alpha3': code3,
                    'name': name,
                    'native_name': native,
                    'phone_code': phone,
                    'continent': continent,
                    'is_shipping_available': shipping,
                    'is_active': True
                }
            )
        
        self.stdout.write(f'Loaded {len(countries)} countries')
