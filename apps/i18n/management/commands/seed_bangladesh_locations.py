"""
Management command to seed Bangladesh location data (divisions, districts).
Adapted for Bunoraa i18n app.

Usage:
    python manage.py seed_bangladesh_locations
    python manage.py seed_bangladesh_locations --clear  # Clear existing data first
"""
import json
import os
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.db import transaction
from apps.i18n.models import Country, Division, District, Currency, Language, Timezone


class Command(BaseCommand):
    help = 'Seeds Bangladesh divisions, districts, and shipping zones'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing Bangladesh location data before seeding',
        )
        parser.add_argument(
            '--with-defaults',
            action='store_true',
            help='Also set up default language, currency, and timezone for Bangladesh',
        )

    def handle(self, *args, **options):
        fixtures_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'fixtures'
        )
        
        # Load fixture files
        divisions_file = os.path.join(fixtures_dir, 'bangladesh_divisions.json')
        districts_file = os.path.join(fixtures_dir, 'bangladesh_districts.json')
        
        if not os.path.exists(divisions_file):
            self.stderr.write(self.style.ERROR(f'Divisions file not found: {divisions_file}'))
            self.stdout.write('Creating fixtures directory structure...')
            os.makedirs(fixtures_dir, exist_ok=True)
            return
        
        if not os.path.exists(districts_file):
            self.stderr.write(self.style.ERROR(f'Districts file not found: {districts_file}'))
            return
        
        with open(divisions_file, 'r', encoding='utf-8') as f:
            divisions_data = json.load(f)
        
        with open(districts_file, 'r', encoding='utf-8') as f:
            districts_data = json.load(f)
        
        with transaction.atomic():
            # Set up defaults if requested
            default_language = None
            default_currency = None
            default_timezone = None
            
            if options['with_defaults']:
                default_language, default_currency, default_timezone = self._setup_defaults()
            
            # Get or create Bangladesh
            bangladesh, created = Country.objects.update_or_create(
                code='BD',
                defaults={
                    'name': 'Bangladesh',
                    'native_name': 'বাংলাদেশ',
                    'code_alpha3': 'BGD',
                    'code_numeric': '050',
                    'phone_code': '+880',
                    'phone_format': '+880 XXXX-XXXXXX',
                    'continent': 'asia',
                    'region': 'South Asia',
                    'default_language': default_language,
                    'default_currency': default_currency,
                    'default_timezone': default_timezone,
                    'default_tax_rate': Decimal('15.00'),  # 15% VAT
                    'is_active': True,
                    'is_shipping_available': True,
                    'is_billing_allowed': True,
                    'sort_order': 1,
                    'address_format': '{name}\n{address_line1}\n{address_line2}\n{upazila}, {district}\n{division} {postal_code}\n{country}',
                    'postal_code_format': r'^\d{4}$',
                }
            )
            
            if created:
                self.stdout.write(self.style.SUCCESS('Created Bangladesh country record'))
            else:
                self.stdout.write('Updated Bangladesh country record')
            
            if options['clear']:
                # Clear existing divisions (cascades to districts)
                Division.objects.filter(country=bangladesh).delete()
                self.stdout.write(self.style.WARNING('Cleared existing Bangladesh location data'))
            
            # Create divisions
            divisions_map = {}
            for div_data in divisions_data.get('divisions', []):
                division, created = Division.objects.update_or_create(
                    country=bangladesh,
                    code=div_data['code'],
                    defaults={
                        'name': div_data['name'],
                        'native_name': div_data.get('native_name', ''),
                        'division_type': 'division',
                        'latitude': Decimal(str(div_data.get('latitude', 0))) if div_data.get('latitude') else None,
                        'longitude': Decimal(str(div_data.get('longitude', 0))) if div_data.get('longitude') else None,
                        'is_active': True,
                        'is_shipping_available': True,
                        'sort_order': div_data.get('sort_order', 0),
                    }
                )
                divisions_map[div_data['code']] = division
                action = 'Created' if created else 'Updated'
                self.stdout.write(f'  {action} division: {division.name}')
            
            # Create districts
            for dist_data in districts_data.get('districts', []):
                division_code = dist_data.get('division')
                division = divisions_map.get(division_code)
                
                if not division:
                    self.stderr.write(
                        self.style.WARNING(f'Division {division_code} not found for district {dist_data["name"]}')
                    )
                    continue
                
                # Determine delivery estimates based on shipping zone
                shipping_zone = dist_data.get('shipping_zone', 'suburban')
                delivery_min, delivery_max = self._get_delivery_estimates(shipping_zone)
                
                district, created = District.objects.update_or_create(
                    division=division,
                    code=dist_data['code'],
                    defaults={
                        'name': dist_data['name'],
                        'native_name': dist_data.get('native_name', ''),
                        'latitude': Decimal(str(dist_data.get('latitude', 0))) if dist_data.get('latitude') else None,
                        'longitude': Decimal(str(dist_data.get('longitude', 0))) if dist_data.get('longitude') else None,
                        'shipping_zone': shipping_zone,
                        'estimated_delivery_days_min': delivery_min,
                        'estimated_delivery_days_max': delivery_max,
                        'is_active': True,
                        'is_shipping_available': True,
                    }
                )
                action = 'Created' if created else 'Updated'
                self.stdout.write(f'    {action} district: {district.name} ({district.shipping_zone})')
        
        # Summary
        division_count = Division.objects.filter(country=bangladesh).count()
        district_count = District.objects.filter(division__country=bangladesh).count()
        
        self.stdout.write(self.style.SUCCESS(
            f'\nSuccessfully seeded Bangladesh location data:'
            f'\n  - {division_count} divisions'
            f'\n  - {district_count} districts'
        ))
        
        # Print shipping zone breakdown
        self.stdout.write('\nShipping zone breakdown:')
        for zone in ['metro', 'suburban', 'rural', 'remote']:
            count = District.objects.filter(
                division__country=bangladesh,
                shipping_zone=zone
            ).count()
            if count > 0:
                self.stdout.write(f'  - {zone}: {count} districts')
    
    def _get_delivery_estimates(self, shipping_zone):
        """Get delivery time estimates based on shipping zone."""
        estimates = {
            'metro': (1, 2),      # Same day to next day
            'suburban': (2, 4),   # 2-4 days
            'rural': (4, 7),      # 4-7 days
            'remote': (7, 14),    # 7-14 days
        }
        return estimates.get(shipping_zone, (3, 5))
    
    def _setup_defaults(self):
        """Set up default language, currency, and timezone for Bangladesh."""
        self.stdout.write('Setting up Bangladesh defaults...')
        
        # Bengali language
        language, created = Language.objects.update_or_create(
            code='bn',
            defaults={
                'name': 'Bengali',
                'native_name': 'বাংলা',
                'is_rtl': False,
                'flag_code': 'BD',
                'locale_code': 'bn_BD',
                'is_active': True,
                'is_default': True,
                'sort_order': 0,
            }
        )
        if created:
            self.stdout.write(f'  Created language: {language.native_name}')
        
        # BDT currency
        currency, created = Currency.objects.update_or_create(
            code='BDT',
            defaults={
                'name': 'Bangladeshi Taka',
                'symbol': '৳',
                'native_symbol': '৳',
                'decimal_places': 2,
                'symbol_position': 'before',
                'thousand_separator': ',',
                'decimal_separator': '.',
                'number_system': 'bengali',
                'is_base_currency': True,
                'is_active': True,
                'is_default': True,
                'sort_order': 0,
            }
        )
        if created:
            self.stdout.write(f'  Created currency: {currency.symbol} {currency.name}')
        
        # Dhaka timezone
        timezone_obj, created = Timezone.objects.update_or_create(
            name='Asia/Dhaka',
            defaults={
                'display_name': 'Bangladesh Standard Time (BST)',
                'offset': '+06:00',
                'offset_minutes': 360,
                'has_dst': False,
                'is_active': True,
                'is_common': True,
            }
        )
        if created:
            self.stdout.write(f'  Created timezone: {timezone_obj.display_name}')
        
        return language, currency, timezone_obj
