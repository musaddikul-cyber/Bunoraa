"""
Management command to seed Bangladesh shipping zones and rates.
"""
import json
import os
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.db import transaction
from apps.shipping.models import ShippingZone, ShippingMethod, ShippingRate, ShippingCarrier


class Command(BaseCommand):
    help = 'Seeds Bangladesh shipping zones, methods, carriers, and rates'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing shipping data before seeding',
        )

    def handle(self, *args, **options):
        fixtures_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'fixtures'
        )
        
        shipping_file = os.path.join(fixtures_dir, 'bangladesh_shipping.json')
        
        if not os.path.exists(shipping_file):
            self.stderr.write(self.style.ERROR(f'Shipping file not found: {shipping_file}'))
            return
        
        with open(shipping_file, 'r', encoding='utf-8') as f:
            shipping_data = json.load(f)
        
        with transaction.atomic():
            if options['clear']:
                ShippingRate.objects.all().delete()
                ShippingMethod.objects.all().delete()
                ShippingZone.objects.all().delete()
                ShippingCarrier.objects.all().delete()
                self.stdout.write(self.style.WARNING('Cleared existing shipping data'))
            
            # Create shipping carriers first
            carriers_map = {}
            for carrier_data in shipping_data.get('shipping_carriers', []):
                carrier, created = ShippingCarrier.objects.update_or_create(
                    code=carrier_data['code'],
                    defaults={
                        'name': carrier_data['name'],
                        'website': carrier_data.get('website', ''),
                        'tracking_url_template': carrier_data.get('tracking_url_template', ''),
                        'is_active': carrier_data.get('is_active', True),
                    }
                )
                carriers_map[carrier_data['code']] = carrier
                action = 'Created' if created else 'Updated'
                self.stdout.write(f'  {action} shipping carrier: {carrier.name}')
            
            # Create shipping methods (link to default carrier if specified)
            methods_map = {}
            for method_data in shipping_data.get('shipping_methods', []):
                carrier = None
                carrier_code = method_data.get('carrier')
                if carrier_code and carrier_code in carriers_map:
                    carrier = carriers_map[carrier_code]
                
                method, created = ShippingMethod.objects.update_or_create(
                    code=method_data['code'],
                    defaults={
                        'name': method_data['name'],
                        'description': method_data.get('description', ''),
                        'min_delivery_days': method_data.get('min_delivery_days', 3),
                        'max_delivery_days': method_data.get('max_delivery_days', 7),
                        'delivery_time_text': method_data.get('delivery_time_text', ''),
                        'is_express': method_data.get('is_express', False),
                        'is_active': True,
                        'carrier': carrier,
                    }
                )
                methods_map[method_data['code']] = method
                action = 'Created' if created else 'Updated'
                self.stdout.write(f'  {action} shipping method: {method.name}')
            
            # Create shipping zones
            zones_map = {}
            for zone_data in shipping_data.get('shipping_zones', []):
                # Build countries list
                countries = zone_data.get('countries', [])
                states = zone_data.get('states', [])
                cities = zone_data.get('cities', [])
                postal_codes = zone_data.get('postal_codes', [])
                
                if not countries:
                    countries = ['BD']  # Default to Bangladesh
                
                zone, created = ShippingZone.objects.update_or_create(
                    name=zone_data['name'],
                    defaults={
                        'description': zone_data.get('description', ''),
                        'countries': countries,
                        'states': states,
                        'cities': cities,
                        'postal_codes': postal_codes,
                        'is_active': True,
                        'is_default': zone_data.get('is_default', False),
                        'priority': zone_data.get('priority', 0),
                    }
                )
                zones_map[zone_data['code']] = zone
                action = 'Created' if created else 'Updated'
                self.stdout.write(f'  {action} shipping zone: {zone.name}')
            
            # Create shipping rates
            for rate_data in shipping_data.get('shipping_rates', []):
                zone_code = rate_data.get('zone')
                method_code = rate_data.get('method')
                
                zone = zones_map.get(zone_code)
                method = methods_map.get(method_code)
                
                if not zone or not method:
                    self.stderr.write(
                        self.style.WARNING(f'Zone or method not found for rate: {zone_code}/{method_code}')
                    )
                    continue
                
                rate, created = ShippingRate.objects.update_or_create(
                    zone=zone,
                    method=method,
                    defaults={
                        'rate_type': rate_data.get('rate_type', 'flat'),
                        'base_rate': Decimal(str(rate_data.get('base_rate', 0))),
                        'free_shipping_threshold': Decimal(str(rate_data.get('free_shipping_threshold', 0))) if rate_data.get('free_shipping_threshold') else None,
                        'is_active': True,
                    }
                )
                action = 'Created' if created else 'Updated'
                self.stdout.write(f'    {action} rate: {zone.name} + {method.name} = ৳{rate.base_rate}')
        
        # Summary
        carrier_count = ShippingCarrier.objects.count()
        zone_count = ShippingZone.objects.count()
        method_count = ShippingMethod.objects.count()
        rate_count = ShippingRate.objects.count()
        
        self.stdout.write(self.style.SUCCESS(
            f'\nSuccessfully seeded shipping data:'
            f'\n  - {carrier_count} carriers'
            f'\n  - {zone_count} zones'
            f'\n  - {method_count} methods'
            f'\n  - {rate_count} rates'
        ))
