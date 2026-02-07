"""
Management command to seed ShippingSettings.
"""
from decimal import Decimal
from django.core.management.base import BaseCommand
from apps.shipping.models import ShippingSettings


class Command(BaseCommand):
    help = 'Seeds the ShippingSettings singleton with Bangladesh defaults'

    def add_arguments(self, parser):
        parser.add_argument(
            '--threshold',
            type=float,
            default=3000,
            help='Free shipping threshold amount (default: 2000)'
        )
        parser.add_argument(
            '--handling-days',
            type=int,
            default=1,
            help='Order handling days (default: 1)'
        )
        parser.add_argument(
            '--enable-free-shipping',
            action='store_true',
            default=True,
            help='Enable free shipping above threshold'
        )

    def handle(self, *args, **options):
        threshold = options['threshold']
        handling_days = options['handling_days']
        enable_free_shipping = options['enable_free_shipping']

        settings, created = ShippingSettings.objects.update_or_create(
            pk=1,
            defaults={
                # Origin address (Bangladesh example)
                'origin_address_line1': 'House 123, Road 12',
                'origin_address_line2': 'Gulshan-1',
                'origin_city': 'Dhaka',
                'origin_state': 'Dhaka Division',
                'origin_postal_code': '1212',
                'origin_country': 'BD',
                'origin_phone': '+880 1700 000000',
                
                # Units
                'default_weight_unit': 'kg',
                'default_dimension_unit': 'cm',
                'default_package_weight': Decimal('0.45'),
                
                # Display
                'show_delivery_estimates': True,
                'show_carrier_logos': True,
                
                # Free shipping
                'enable_free_shipping': enable_free_shipping,
                'free_shipping_threshold': Decimal(str(threshold)),
                'free_shipping_countries': ['BD'],  # Bangladesh only
                
                # Processing
                'handling_days': handling_days,
                'cutoff_time': '15:00:00',  # 3 PM cutoff
            }
        )

        action = 'Created' if created else 'Updated'
        self.stdout.write(self.style.SUCCESS(
            f'{action} ShippingSettings: Free shipping threshold = ৳{threshold}'
        ))
