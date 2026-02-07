"""
Management command to seed default payment gateways.
"""
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.db import transaction
from apps.payments.models import PaymentGateway


class Command(BaseCommand):
    help = 'Seeds default payment gateways'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing payment gateways before seeding',
        )

    def handle(self, *args, **options):
        gateways_data = [
            {
                'code': 'stripe',
                'name': 'Credit/Debit Card',
                'description': 'Visa, Mastercard, American Express',
                'icon_class': 'card',
                'color': 'blue',
                'fee_type': 'none',
                'fee_amount': Decimal('0'),
                'fee_text': 'No extra fee',
                'is_active': True,
                'is_sandbox': True,
                'currencies': [],  # All currencies
                'countries': [],  # All countries
                'instructions': 'Your card will be charged securely via Stripe.',
                'sort_order': 1,
            },
            {
                'code': 'bkash',
                'name': 'bKash',
                'description': 'Pay with your bKash wallet',
                'icon_class': 'bkash',
                'color': 'pink',
                'fee_type': 'none',
                'fee_amount': Decimal('0'),
                'fee_text': 'No extra fee',
                'is_active': True,
                'is_sandbox': True,
                'currencies': ['BDT'],
                'countries': ['BD'],
                'instructions': 'You will be redirected to bKash to complete your payment.',
                'sort_order': 2,
            },
            {
                'code': 'nagad',
                'name': 'Nagad',
                'description': 'Pay with your Nagad wallet',
                'icon_class': 'nagad',
                'color': 'orange',
                'fee_type': 'none',
                'fee_amount': Decimal('0'),
                'fee_text': 'No extra fee',
                'is_active': True,
                'is_sandbox': True,
                'currencies': ['BDT'],
                'countries': ['BD'],
                'instructions': 'You will be redirected to Nagad to complete your payment.',
                'sort_order': 3,
            },
            {
                'code': 'cod',
                'name': 'Cash on Delivery',
                'description': 'Pay when you receive your order',
                'icon_class': 'cash',
                'color': 'green',
                'fee_type': 'none',
                'fee_amount': Decimal('0'),
                'fee_text': 'No extra fee',
                'is_active': True,
                'is_sandbox': False,
                'currencies': ['BDT'],
                'countries': ['BD'],
                'instructions': 'Please have exact cash ready for the delivery person.',
                'sort_order': 4,
            },
            {
                'code': 'bank_transfer',
                'name': 'Bank Transfer',
                'description': 'Direct bank transfer',
                'icon_class': 'bank',
                'color': 'gray',
                'fee_type': 'none',
                'fee_amount': Decimal('0'),
                'fee_text': 'No extra fee',
                'is_active': True,
                'is_sandbox': False,
                'currencies': ['BDT'],
                'countries': ['BD'],
                'bank_name': 'Dutch Bangla Bank',
                'bank_account_name': 'Bunoraa',
                'bank_account_number': '1234567890',
                'bank_routing_number': '090123456',
                'bank_branch': 'Dhaka Main Branch',
                'instructions': 'Transfer the total amount to our bank account. Your order will be processed after payment confirmation.',
                'sort_order': 5,
            },
        ]
        
        with transaction.atomic():
            if options['clear']:
                PaymentGateway.objects.all().delete()
                self.stdout.write(self.style.WARNING('Cleared existing payment gateways'))
            
            for gateway_data in gateways_data:
                gateway, created = PaymentGateway.objects.update_or_create(
                    code=gateway_data['code'],
                    defaults=gateway_data
                )
                action = 'Created' if created else 'Updated'
                self.stdout.write(f'  {action} gateway: {gateway.name}')
        
        count = PaymentGateway.objects.filter(is_active=True).count()
        self.stdout.write(self.style.SUCCESS(f'\nSuccessfully seeded {count} payment gateways'))
