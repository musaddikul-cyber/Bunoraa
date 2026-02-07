import random
from django.core.management.base import BaseCommand
from django.utils import timezone
from apps.contacts.models import CustomizationRequest
from apps.catalog.models import Product
from django.contrib.auth import get_user_model

User = get_user_model()

class Command(BaseCommand):
    help = 'Seeds the database with sample CustomizationRequest data'

    def handle(self, *args, **options):
        self.stdout.write('Seeding CustomizationRequest data...')

        products = list(Product.objects.filter(can_be_customized=True))
        users = list(User.objects.filter(is_superuser=False))

        if not products:
            self.stdout.write('  No customizable products available.')
            return
        if not users:
            self.stdout.write('  No users available to create customization requests.')
            return

        created_count = 0
        for _ in range(min(10, len(products) * len(users))): # Create up to 10 requests
            product = random.choice(products)
            user = random.choice(users)
            
            message = f"I would like to customize the '{product.name}' with " \
                      f"{random.choice(['blue thread', 'my initials', 'a different pattern'])}"
            
            request, created = CustomizationRequest.objects.get_or_create(
                product=product,
                user=user,
                defaults={
                    'name': user.get_full_name(),
                    'email': user.email,
                    'phone': user.phone or '',
                    'message': message,
                    'status': random.choice(['new', 'in_progress']),
                }
            )
            if created:
                created_count += 1
                self.stdout.write(f"  Created customization request for {product.name} by {user.email}")

        self.stdout.write(self.style.SUCCESS(f'Successfully created {created_count} customization requests.'))
