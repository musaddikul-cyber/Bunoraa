import random
import uuid # Added
from django.core.management.base import BaseCommand
from apps.notifications.models import BackInStockNotification
from apps.catalog.models import Product, ProductVariant
from django.contrib.auth import get_user_model

User = get_user_model()

class Command(BaseCommand):
    help = 'Seeds the database with sample BackInStockNotification data'

    def handle(self, *args, **options):
        self.stdout.write('Seeding BackInStockNotification data...')

        products = list(Product.objects.all())
        users = list(User.objects.filter(is_superuser=False))

        if not products:
            self.stdout.write('  No products available to create back-in-stock notifications.')
            return

        created_count = 0
        for product in random.sample(products, min(5, len(products))): # Create notifications for a few random products
            # Determine if it's a variant-specific or general product request
            if product.variants.exists() and random.random() > 0.5:
                variant = random.choice(list(product.variants.all()))
            else:
                variant = None
            
            # Determine requester (authenticated user or anonymous email)
            if users and random.random() > 0.5:
                user = random.choice(users)
                email = '' # Email not needed if user is authenticated
                requester_identifier = user.email
            else:
                user = None
                email = f"anon_{uuid.uuid4().hex[:8]}@example.com"
                requester_identifier = email
            
            # Ensure no duplicate requests
            if user:
                if BackInStockNotification.objects.filter(product=product, variant=variant, user=user).exists():
                    continue
            else:
                if BackInStockNotification.objects.filter(product=product, variant=variant, email=email).exists():
                    continue

            notification, created = BackInStockNotification.objects.get_or_create(
                product=product,
                variant=variant,
                user=user,
                defaults={
                    'email': email,
                    'is_notified': False,
                }
            )
            if created:
                created_count += 1
                self.stdout.write(f"  Created back-in-stock request for {product.name} ({variant or 'no variant'}) by {requester_identifier}.")

        self.stdout.write(self.style.SUCCESS(f'Successfully created {created_count} back-in-stock notifications.'))
