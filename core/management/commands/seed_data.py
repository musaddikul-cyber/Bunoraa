"""
Management command to seed the database with sample data for development.
Usage: python manage.py seed_data
"""

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.utils.text import slugify
from django.utils import timezone
from decimal import Decimal
import random
import uuid
from django.core.management import call_command # Added

User = get_user_model()


class Command(BaseCommand):
    help = 'Seeds the database with sample data for development'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before seeding',
        )
        parser.add_argument(
            '--products',
            type=int,
            default=50,
            help='Number of products to create',
        )

    def handle(self, *args, **options):
        self.stdout.write('Starting database seeding...\n')

        if options['clear']:
            self.clear_data()
        
        # Initialize lists to hold created objects for later use
        self.users = list(User.objects.all()) # Existing users
        self.categories = []
        self.brands = []
        self.products = []
        self.artisans = []

        # self.create_users()
        # self.create_categories()
        # self.create_brands()
        # self.create_products(options['products'])
        # self.create_promotions()
        self.create_pages()
        self.create_reviews()

        # Seed new features
        call_command('seed_categories')
        call_command('seed_tags')
        call_command('seed_catalog_data')
        call_command('seed_artisans_data') 
        call_command('seed_contacts_data')
        call_command('seed_referral_data')
        call_command('seed_notifications_data')
        call_command('seed_chat_data')
        call_command('seed_email_service')
        call_command('seed_currencies')
        call_command('seed_bangladesh_locations')
        call_command('seed_shipping_settings')
        call_command('seed_bangladesh_shipping')
        call_command('seed_payment_gateways')

        self.stdout.write(self.style.SUCCESS('\nDatabase seeding completed successfully!'))

    def clear_data(self):
        """Clear existing data"""
        self.stdout.write('Clearing existing data...')
        
        from apps.catalog.models import (
            Product, Category, ProductMakingOf, CustomerPhoto,
            ProductQuestion, ProductAnswer
        )
        from apps.promotions.models import Promotion, Coupon
        from apps.pages.models import Page
        from apps.contacts.models import CustomizationRequest
        from apps.referral.models import ReferralCode, ReferralReward
        from apps.notifications.models import BackInStockNotification
        from apps.artisans.models import Artisan 
        from apps.chat.models import Conversation, Message 

        # Delete in order of dependencies
        Message.objects.all().delete() 
        Conversation.objects.all().delete() 
        BackInStockNotification.objects.all().delete()
        ReferralReward.objects.all().delete()
        ReferralCode.objects.all().delete()
        CustomizationRequest.objects.all().delete()
        ProductAnswer.objects.all().delete()
        ProductQuestion.objects.all().delete()
        CustomerPhoto.objects.all().delete()
        ProductMakingOf.objects.all().delete()
        Product.objects.all().delete()
        Category.objects.all().delete()
        Promotion.objects.all().delete()
        Coupon.objects.all().delete()
        Artisan.objects.all().delete() 
        Page.objects.all().delete()
        User.objects.filter(is_superuser=False).delete()
        self.stdout.write('Existing data cleared.\n')


    def create_pages(self):
        """Create sample CMS pages"""
        self.stdout.write('Creating pages...')

        from apps.pages.models import Page

        pages_data = [
            {
                'title': 'About Us',
                'slug': 'about-us',
                'content': '''
                    <h2>Welcome to Bunoraa</h2>
                    <p>Bunoraa is your premier destination for premium products at competitive prices. 
                    Founded in 2024, we've quickly become a trusted name in online retail.</p>
                    <h3>Our Mission</h3>
                    <p>To provide exceptional products with outstanding customer service, 
                    making quality accessible to everyone.</p>
                    <h3>Our Values</h3>
                    <ul>
                        <li>Quality First</li>
                        <li>Customer Satisfaction</li>
                        <li>Integrity & Transparency</li>
                        <li>Innovation</li>
                    </ul>
                ''',
            },
            {
                'title': 'Contact Us',
                'slug': 'contact-us',
                'content': '''
                    <h2>Get in Touch</h2>
                    <p>We'd love to hear from you! Contact us through any of the following methods:</p>
                    <h3>Customer Service</h3>
                    <p>Email: support@bunoraa.com<br>Phone: 1-800-BUNORAA</p>
                    <h3>Business Hours</h3>
                    <p>Monday - Friday: 9AM - 6PM EST<br>Saturday: 10AM - 4PM EST<br>Sunday: Closed</p>
                ''',
            },
            {
                'title': 'Shipping Information',
                'slug': 'shipping',
                'content': '''
                    <h2>Shipping Policy</h2>
                    <h3>Delivery Times</h3>
                    <ul>
                        <li>Standard Shipping: 5-7 business days</li>
                        <li>Express Shipping: 2-3 business days</li>
                        <li>Overnight Shipping: Next business day</li>
                    </ul>
                    <h3>Free Shipping</h3>
                    <p>Enjoy free standard shipping on all orders over $50!</p>
                ''',
            },
            {
                'title': 'Returns & Refunds',
                'slug': 'returns',
                'content': '''
                    <h2>Returns Policy</h2>
                    <p>We want you to be completely satisfied with your purchase. 
                    If you're not happy, you can return items within 30 days.</p>
                    <h3>How to Return</h3>
                    <ol>
                        <li>Log into your account</li>
                        <li>Navigate to your orders</li>
                        <li>Select the item to return</li>
                        <li>Print the return label</li>
                        <li>Ship the item back to us</li>
                    </ol>
                ''',
            },
        ]

        for page_data in pages_data:
            page, created = Page.objects.get_or_create(
                slug=page_data['slug'],
                defaults={
                    'title': page_data['title'],
                    'content': page_data['content'],
                    'is_published': True,
                }
            )
            if created:
                self.stdout.write(f"  Created page: {page.title}")

    def create_reviews(self):
        """Create sample reviews"""
        self.stdout.write('Creating reviews...')

        from apps.catalog.models import Product, Review, ProductQuestion, ProductAnswer 

        review_templates = [
            {'rating': 5, 'title': 'Excellent product!', 'content': 'Exceeded my expectations. Highly recommend!'},
            {'rating': 5, 'title': 'Perfect!', 'content': 'Exactly what I was looking for. Great quality.'},
            {'rating': 4, 'title': 'Very good', 'content': 'Good product, fast shipping. Minor issues but overall satisfied.'},
            {'rating': 4, 'title': 'Great value', 'content': 'Good quality for the price. Would buy again.'},
            {'rating': 3, 'title': 'Decent', 'content': 'Average product. Does the job but nothing special.'},
            {'rating': 5, 'title': 'Amazing!', 'content': 'Best purchase I\'ve made. Five stars!'},
        ]

        users = list(User.objects.filter(is_superuser=False))
        products = list(Product.objects.all()[:30])

        review_count = 0
        for product in products:
            num_reviews = random.randint(1, 5)
            for _ in range(num_reviews):
                if not users:
                    continue
                    
                template = random.choice(review_templates)
                user = random.choice(users)

                review, created = Review.objects.get_or_create(
                    product=product,
                    user=user,
                    defaults={
                        'rating': template['rating'],
                        'title': template['title'],
                        'content': template['content'],
                        'is_approved': True,
                    }
                )
                if created:
                    review_count += 1

        self.stdout.write(f'  Created {review_count} reviews')




