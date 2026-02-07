import random
from django.core.management.base import BaseCommand
from django.utils.text import slugify
from apps.artisans.models import Artisan

class Command(BaseCommand):
    help = 'Seeds the database with sample Artisan data'

    def handle(self, *args, **options):
        self.stdout.write('Seeding Artisan data...')

        artisans_data = [
            {'name': 'Bunoraa', 'bio': 'Specializes in Nakshi Kantha embroidery, a traditional art form of Bangladesh.', 'website': 'http://bunoraa.com', 'instagram': 'http://instagram.com/bunoraa_bd'},
        ]

        created_count = 0
        for data in artisans_data:
            artisan, created = Artisan.objects.get_or_create(
                name=data['name'],
                defaults={
                    'bio': data['bio'],
                    'website': data['website'],
                    'instagram': data['instagram'],
                    'is_active': True,
                }
            )
            if created:
                created_count += 1
                self.stdout.write(f"  Created artisan: {artisan.name}")

        self.stdout.write(self.style.SUCCESS(f'Successfully created {created_count} artisans.'))