"""
Create API Key Management Command
==================================

Creates an API key for email service access.

Usage:
    python manage.py create_email_api_key --user <username> --name <key_name>
"""

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from apps.email_service.models import APIKey


class Command(BaseCommand):
    help = 'Create an API key for the email service'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--user',
            type=str,
            required=True,
            help='Username of the user to create the API key for'
        )
        parser.add_argument(
            '--name',
            type=str,
            default='Default API Key',
            help='Name/description for the API key'
        )
        parser.add_argument(
            '--permission',
            type=str,
            default='send',
            choices=['send', 'full', 'read'],
            help='Permission level for the API key'
        )
        parser.add_argument(
            '--rate-limit',
            type=int,
            default=100,
            help='Rate limit per minute'
        )
        parser.add_argument(
            '--daily-limit',
            type=int,
            default=10000,
            help='Daily sending limit'
        )
    
    def handle(self, *args, **options):
        User = get_user_model()
        
        try:
            user = User.objects.get(email=options['user'])
        except User.DoesNotExist:
            raise CommandError(f"User '{options['user']}' not found")
        
        # Create API key
        api_key, full_key = APIKey.create_key(
            user=user,
            name=options['name'],
            permission=options['permission'],
            rate_limit_per_minute=options['rate_limit'],
            rate_limit_per_day=options['daily_limit'],
        )
        
        self.stdout.write(self.style.SUCCESS('API Key created successfully!\n'))
        self.stdout.write(f'Key Prefix: {api_key.key_prefix}')
        self.stdout.write(f'Name: {api_key.name}')
        self.stdout.write(f'Permission: {api_key.permission}')
        self.stdout.write(f'Rate Limit: {api_key.rate_limit_per_minute}/min')
        self.stdout.write(f'Daily Limit: {api_key.rate_limit_per_day}/day\n')
        self.stdout.write(self.style.WARNING(
            '\n⚠️  IMPORTANT: Save this API key now! You will not be able to see it again.\n'
        ))
        self.stdout.write(self.style.SUCCESS(f'API Key: {full_key}'))
