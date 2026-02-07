"""
Send Test Email Command
========================

Send a test email to verify email service configuration.

Usage:
    python manage.py send_test_email --to <email> --api-key <key>
"""

from django.core.management.base import BaseCommand, CommandError

from apps.email_service.models import APIKey, EmailMessage


class Command(BaseCommand):
    help = 'Send a test email to verify email service configuration'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--to',
            type=str,
            required=True,
            help='Email address to send test email to'
        )
        parser.add_argument(
            '--api-key',
            type=str,
            help='API key to use (optional, uses first active key)'
        )
        parser.add_argument(
            '--subject',
            type=str,
            default='Test Email from Bunoraa Email Service',
            help='Subject line'
        )
        parser.add_argument(
            '--from-email',
            type=str,
            default='test@bunoraa.com',
            help='From email address'
        )
    
    def handle(self, *args, **options):
        # Get API key
        if options['api_key']:
            api_key = APIKey.verify_key(options['api_key'])
            if not api_key:
                raise CommandError("Invalid API key")
        else:
            api_key = APIKey.objects.filter(is_active=True).first()
            if not api_key:
                raise CommandError("No active API keys found. Create one first.")
        
        self.stdout.write(f'Using API key: {api_key.name}')
        
        # Create test email
        html_content = """
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h1 style="color: #2c3e50;">🎉 Test Email Successful!</h1>
            <p>Congratulations! Your Bunoraa Email Service is working correctly.</p>
            <hr>
            <h3>Email Service Features:</h3>
            <ul>
                <li>✅ Email Queue Processing</li>
                <li>✅ Open & Click Tracking</li>
                <li>✅ Bounce Handling</li>
                <li>✅ Webhook Delivery</li>
                <li>✅ Template Management</li>
                <li>✅ Domain Verification</li>
            </ul>
            <hr>
            <p style="color: #7f8c8d; font-size: 12px;">
                This is a test email from Bunoraa Email Service.
            </p>
        </body>
        </html>
        """
        
        text_content = """
        Test Email Successful!
        
        Congratulations! Your Bunoraa Email Service is working correctly.
        
        Email Service Features:
        - Email Queue Processing
        - Open & Click Tracking
        - Bounce Handling
        - Webhook Delivery
        - Template Management
        - Domain Verification
        
        This is a test email from Bunoraa Email Service.
        """
        
        message = EmailMessage.objects.create(
            user=api_key.user,
            api_key=api_key,
            from_email=options['from_email'],
            from_name='Bunoraa Email Service',
            to_email=options['to'],
            subject=options['subject'],
            html_body=html_content,
            text_body=text_content,
            status='queued',
        )
        
        self.stdout.write(self.style.SUCCESS(
            f'\n✅ Test email queued successfully!'
        ))
        self.stdout.write(f'Message ID: {message.message_id}')
        self.stdout.write(f'To: {options["to"]}')
        self.stdout.write(f'Subject: {options["subject"]}')
        self.stdout.write(
            '\nThe email will be sent when the queue processor runs.'
        )
        self.stdout.write(
            'Run "celery -A core worker" to process the queue.'
        )
