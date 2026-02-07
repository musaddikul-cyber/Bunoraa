"""
Seed Email Service Command
===========================

Initialize email service with default data:
- API Keys
- Sender Domains
- Email Templates
- Unsubscribe Groups
- Webhooks

Usage:
    python manage.py seed_email_service
    python manage.py seed_email_service --user <username>
"""

import secrets
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model
from django.utils import timezone

from apps.email_service.models import (
    APIKey,
    SenderDomain,
    SenderIdentity,
    EmailTemplate,
    UnsubscribeGroup,
    Webhook,
)

User = get_user_model()


class Command(BaseCommand):
    help = 'Seed email service with default configuration'

    def add_arguments(self, parser):
        parser.add_argument(
            '--user',
            type=str,
            help='Email or ID of user to associate with the API key and templates'
        )
        parser.add_argument(
            '--skip-api-key',
            action='store_true',
            help='Skip creating API key'
        )
        parser.add_argument(
            '--skip-domain',
            action='store_true',
            help='Skip creating sender domain'
        )
        parser.add_argument(
            '--skip-templates',
            action='store_true',
            help='Skip creating email templates'
        )

    def handle(self, *args, **options):
        # Get user
        user_identifier = options.get('user')
        if user_identifier:
            try:
                # Try email first, then by ID
                user = User.objects.get(email=user_identifier)
            except User.DoesNotExist:
                try:
                    user = User.objects.get(id=user_identifier)
                except User.DoesNotExist:
                    raise CommandError(f'User "{user_identifier}" not found (by email or ID)')
        else:
            # Try to get the first superuser
            user = User.objects.filter(is_superuser=True).first()
            if not user:
                raise CommandError(
                    'No superuser found. Create one first or specify --user <email>'
                )
            self.stdout.write(f'Using user: {user.email}')

        # Seed API Key
        if not options['skip_api_key']:
            self.create_api_key(user)

        # Seed Sender Domain
        if not options['skip_domain']:
            self.create_sender_domain(user)

        # Seed Email Templates
        if not options['skip_templates']:
            self.create_email_templates(user)

        # Seed Unsubscribe Groups
        self.create_unsubscribe_groups(user)

        self.stdout.write(
            self.style.SUCCESS('✅ Email service seeded successfully!')
        )

    def create_api_key(self, user):
        """Create default API key"""
        # Check if key already exists
        if APIKey.objects.filter(name='Default API Key').exists():
            self.stdout.write(
                self.style.WARNING('⚠️  Default API Key already exists')
            )
            return

        api_key_obj, full_key = APIKey.create_key(
            user=user,
            name='Default API Key',
            permission=APIKey.Permission.FULL_ACCESS
        )
        self.stdout.write(
            self.style.SUCCESS(f'✅ Created API Key: {api_key_obj.name}')
        )
        self.stdout.write(
            f'   Key: {full_key}\n'
            f'   ⚠️  Save this key somewhere safe - you won\'t be able to see it again!'
        )

    def create_sender_domain(self, user):
        """Create default sender domain"""
        domain = 'bunoraa.com'

        # Check if domain already exists
        if SenderDomain.objects.filter(domain=domain).exists():
            self.stdout.write(
                self.style.WARNING(f'⚠️  Domain {domain} already exists')
            )
            return

        sender_domain = SenderDomain.objects.create(
            user=user,
            domain=domain,
            verification_status=SenderDomain.VerificationStatus.PENDING,
            verification_token=secrets.token_urlsafe(32),
            is_default=True,
            is_active=True,
            dkim_selector='bunoraa',
            dns_records={
                'spf': f'v=spf1 include:bunoraa.com ~all',
                'dkim': f'v=DKIM1; k=rsa; p=YOUR_DKIM_PUBLIC_KEY',
                'dmarc': f'v=DMARC1; p=quarantine; rua=mailto:admin@{domain}'
            }
        )

        self.stdout.write(
            self.style.SUCCESS(f'✅ Created Sender Domain: {domain}')
        )
        self.stdout.write(
            '\n📋 DNS Records to add:\n'
            f'   SPF:  {sender_domain.dns_records["spf"]}\n'
            f'   DKIM: {sender_domain.dns_records["dkim"]}\n'
            f'   DMARC: {sender_domain.dns_records["dmarc"]}\n'
        )

        # Create default sender identity
        self.create_sender_identity(user, sender_domain)

    def create_sender_identity(self, user, domain):
        """Create default sender identity"""
        identity = SenderIdentity.objects.create(
            user=user,
            domain=domain,
            email='noreply@bunoraa.com',
            from_name='Bunoraa Team',
            reply_to='support@bunoraa.com',
            is_default=True,
            is_active=True
        )
        self.stdout.write(
            self.style.SUCCESS(f'✅ Created Sender Identity: {identity.email}')
        )

    def create_email_templates(self, user):
        """Create default email templates"""
        templates = [
            {
                'name': 'Password Reset',
                'template_id': 'password-reset',
                'subject': 'Reset Your Bunoraa Password',
                'html_content': self.get_password_reset_template(),
                'description': 'Email for password reset requests'
            },
            {
                'name': 'Email Verification',
                'template_id': 'email-verification',
                'subject': 'Verify Your Email Address',
                'html_content': self.get_email_verification_template(),
                'description': 'Email for new account verification'
            },
            {
                'name': 'Welcome Email',
                'template_id': 'welcome',
                'subject': 'Welcome to Bunoraa!',
                'html_content': self.get_welcome_template(),
                'description': 'Welcome email for new customers'
            },
            {
                'name': 'Order Confirmation',
                'template_id': 'order-confirmation',
                'subject': 'Your Order #{{order_id}} Confirmed',
                'html_content': self.get_order_confirmation_template(),
                'description': 'Order confirmation email'
            },
            {
                'name': 'Order Shipped',
                'template_id': 'order-shipped',
                'subject': 'Your Order #{{order_id}} Has Shipped',
                'html_content': self.get_order_shipped_template(),
                'description': 'Order shipment notification'
            },
        ]

        created_count = 0
        for template_data in templates:
            if EmailTemplate.objects.filter(template_id=template_data['template_id']).exists():
                self.stdout.write(
                    self.style.WARNING(
                        f'⚠️  Template "{template_data["name"]}" already exists'
                    )
                )
                continue

            template = EmailTemplate.objects.create(
                user=user,
                name=template_data['name'],
                template_id=template_data['template_id'],
                subject=template_data['subject'],
                html_content=template_data['html_content'],
                description=template_data['description'],
                is_active=True
            )
            created_count += 1
            self.stdout.write(
                self.style.SUCCESS(f'✅ Created Template: {template.name}')
            )

        if created_count > 0:
            self.stdout.write(
                f'   Created {created_count} email template(s)'
            )

    def create_unsubscribe_groups(self, user):
        """Create default unsubscribe groups"""
        groups = [
            {
                'name': 'Marketing Emails',
                'description': 'Promotional emails and special offers',
                'is_default': True
            },
            {
                'name': 'Order Updates',
                'description': 'Order confirmations and shipment notifications',
                'is_default': False
            },
            {
                'name': 'Account Notifications',
                'description': 'Important account and security notifications',
                'is_default': False
            },
        ]

        created_count = 0
        for group_data in groups:
            if UnsubscribeGroup.objects.filter(name=group_data['name']).exists():
                self.stdout.write(
                    self.style.WARNING(
                        f'⚠️  Unsubscribe Group "{group_data["name"]}" already exists'
                    )
                )
                continue

            group = UnsubscribeGroup.objects.create(
                user=user,
                name=group_data['name'],
                description=group_data['description'],
                is_default=group_data['is_default']
            )
            created_count += 1
            self.stdout.write(
                self.style.SUCCESS(f'✅ Created Unsubscribe Group: {group.name}')
            )

        if created_count > 0:
            self.stdout.write(
                f'   Created {created_count} unsubscribe group(s)'
            )

    @staticmethod
    def get_password_reset_template():
        """HTML template for password reset emails"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: #f9f9f9; padding: 20px; border: 1px solid #e0e0e0; border-radius: 0 0 8px 8px; }
        .button { display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 4px; margin: 20px 0; }
        .footer { text-align: center; color: #999; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Password Reset</h1>
        </div>
        <div class="content">
            <p>Hi {{first_name}},</p>
            <p>We received a request to reset your password. Click the button below to set a new password:</p>
            <center>
                <a href="{{reset_link}}" class="button">Reset Password</a>
            </center>
            <p>Or copy and paste this link:</p>
            <p style="word-break: break-all; color: #666;">{{reset_link}}</p>
            <p style="color: #999; font-size: 12px;">This link will expire in 24 hours.</p>
            <hr>
            <p style="font-size: 12px; color: #999;">
                If you didn't request this, you can safely ignore this email.
            </p>
        </div>
        <div class="footer">
            <p>&copy; 2026 Bunoraa. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
        """

    @staticmethod
    def get_email_verification_template():
        """HTML template for email verification"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: #f9f9f9; padding: 20px; border: 1px solid #e0e0e0; border-radius: 0 0 8px 8px; }
        .button { display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 4px; margin: 20px 0; }
        .footer { text-align: center; color: #999; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Verify Your Email</h1>
        </div>
        <div class="content">
            <p>Hi {{first_name}},</p>
            <p>Thank you for signing up! Please verify your email address to complete your account setup:</p>
            <center>
                <a href="{{verify_link}}" class="button">Verify Email</a>
            </center>
            <p>Or copy and paste this code:</p>
            <p style="font-family: monospace; background: #fff; padding: 10px; border: 1px solid #e0e0e0; text-align: center; font-size: 18px;">{{verification_code}}</p>
            <hr>
            <p style="font-size: 12px; color: #999;">
                If you didn't create this account, you can safely ignore this email.
            </p>
        </div>
        <div class="footer">
            <p>&copy; 2026 Bunoraa. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
        """

    @staticmethod
    def get_welcome_template():
        """HTML template for welcome emails"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: #f9f9f9; padding: 20px; border: 1px solid #e0e0e0; border-radius: 0 0 8px 8px; }
        .button { display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 4px; margin: 20px 0; }
        .feature { margin: 15px 0; padding: 15px; background: white; border-left: 4px solid #667eea; }
        .footer { text-align: center; color: #999; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Welcome to Bunoraa! 🎉</h1>
        </div>
        <div class="content">
            <p>Hi {{first_name}},</p>
            <p>Welcome to Bunoraa! We're excited to have you on board.</p>
            
            <div class="feature">
                <strong>📦 Browse Products</strong><br>
                Explore our collection of high-quality products and find what you need.
            </div>
            
            <div class="feature">
                <strong>🛒 Easy Checkout</strong><br>
                Fast and secure checkout with multiple payment options.
            </div>
            
            <div class="feature">
                <strong>🚚 Fast Shipping</strong><br>
                Quick delivery to your doorstep with tracking information.
            </div>
            
            <center>
                <a href="{{shop_url}}" class="button">Start Shopping</a>
            </center>
            
            <p>If you have any questions, feel free to contact our support team.</p>
        </div>
        <div class="footer">
            <p>&copy; 2026 Bunoraa. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
        """

    @staticmethod
    def get_order_confirmation_template():
        """HTML template for order confirmation"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: #f9f9f9; padding: 20px; border: 1px solid #e0e0e0; border-radius: 0 0 8px 8px; }
        .order-info { background: white; padding: 15px; border: 1px solid #e0e0e0; margin: 15px 0; }
        .button { display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 4px; margin: 20px 0; }
        .footer { text-align: center; color: #999; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Order Confirmed ✓</h1>
        </div>
        <div class="content">
            <p>Hi {{first_name}},</p>
            <p>Thank you for your order! We're processing it now.</p>
            
            <div class="order-info">
                <strong>Order Number:</strong> {{order_id}}<br>
                <strong>Order Date:</strong> {{order_date}}<br>
                <strong>Total:</strong> {{total_amount}}
            </div>
            
            <center>
                <a href="{{order_url}}" class="button">View Order</a>
            </center>
            
            <p>You'll receive a shipping notification as soon as your items are on their way.</p>
            
            <hr>
            <p style="font-size: 12px; color: #999;">
                Questions? Contact our support team anytime.
            </p>
        </div>
        <div class="footer">
            <p>&copy; 2026 Bunoraa. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
        """

    @staticmethod
    def get_order_shipped_template():
        """HTML template for order shipped notification"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: #f9f9f9; padding: 20px; border: 1px solid #e0e0e0; border-radius: 0 0 8px 8px; }
        .shipping-info { background: white; padding: 15px; border: 1px solid #e0e0e0; margin: 15px 0; }
        .button { display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 4px; margin: 20px 0; }
        .footer { text-align: center; color: #999; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Your Order is On the Way! 🚚</h1>
        </div>
        <div class="content">
            <p>Hi {{first_name}},</p>
            <p>Great news! Your order has shipped and is on its way to you.</p>
            
            <div class="shipping-info">
                <strong>Tracking Number:</strong> {{tracking_number}}<br>
                <strong>Carrier:</strong> {{carrier}}<br>
                <strong>Estimated Delivery:</strong> {{estimated_delivery}}
            </div>
            
            <center>
                <a href="{{tracking_url}}" class="button">Track Your Order</a>
            </center>
            
            <p>You can track your package status using the link above.</p>
        </div>
        <div class="footer">
            <p>&copy; 2026 Bunoraa. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
        """
