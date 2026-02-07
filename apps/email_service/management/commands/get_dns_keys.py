"""
Get DNS Keys Command
====================

Get all the keys and values needed for DNS configuration.

Usage:
    python manage.py get_dns_keys bunoraa.com
    python manage.py get_dns_keys bunoraa.com --generate-dkim
"""

import os
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from django.core.management.base import BaseCommand, CommandError
from apps.email_service.models import SenderDomain


class Command(BaseCommand):
    help = 'Get DNS keys and values for domain configuration'
    
    def add_arguments(self, parser):
        parser.add_argument(
            'domain',
            type=str,
            nargs='?',
            default='bunoraa.com',
            help='Domain name (default: bunoraa.com)'
        )
        parser.add_argument(
            '--generate-dkim',
            action='store_true',
            help='Generate new DKIM keys if missing'
        )
    
    def handle(self, *args, **options):
        domain_name = options['domain']
        generate_dkim = options['generate_dkim']
        
        try:
            domain = SenderDomain.objects.get(domain=domain_name)
        except SenderDomain.DoesNotExist:
            raise CommandError(f"Domain '{domain_name}' not found")
        
        # Generate DKIM keys if needed
        if generate_dkim and not domain.dkim_public_key:
            self.generate_dkim_keys(domain)
        
        self.stdout.write(self.style.SUCCESS(f'\n🔑 DNS Keys for {domain_name}\n'))
        self.stdout.write('=' * 80)
        
        # Verification Token
        self.stdout.write('\n1️⃣  VERIFICATION TXT RECORD')
        self.stdout.write('   Add this to verify your domain ownership\n')
        self.stdout.write(f'   Host/Name: _bunoraa.{domain_name}')
        self.stdout.write(f'   Type:      TXT')
        self.stdout.write(f'   Value:     bunoraa-verification={domain.verification_token}\n')
        
        # SPF Record
        self.stdout.write('2️⃣  SPF TXT RECORD')
        self.stdout.write('   Allows Bunoraa to send emails from your domain\n')
        self.stdout.write(f'   Host/Name: {domain_name}')
        self.stdout.write(f'   Type:      TXT')
        self.stdout.write(f'   Value:     v=spf1 include:bunoraa.com ~all\n')
        
        # DKIM Record
        self.stdout.write('3️⃣  DKIM TXT RECORD')
        self.stdout.write('   Signs emails with your domain\'s private key\n')
        self.stdout.write(f'   Host/Name: {domain.dkim_selector}._domainkey.{domain_name}')
        self.stdout.write(f'   Type:      TXT')
        
        if domain.dkim_public_key:
            self.stdout.write(f'   Value:     {domain.dkim_public_key}\n')
        else:
            self.stdout.write(f'   Value:     (not generated yet)\n')
            self.stdout.write(self.style.WARNING('   ⚠️  Run with --generate-dkim to create DKIM keys'))
            self.stdout.write('\n')
        
        # DMARC Record (Optional)
        self.stdout.write('4️⃣  DMARC TXT RECORD (Optional)')
        self.stdout.write('   Sets policy for handling emails that fail SPF/DKIM\n')
        self.stdout.write(f'   Host/Name: _dmarc.{domain_name}')
        self.stdout.write(f'   Type:      TXT')
        self.stdout.write(f'   Value:     v=DMARC1; p=quarantine; rua=mailto:admin@{domain_name}\n')
        
        self.stdout.write('=' * 80)
        self.stdout.write('\n📋 QUICK COPY-PASTE FOR CLOUDFLARE:\n')
        self.stdout.write(self.style.SUCCESS('Record 1 (Verification):'))
        self.stdout.write(f'  Name:  _bunoraa')
        self.stdout.write(f'  Value: bunoraa-verification={domain.verification_token}\n')
        
        self.stdout.write(self.style.SUCCESS('Record 2 (SPF):'))
        self.stdout.write(f'  Name:  {domain_name}')
        self.stdout.write(f'  Value: v=spf1 include:bunoraa.com ~all\n')
        
        if domain.dkim_public_key:
            self.stdout.write(self.style.SUCCESS('Record 3 (DKIM):'))
            self.stdout.write(f'  Name:  {domain.dkim_selector}._domainkey')
            self.stdout.write(f'  Value: {domain.dkim_public_key}\n')
        
        self.stdout.write('=' * 80)
        self.stdout.write('\n')
    
    def generate_dkim_keys(self, domain):
        """Generate DKIM RSA key pair"""
        self.stdout.write(self.style.WARNING('\n🔐 Generating DKIM keys... This may take a moment...\n'))
        
        # Generate RSA key pair (2048 bits)
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Get public key in PEM format
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        # Extract just the key part (without header/footer and newlines)
        public_key_base64 = public_pem.split('\n')[1:-2]
        public_key_b64 = ''.join(public_key_base64)
        
        # Get private key in PEM format
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        
        # Format public key for DNS
        dkim_public_key = f'v=DKIM1; k=rsa; p={public_key_b64}'
        
        # Save to database
        domain.dkim_public_key = dkim_public_key
        domain.dkim_private_key = private_pem
        domain.save(update_fields=['dkim_public_key', 'dkim_private_key'])
        
        self.stdout.write(self.style.SUCCESS('✅ DKIM keys generated and saved!\n'))
