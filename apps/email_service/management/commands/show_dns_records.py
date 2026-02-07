"""
Show DNS Records Command
========================

Show the DNS records that need to be added to verify a sender domain.

Usage:
    python manage.py show_dns_records bunoraa.com
"""

from django.core.management.base import BaseCommand, CommandError
from apps.email_service.models import SenderDomain


class Command(BaseCommand):
    help = 'Show DNS records required for domain verification'
    
    def add_arguments(self, parser):
        parser.add_argument(
            'domain',
            type=str,
            nargs='?',
            default='bunoraa.com',
            help='Domain name (default: bunoraa.com)'
        )
    
    def handle(self, *args, **options):
        domain_name = options['domain']
        
        try:
            domain = SenderDomain.objects.get(domain=domain_name)
        except SenderDomain.DoesNotExist:
            raise CommandError(f"Domain '{domain_name}' not found")
        
        self.stdout.write(self.style.SUCCESS(f'\n📋 DNS Records for {domain_name}\n'))
        self.stdout.write('=' * 70)
        
        # SPF Record
        self.stdout.write('\nSPF Record:')
        self.stdout.write(f'  Type:  TXT')
        self.stdout.write(f'  Name:  {domain_name}')
        self.stdout.write(f'  Value: {domain.dns_records.get("spf", "v=spf1 include:bunoraa.com ~all")}')
        
        # DKIM Record
        self.stdout.write('\nDKIM Record:')
        self.stdout.write(f'  Type:  TXT')
        self.stdout.write(f'  Name:  {domain.dkim_selector}._domainkey.{domain_name}')
        self.stdout.write(f'  Value: {domain.dns_records.get("dkim", "v=DKIM1; k=rsa; p=YOUR_DKIM_PUBLIC_KEY")}')
        
        # DMARC Record
        self.stdout.write('\nDMARC Record:')
        self.stdout.write(f'  Type:  TXT')
        self.stdout.write(f'  Name:  _dmarc.{domain_name}')
        self.stdout.write(f'  Value: {domain.dns_records.get("dmarc", "v=DMARC1; p=quarantine; rua=mailto:admin@bunoraa.com")}')
        
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.WARNING(
            '\n⚠️  Add these records to your domain registrar (Cloudflare, GoDaddy, etc.)'
        ))
        self.stdout.write(f'   Status: {domain.get_verification_status_display()}')
        self.stdout.write(f'   SPF Verified: {domain.spf_verified}')
        self.stdout.write(f'   DKIM Verified: {domain.dkim_verified}')
        self.stdout.write(f'   DMARC Verified: {domain.dmarc_verified}')
        self.stdout.write('\n')
