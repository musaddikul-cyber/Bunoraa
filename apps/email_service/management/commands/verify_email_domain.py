"""
Verify Domain Command
======================

Verify DNS records for a sender domain.

Usage:
    python manage.py verify_email_domain example.com
"""

import dns.resolver

from django.core.management.base import BaseCommand, CommandError

from apps.email_service.models import SenderDomain


class Command(BaseCommand):
    help = 'Verify DNS records for an email sender domain'
    
    def add_arguments(self, parser):
        parser.add_argument(
            'domain',
            type=str,
            help='Domain name to verify'
        )
        parser.add_argument(
            '--show-records',
            action='store_true',
            help='Show required DNS records'
        )
    
    def handle(self, *args, **options):
        domain_name = options['domain']
        
        try:
            domain = SenderDomain.objects.get(domain=domain_name)
        except SenderDomain.DoesNotExist:
            raise CommandError(f"Domain '{domain_name}' not found")
        
        self.stdout.write(f'\nVerifying domain: {domain_name}\n')
        self.stdout.write('=' * 50)
        
        # Check verification TXT record
        self.stdout.write('\n1. Verification TXT Record')
        self._check_txt_record(domain)
        
        # Check SPF record
        self.stdout.write('\n2. SPF Record')
        self._check_spf_record(domain)
        
        # Check DKIM record
        self.stdout.write('\n3. DKIM Record')
        self._check_dkim_record(domain)
        
        # Summary
        self.stdout.write('\n' + '=' * 50)
        self.stdout.write('\nVerification Summary:')
        self.stdout.write(f'  SPF: {"✅" if domain.spf_verified else "❌"}')
        self.stdout.write(f'  DKIM: {"✅" if domain.dkim_verified else "❌"}')
        self.stdout.write(f'  DMARC: {"✅" if domain.dmarc_verified else "❌"}')
        
        if options['show_records']:
            self._show_required_records(domain)
    
    def _check_txt_record(self, domain):
        """Check domain verification TXT record."""
        try:
            answers = dns.resolver.resolve(domain.domain, 'TXT')
            expected = f'bunoraa-verification={domain.verification_token}'
            
            for record in answers:
                txt_value = str(record).strip('"')
                if expected in txt_value:
                    domain.is_verified = True
                    domain.save()
                    self.stdout.write(self.style.SUCCESS(f'  ✅ Found: {txt_value}'))
                    return
            
            self.stdout.write(self.style.ERROR(f'  ❌ Not found'))
            self.stdout.write(f'  Expected: {expected}')
        except dns.resolver.NXDOMAIN:
            self.stdout.write(self.style.ERROR(f'  ❌ Domain not found'))
        except dns.resolver.NoAnswer:
            self.stdout.write(self.style.ERROR(f'  ❌ No TXT records'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'  ❌ Error: {e}'))
    
    def _check_spf_record(self, domain):
        """Check SPF record."""
        try:
            answers = dns.resolver.resolve(domain.domain, 'TXT')
            expected_includes = ['include:_spf.bunoraa.com', 'include:bunoraa.com']
            
            for record in answers:
                txt_value = str(record).strip('"')
                if txt_value.startswith('v=spf1'):
                    self.stdout.write(f'  Found: {txt_value}')
                    for inc in expected_includes:
                        if inc in txt_value:
                            domain.spf_verified = True
                            domain.save()
                            self.stdout.write(self.style.SUCCESS(f'  ✅ SPF includes Bunoraa'))
                            return
                    
                    self.stdout.write(self.style.WARNING(
                        f'  ⚠️  SPF exists but doesn\'t include Bunoraa'
                    ))
                    return
            
            self.stdout.write(self.style.ERROR(f'  ❌ No SPF record'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'  ❌ Error: {e}'))
    
    def _check_dkim_record(self, domain):
        """Check DKIM record."""
        dkim_domain = f'{domain.dkim_selector}._domainkey.{domain.domain}'
        
        try:
            answers = dns.resolver.resolve(dkim_domain, 'TXT')
            
            for record in answers:
                txt_value = str(record).strip('"')
                if 'v=DKIM1' in txt_value and domain.dkim_public_key[:20] in txt_value:
                    domain.dkim_verified = True
                    domain.save()
                    self.stdout.write(self.style.SUCCESS(f'  ✅ DKIM record found'))
                    return
            
            self.stdout.write(self.style.ERROR(f'  ❌ DKIM record mismatch'))
        except dns.resolver.NXDOMAIN:
            self.stdout.write(self.style.ERROR(f'  ❌ DKIM record not found'))
            self.stdout.write(f'  Check: {dkim_domain}')
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'  ❌ Error: {e}'))
    
    def _show_required_records(self, domain):
        """Display required DNS records."""
        self.stdout.write('\n' + '=' * 50)
        self.stdout.write('\nRequired DNS Records:\n')
        
        # Verification TXT
        self.stdout.write('1. Verification TXT Record:')
        self.stdout.write(f'   Host: {domain.domain}')
        self.stdout.write(f'   Type: TXT')
        self.stdout.write(f'   Value: bunoraa-verification={domain.verification_token}')
        
        # SPF
        self.stdout.write('\n2. SPF Record (add to existing or create):')
        self.stdout.write(f'   Host: {domain.domain}')
        self.stdout.write(f'   Type: TXT')
        self.stdout.write(f'   Value: v=spf1 include:_spf.bunoraa.com ~all')
        
        # DKIM
        if domain.dkim_public_key:
            self.stdout.write('\n3. DKIM Record:')
            self.stdout.write(f'   Host: {domain.dkim_selector}._domainkey.{domain.domain}')
            self.stdout.write(f'   Type: TXT')
            self.stdout.write(f'   Value: v=DKIM1; k=rsa; p={domain.dkim_public_key}')
        
        # DMARC
        self.stdout.write('\n4. DMARC Record (recommended):')
        self.stdout.write(f'   Host: _dmarc.{domain.domain}')
        self.stdout.write(f'   Type: TXT')
        self.stdout.write(f'   Value: v=DMARC1; p=quarantine; rua=mailto:dmarc@{domain.domain}')
