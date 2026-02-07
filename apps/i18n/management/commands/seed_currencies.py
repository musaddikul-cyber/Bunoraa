"""
Unified management command to seed currencies and exchange rates.

Combines functionality from:
  - add_currency
  - seed_exchange_rates
  - update_exchange_rates

Usage:
  python manage.py seed_currencies                    # Seed all common currencies with default rates
  python manage.py seed_currencies --base=BDT         # Use BDT as base currency (default)
  python manage.py seed_currencies --update           # Update existing rates
  python manage.py seed_currencies --fetch            # Fetch live rates from API
  python manage.py seed_currencies --provider=ecb     # Use specific provider for live rates
  python manage.py seed_currencies --list             # List all available currencies
  python manage.py seed_currencies USD                # Add/update single currency
  python manage.py seed_currencies USD EUR GBP        # Add/update multiple currencies
"""
import requests
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from django.conf import settings

from apps.i18n.models import Currency, ExchangeRate


# =============================================================================
# CURRENCY DATA
# =============================================================================

# Format: code -> (name, symbol, native_symbol, decimal_places, symbol_position, number_system)
CURRENCIES = {
    # South Asian currencies (primary markets)
    'BDT': ('Bangladeshi Taka', '৳', '৳', 2, 'before', 'bengali'),
    'INR': ('Indian Rupee', '₹', '₹', 2, 'before', 'indian'),
    'PKR': ('Pakistani Rupee', 'Rs', '₨', 2, 'before', 'indian'),
    'NPR': ('Nepalese Rupee', 'Rs', 'रू', 2, 'before', 'indian'),
    'LKR': ('Sri Lankan Rupee', 'Rs', 'රු', 2, 'before', 'western'),
    
    # Major world currencies
    'USD': ('US Dollar', '$', '$', 2, 'before', 'western'),
    'EUR': ('Euro', '€', '€', 2, 'before', 'western'),
    'GBP': ('British Pound', '£', '£', 2, 'before', 'western'),
    'JPY': ('Japanese Yen', '¥', '¥', 0, 'before', 'western'),
    'CAD': ('Canadian Dollar', 'C$', 'C$', 2, 'before', 'western'),
    'AUD': ('Australian Dollar', 'A$', 'A$', 2, 'before', 'western'),
    'CHF': ('Swiss Franc', 'CHF', 'CHF', 2, 'before', 'western'),
    'CNY': ('Chinese Yuan', '¥', '¥', 2, 'before', 'western'),
    
    # Southeast Asian currencies
    'SGD': ('Singapore Dollar', 'S$', 'S$', 2, 'before', 'western'),
    'MYR': ('Malaysian Ringgit', 'RM', 'RM', 2, 'before', 'western'),
    'THB': ('Thai Baht', '฿', '฿', 2, 'before', 'western'),
    'PHP': ('Philippine Peso', '₱', '₱', 2, 'before', 'western'),
    'IDR': ('Indonesian Rupiah', 'Rp', 'Rp', 0, 'before', 'western'),
    'VND': ('Vietnamese Dong', '₫', '₫', 0, 'after', 'western'),
    
    # Middle Eastern currencies
    'AED': ('UAE Dirham', 'AED', 'د.إ', 2, 'after', 'western'),
    'SAR': ('Saudi Riyal', 'SAR', '﷼', 2, 'after', 'western'),
    'QAR': ('Qatari Riyal', 'QAR', '﷼', 2, 'after', 'western'),
    'KWD': ('Kuwaiti Dinar', 'KD', 'د.ك', 3, 'before', 'western'),
    'BHD': ('Bahraini Dinar', 'BD', '.د.ب', 3, 'before', 'western'),
    'OMR': ('Omani Rial', 'OMR', '﷼', 3, 'before', 'western'),
    
    # Other currencies
    'HKD': ('Hong Kong Dollar', 'HK$', 'HK$', 2, 'before', 'western'),
    'KRW': ('South Korean Won', '₩', '₩', 0, 'before', 'western'),
    'NZD': ('New Zealand Dollar', 'NZ$', 'NZ$', 2, 'before', 'western'),
    'SEK': ('Swedish Krona', 'kr', 'kr', 2, 'after', 'western'),
    'NOK': ('Norwegian Krone', 'kr', 'kr', 2, 'after', 'western'),
    'DKK': ('Danish Krone', 'kr', 'kr', 2, 'after', 'western'),
    'ZAR': ('South African Rand', 'R', 'R', 2, 'before', 'western'),
    'RUB': ('Russian Ruble', '₽', '₽', 2, 'after', 'western'),
    'BRL': ('Brazilian Real', 'R$', 'R$', 2, 'before', 'western'),
    'MXN': ('Mexican Peso', '$', '$', 2, 'before', 'western'),
    'TRY': ('Turkish Lira', '₺', '₺', 2, 'before', 'western'),
    'PLN': ('Polish Zloty', 'zł', 'zł', 2, 'after', 'western'),
    'TWD': ('Taiwan Dollar', 'NT$', 'NT$', 2, 'before', 'western'),
    'EGP': ('Egyptian Pound', 'E£', 'ج.م', 2, 'before', 'western'),
    'NGN': ('Nigerian Naira', '₦', '₦', 2, 'before', 'western'),
}

# Default exchange rates with BDT as base (approximate as of 2024-2025)
DEFAULT_RATES_FROM_BDT = {
    'USD': Decimal('0.0083'),    # 1 BDT ≈ 0.0083 USD (1 USD ≈ 120 BDT)
    'EUR': Decimal('0.0076'),    # 1 BDT ≈ 0.0076 EUR
    'GBP': Decimal('0.0065'),    # 1 BDT ≈ 0.0065 GBP
    'INR': Decimal('0.69'),      # 1 BDT ≈ 0.69 INR
    'PKR': Decimal('2.32'),      # 1 BDT ≈ 2.32 PKR
    'NPR': Decimal('1.11'),      # 1 BDT ≈ 1.11 NPR
    'LKR': Decimal('2.50'),      # 1 BDT ≈ 2.50 LKR
    'AED': Decimal('0.030'),     # 1 BDT ≈ 0.030 AED
    'SAR': Decimal('0.031'),     # 1 BDT ≈ 0.031 SAR
    'MYR': Decimal('0.038'),     # 1 BDT ≈ 0.038 MYR
    'SGD': Decimal('0.011'),     # 1 BDT ≈ 0.011 SGD
    'CAD': Decimal('0.011'),     # 1 BDT ≈ 0.011 CAD
    'AUD': Decimal('0.012'),     # 1 BDT ≈ 0.012 AUD
    'JPY': Decimal('1.25'),      # 1 BDT ≈ 1.25 JPY
    'CNY': Decimal('0.059'),     # 1 BDT ≈ 0.059 CNY
    'CHF': Decimal('0.0073'),    # 1 BDT ≈ 0.0073 CHF
    'HKD': Decimal('0.065'),     # 1 BDT ≈ 0.065 HKD
    'KRW': Decimal('11.0'),      # 1 BDT ≈ 11.0 KRW
    'THB': Decimal('0.29'),      # 1 BDT ≈ 0.29 THB
    'PHP': Decimal('0.47'),      # 1 BDT ≈ 0.47 PHP
    'IDR': Decimal('132'),       # 1 BDT ≈ 132 IDR
    'VND': Decimal('206'),       # 1 BDT ≈ 206 VND
}

# Priority currencies to always seed first
PRIORITY_CURRENCIES = ['BDT', 'USD', 'EUR', 'GBP', 'INR']

# API providers in order of preference
PROVIDERS = ['exchangerate_api', 'openexchange', 'exchangeratesapi', 'fixer', 'ecb']

# Provider display names
PROVIDER_NAMES = {
    'exchangerate_api': 'ExchangeRate-API (exchangerate-api.com)',
    'openexchange': 'Open Exchange Rates (openexchangerates.org)',
    'exchangeratesapi': 'ExchangeRatesAPI (exchangeratesapi.io)',
    'fixer': 'Fixer.io',
    'ecb': 'European Central Bank (free, no API key)',
}


class Command(BaseCommand):
    help = 'Seed currencies and exchange rates (unified command)'

    def add_arguments(self, parser):
        parser.add_argument(
            'codes',
            nargs='*',
            type=str,
            help='Currency codes to add (e.g., USD EUR GBP). If omitted, adds all common currencies.'
        )
        parser.add_argument(
            '--base',
            type=str,
            default='BDT',
            help='Base currency code (default: BDT)'
        )
        parser.add_argument(
            '--update',
            action='store_true',
            help='Update existing currencies and rates instead of skipping'
        )
        parser.add_argument(
            '--fetch',
            action='store_true',
            help='Fetch live rates from API provider'
        )
        parser.add_argument(
            '--provider',
            type=str,
            choices=['manual', 'openexchange', 'exchangeratesapi', 'fixer', 'ecb', 'exchangerate_api'],
            help='Exchange rate provider to use (with --fetch)'
        )
        parser.add_argument(
            '--api-key',
            type=str,
            help='API key for the provider (overrides settings)'
        )
        parser.add_argument(
            '--list',
            action='store_true',
            help='List all available currencies'
        )
        parser.add_argument(
            '--no-rates',
            action='store_true',
            help='Only add currencies, skip exchange rates'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help="Show what would be done without making changes"
        )

    def handle(self, *args, **options):
        if options['list']:
            return self._list_currencies()

        base_code = options['base'].upper()
        codes = [c.upper() for c in options.get('codes', [])]
        update = options.get('update', False)
        fetch = options.get('fetch', False)
        no_rates = options.get('no_rates', False)
        dry_run = options.get('dry_run', False)
        provider = options.get('provider')
        api_key = options.get('api_key')

        if dry_run:
            self.stdout.write(self.style.NOTICE('DRY RUN - no changes will be made\n'))

        # Determine which currencies to add
        if codes:
            currencies_to_add = [c for c in codes if c in CURRENCIES]
            unknown = [c for c in codes if c not in CURRENCIES]
            if unknown:
                self.stdout.write(self.style.WARNING(f'Unknown currencies: {", ".join(unknown)}'))
        else:
            # Add all currencies, with priority ones first
            currencies_to_add = PRIORITY_CURRENCIES + [c for c in CURRENCIES if c not in PRIORITY_CURRENCIES]

        # Ensure base currency is included and first
        if base_code not in currencies_to_add:
            currencies_to_add.insert(0, base_code)
        elif currencies_to_add[0] != base_code:
            currencies_to_add.remove(base_code)
            currencies_to_add.insert(0, base_code)

        with transaction.atomic():
            # Step 1: Add currencies
            created, updated = self._seed_currencies(currencies_to_add, base_code, update, dry_run)
            
            if not no_rates:
                # Step 2: Add exchange rates
                if fetch:
                    rates_count = self._fetch_live_rates(base_code, provider, api_key, update, dry_run)
                else:
                    rates_count = self._seed_default_rates(base_code, update, dry_run)
            else:
                rates_count = 0

        # Summary
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('=' * 50))
        self.stdout.write(self.style.SUCCESS('Currency seeding complete!'))
        self.stdout.write(f'  Currencies created: {created}')
        self.stdout.write(f'  Currencies updated: {updated}')
        self.stdout.write(f'  Exchange rates: {rates_count}')
        self.stdout.write(self.style.SUCCESS('=' * 50))

    def _list_currencies(self):
        """List all available currencies."""
        self.stdout.write(self.style.MIGRATE_HEADING('\nAvailable currencies:'))
        self.stdout.write('-' * 80)
        self.stdout.write(f'{"Code":<6} {"Name":<25} {"Symbol":<8} {"Decimal":<8} {"Status"}')
        self.stdout.write('-' * 80)
        
        for code in sorted(CURRENCIES.keys()):
            name, symbol, _, decimals, _, _ = CURRENCIES[code]
            exists = Currency.objects.filter(code=code).exists()
            status = self.style.SUCCESS('✓ Exists') if exists else '  Available'
            is_priority = '*' if code in PRIORITY_CURRENCIES else ' '
            self.stdout.write(f'{is_priority}{code:<5} {name:<25} {symbol:<8} {decimals:<8} {status}')
        
        self.stdout.write('-' * 80)
        self.stdout.write(f'Total: {len(CURRENCIES)} currencies (* = priority)\n')
        self.stdout.write('Usage examples:')
        self.stdout.write('  python manage.py seed_currencies                  # Seed all currencies')
        self.stdout.write('  python manage.py seed_currencies USD EUR GBP      # Seed specific currencies')
        self.stdout.write('  python manage.py seed_currencies --fetch          # Fetch live rates')
        self.stdout.write('  python manage.py seed_currencies --update         # Update existing')

    def _seed_currencies(self, codes, base_code, update, dry_run):
        """Seed currency records."""
        created = 0
        updated = 0
        
        self.stdout.write(self.style.MIGRATE_HEADING(f'\nSeeding {len(codes)} currencies...'))
        
        for i, code in enumerate(codes):
            if code not in CURRENCIES:
                continue
                
            name, symbol, native_symbol, decimals, position, num_sys = CURRENCIES[code]
            
            if dry_run:
                self.stdout.write(f'  Would add: {code} - {name}')
                continue
            
            existing = Currency.objects.filter(code=code).first()
            
            if existing:
                if update:
                    existing.name = name
                    existing.symbol = symbol
                    existing.native_symbol = native_symbol
                    existing.decimal_places = decimals
                    existing.symbol_position = position
                    existing.number_system = num_sys
                    existing.is_active = True
                    if code == base_code:
                        existing.is_base_currency = True
                        existing.is_default = True
                    existing.sort_order = i
                    existing.save()
                    updated += 1
                    self.stdout.write(f'  Updated: {code} - {name}')
                else:
                    self.stdout.write(f'  Exists: {code} - {name} (use --update to modify)')
            else:
                Currency.objects.create(
                    code=code,
                    name=name,
                    symbol=symbol,
                    native_symbol=native_symbol,
                    decimal_places=decimals,
                    symbol_position=position,
                    number_system=num_sys,
                    is_active=True,
                    is_default=(code == base_code),
                    is_base_currency=(code == base_code),
                    sort_order=i,
                )
                created += 1
                self.stdout.write(self.style.SUCCESS(f'  Created: {code} - {name}'))
        
        return created, updated

    def _seed_default_rates(self, base_code, update, dry_run):
        """Seed exchange rates from default values."""
        self.stdout.write(self.style.MIGRATE_HEADING(f'\nSeeding exchange rates (base: {base_code})...'))
        
        if base_code != 'BDT':
            self.stdout.write(self.style.WARNING(
                f'Default rates are based on BDT. Use --fetch for {base_code}-based rates.'
            ))
            return 0
        
        try:
            base_currency = Currency.objects.get(code=base_code)
        except Currency.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'Base currency {base_code} not found'))
            return 0
        
        rates_count = 0
        
        for target_code, rate in DEFAULT_RATES_FROM_BDT.items():
            try:
                target_currency = Currency.objects.get(code=target_code)
            except Currency.DoesNotExist:
                continue
            
            if dry_run:
                self.stdout.write(f'  Would add rate: {base_code} → {target_code}: {rate}')
                rates_count += 1
                continue
            
            # Forward rate (BDT -> target)
            self._create_or_update_rate(base_currency, target_currency, rate, update, 'default')
            rates_count += 1
            
            # Inverse rate (target -> BDT)
            inverse_rate = (Decimal('1') / rate).quantize(Decimal('0.00000001'))
            self._create_or_update_rate(target_currency, base_currency, inverse_rate, update, 'default')
            rates_count += 1
        
        return rates_count

    def _create_or_update_rate(self, from_currency, to_currency, rate, update, source='manual'):
        """Create or update a single exchange rate."""
        existing = ExchangeRate.objects.filter(
            from_currency=from_currency,
            to_currency=to_currency,
            is_active=True
        ).first()
        
        if existing:
            if update:
                existing.rate = rate
                existing.valid_from = timezone.now()
                existing.source = source
                existing.save()
                self.stdout.write(
                    f'  Updated: {from_currency.code} → {to_currency.code}: {rate}'
                )
            # Skip if exists and not updating
        else:
            ExchangeRate.objects.create(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=rate,
                source=source,
                valid_from=timezone.now(),
                is_active=True
            )
            self.stdout.write(self.style.SUCCESS(
                f'  Created: {from_currency.code} → {to_currency.code}: {rate}'
            ))

    def _fetch_live_rates(self, base_code, provider, api_key, update, dry_run):
        """Fetch live exchange rates from API with fallback support."""
        self.stdout.write(self.style.MIGRATE_HEADING(f'\nFetching live exchange rates...'))
        self.stdout.write(f'Base currency: {base_code}')
        
        if dry_run:
            self.stdout.write(self.style.NOTICE('DRY RUN - would fetch from API providers'))
            return 0
        
        # Build API keys dict
        api_keys = {
            'exchangerate_api': api_key if provider == 'exchangerate_api' else getattr(settings, 'EXCHANGERATE_API_KEY', ''),
            'openexchange': api_key if provider == 'openexchange' else getattr(settings, 'OPENEXCHANGE_RATES_API_KEY', ''),
            'exchangeratesapi': api_key if provider == 'exchangeratesapi' else getattr(settings, 'EXCHANGERATESAPI_KEY', ''),
            'fixer': api_key if provider == 'fixer' else getattr(settings, 'FIXER_API_KEY', ''),
        }
        
        # Show available providers
        self.stdout.write('')
        self.stdout.write('Available API providers:')
        for p in PROVIDERS:
            key = api_keys.get(p, '')
            if p == 'ecb':
                status = self.style.SUCCESS('✓ Available (no key needed)')
            elif key:
                status = self.style.SUCCESS(f'✓ API key configured')
            else:
                status = self.style.WARNING('✗ No API key')
            self.stdout.write(f'  {PROVIDER_NAMES.get(p, p)}: {status}')
        self.stdout.write('')
        
        # If specific provider requested, try only that one
        if provider:
            self.stdout.write(f'Using specified provider: {PROVIDER_NAMES.get(provider, provider)}')
            count = self._try_provider(provider, api_keys.get(provider, ''), base_code, update)
            if count > 0:
                self.stdout.write(self.style.SUCCESS(f'\n✓ Successfully fetched {count} rates from {PROVIDER_NAMES.get(provider, provider)}'))
                return count
            else:
                self.stdout.write(self.style.ERROR(f'\n✗ Failed to fetch from {provider}'))
                self.stdout.write('Falling back to default rates...')
                return self._seed_default_rates(base_code, update, dry_run)
        
        # Try providers in order
        self.stdout.write('Trying providers in order...\n')
        
        for p in PROVIDERS:
            key = api_keys.get(p, '')
            
            # Skip if no API key (except ECB)
            if p != 'ecb' and not key:
                self.stdout.write(f'  Skipping {PROVIDER_NAMES.get(p, p)}: No API key')
                continue
            
            self.stdout.write(f'  Trying {PROVIDER_NAMES.get(p, p)}...')
            
            try:
                count = self._try_provider(p, key, base_code, update)
                if count > 0:
                    self.stdout.write(self.style.SUCCESS(f'\n✓ Successfully fetched {count} rates from {PROVIDER_NAMES.get(p, p)}'))
                    return count
                else:
                    self.stdout.write(self.style.WARNING(f'    → Returned 0 rates'))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'    → Failed: {str(e)[:50]}'))
        
        # All providers failed
        self.stdout.write(self.style.WARNING('\nAll providers failed. Using default rates...'))
        return self._seed_default_rates(base_code, update, dry_run)
    
    def _try_provider(self, provider, api_key, base_code, update):
        """Try fetching rates from a specific provider."""
        import requests
        
        if provider == 'exchangerate_api':
            if not api_key:
                return 0
            response = requests.get(
                f'https://v6.exchangerate-api.com/v6/{api_key}/latest/USD',
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            if data.get('result') != 'success':
                raise Exception(data.get('error-type', 'Unknown error'))
            return self._process_usd_rates(data.get('conversion_rates', {}), 'exchangerate_api', update)
        
        elif provider == 'openexchange':
            if not api_key:
                return 0
            response = requests.get(
                f'https://openexchangerates.org/api/latest.json?app_id={api_key}',
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return self._process_usd_rates(data.get('rates', {}), 'openexchange', update)
        
        elif provider == 'exchangeratesapi':
            if not api_key:
                return 0
            response = requests.get(
                f'http://api.exchangeratesapi.io/v1/latest?access_key={api_key}',
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            if not data.get('success', True):
                error = data.get('error', {})
                raise Exception(error.get('info', 'Unknown error'))
            return self._process_eur_rates(data.get('rates', {}), 'exchangeratesapi', update)
        
        elif provider == 'fixer':
            if not api_key:
                return 0
            response = requests.get(
                f'http://data.fixer.io/api/latest?access_key={api_key}',
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            if not data.get('success'):
                error = data.get('error', {})
                raise Exception(error.get('info', 'Unknown error'))
            return self._process_eur_rates(data.get('rates', {}), 'fixer', update)
        
        elif provider == 'ecb':
            import xml.etree.ElementTree as ET
            response = requests.get(
                'https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml',
                timeout=30
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            ns = {
                'gesmes': 'http://www.gesmes.org/xml/2002-08-01',
                'eurofxref': 'http://www.ecb.int/vocabulary/2002-08-01/eurofxref'
            }
            rates = {}
            for cube in root.findall('.//eurofxref:Cube[@currency]', ns):
                rates[cube.get('currency')] = float(cube.get('rate'))
            return self._process_eur_rates(rates, 'ecb', update)
        
        return 0
    
    def _process_usd_rates(self, rates, source, update):
        """Process rates with USD as base."""
        try:
            usd = Currency.objects.get(code='USD')
        except Currency.DoesNotExist:
            self.stdout.write(self.style.ERROR('USD currency not found'))
            return 0
        
        count = 0
        for code, rate in rates.items():
            if code == 'USD':
                continue
            try:
                target = Currency.objects.get(code=code, is_active=True)
                self._create_or_update_rate(usd, target, Decimal(str(rate)), update, source)
                count += 1
            except Currency.DoesNotExist:
                continue
        
        return count
    
    def _process_eur_rates(self, rates, source, update):
        """Process rates with EUR as base."""
        try:
            eur = Currency.objects.get(code='EUR')
        except Currency.DoesNotExist:
            self.stdout.write(self.style.ERROR('EUR currency not found'))
            return 0
        
        count = 0
        for code, rate in rates.items():
            if code == 'EUR':
                continue
            try:
                target = Currency.objects.get(code=code, is_active=True)
                self._create_or_update_rate(eur, target, Decimal(str(rate)), update, source)
                count += 1
            except Currency.DoesNotExist:
                continue
        
        return count

    def _get_api_key(self, provider):
        """Get API key from settings for the given provider."""
        key_mapping = {
            'openexchange': 'OPENEXCHANGE_RATES_API_KEY',
            'exchangeratesapi': 'EXCHANGERATESAPI_KEY',
            'fixer': 'FIXER_API_KEY',
            'exchangerate_api': 'EXCHANGERATE_API_KEY',
        }
        setting_name = key_mapping.get(provider)
        if setting_name:
            return getattr(settings, setting_name, None)
        return None
