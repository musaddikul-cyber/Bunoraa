import os
import json
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from apps.seo.models import Keyword, GSCMetric


class Command(BaseCommand):
    help = 'Fetch GSC metrics for tracked keywords (requires service account key in env GSC_SERVICE_ACCOUNT_FILE)'

    def add_arguments(self, parser):
        parser.add_argument('--start-date', type=str, help='YYYY-MM-DD start', required=False)
        parser.add_argument('--end-date', type=str, help='YYYY-MM-DD end', required=False)

    def handle(self, *args, **options):
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except Exception as exc:
            raise CommandError('google-api-python-client and google-auth must be installed: pip install google-api-python-client google-auth')

        keyfile = os.environ.get('GSC_SERVICE_ACCOUNT_FILE') or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not keyfile or not os.path.exists(keyfile):
            raise CommandError('GSC service account JSON key file not configured. Set GSC_SERVICE_ACCOUNT_FILE env var or GOOGLE_APPLICATION_CREDENTIALS to a valid path.')

        start_date = options.get('start_date') or (timezone.now().date().isoformat())
        end_date = options.get('end_date') or start_date

        creds = service_account.Credentials.from_service_account_file(keyfile, scopes=['https://www.googleapis.com/auth/webmasters.readonly'])
        service = build('searchconsole', 'v1', credentials=creds)

        # Use SearchAnalytics.query for each tracked site and keyword
        site_url = os.environ.get('GSC_SITE_URL') or 'https://bunoraa.com'

        keywords = Keyword.objects.filter(is_target=True)
        for k in keywords:
            request = {
                'startDate': start_date,
                'endDate': end_date,
                'dimensions': ['query'],
                'dimensionFilterGroups': [{
                    'filters': [{
                        'dimension': 'query',
                        'operator': 'equals',
                        'expression': k.term
                    }]
                }],
                'rowLimit': 25000
            }
            resp = service.searchanalytics().query(siteUrl=site_url, body=request).execute()
            rows = resp.get('rows', [])
            for r in rows:
                clicks = int(r.get('clicks', 0))
                impressions = int(r.get('impressions', 0))
                ctr = float(r.get('ctr', 0.0))
                position = float(r.get('position', 0.0))
                GSCMetric.objects.update_or_create(keyword=k, date=start_date, defaults={
                    'clicks': clicks, 'impressions': impressions, 'ctr': ctr, 'position': position, 'raw': r
                })
            self.stdout.write(self.style.SUCCESS(f'Imported GSC data for "{k.term}" ({len(rows)} rows)'))
