from django.core.management.base import BaseCommand, CommandError
from apps.seo.services import snapshot_keyword_serp


class Command(BaseCommand):
    help = 'Fetch SERP snapshot for a keyword and store results in the DB'

    def add_arguments(self, parser):
        parser.add_argument('keyword', type=str, help='Keyword to snapshot')
        parser.add_argument('--num', type=int, default=10, help='Number of top results to capture')

    def handle(self, *args, **options):
        keyword = options['keyword']
        num = options['num']
        try:
            objs = snapshot_keyword_serp(keyword, num)
            self.stdout.write(self.style.SUCCESS(f'Saved {len(objs)} SERP rows for "{keyword}"'))
        except Exception as exc:
            raise CommandError(f'Error fetching SERP: {exc}')