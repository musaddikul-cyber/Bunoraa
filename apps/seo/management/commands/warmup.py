from django.core.management.base import BaseCommand, CommandError
from apps.seo.tasks import warmup_service


class Command(BaseCommand):
    help = 'Run the warmup task once (calls warmup_service Celery task synchronously)'

    def handle(self, *args, **options):
        res = warmup_service()
        self.stdout.write(self.style.SUCCESS(f'Warmup results: {res}'))