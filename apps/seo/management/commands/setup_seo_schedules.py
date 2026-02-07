from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Create celery-beat periodic tasks for warmup and prerendering if django_celery_beat is installed.'

    def handle(self, *args, **options):
        try:
            from django_celery_beat.models import PeriodicTask, CrontabSchedule, IntervalSchedule
        except Exception as exc:
            self.stdout.write(self.style.ERROR('django_celery_beat not installed or not migrated. Install and migrate to use this command.'))
            return

        # Warmup every 3 minutes
        try:
            interval, _ = IntervalSchedule.objects.get_or_create(every=3, period=IntervalSchedule.MINUTES)
            PeriodicTask.objects.update_or_create(
                name='seo-warmup-every-3-min',
                defaults={
                    'interval': interval,
                    'task': 'apps.seo.tasks.warmup_service',
                    'enabled': True,
                }
            )
            self.stdout.write(self.style.SUCCESS('Scheduled warmup every 3 minutes'))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f'Failed to schedule warmup: {exc}'))

        # Prerender daily at 02:00 UTC
        try:
            cron, _ = CrontabSchedule.objects.get_or_create(minute='0', hour='2', day_of_week='*', day_of_month='*', month_of_year='*', timezone='UTC')
            PeriodicTask.objects.update_or_create(
                name='seo-prerender-daily',
                defaults={
                    'crontab': cron,
                    'task': 'apps.seo.tasks.prerender_top_task',
                    'args': '[10,20,true]',
                    'enabled': True,
                }
            )
            self.stdout.write(self.style.SUCCESS('Scheduled prerender daily at 02:00 UTC'))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f'Failed to schedule prerender: {exc}'))
