"""
Backup the site: database fixtures for apps and optional media/static files.

Usage examples:
  python manage.py backup_site                 # full site fixtures -> timestamped tar.gz
  python manage.py backup_site --apps=categories,products --include-media --output=./backups/site.tar.gz

Notes:
- By default this will dump each app as a separate JSON fixture file under `fixtures/` inside an archive.
- Media and static inclusion is optional to avoid large archives when not required.
"""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterable, List

from django.core.management.base import BaseCommand, CommandError
from django.core.management import call_command
from django.apps import apps as django_apps
from django.conf import settings

# Optional boto3 for uploads
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:
    boto3 = None
    BotoCoreError = ClientError = Exception


class Command(BaseCommand):
    help = 'Create a site backup: DB fixtures (per-app) and optional media/static contents packed into a tar.gz'

    def add_arguments(self, parser):
        parser.add_argument('--apps', type=str, help='Comma-separated app labels to include (default: all apps)')
        parser.add_argument('--exclude-apps', type=str, help='Comma-separated app labels to exclude')
        parser.add_argument('--include-media', action='store_true', help='Include MEDIA_ROOT directory in backup')
        parser.add_argument('--include-static', action='store_true', help='Include STATIC_ROOT directory in backup')
        parser.add_argument('--output', type=str, help='Output archive path (defaults to backups/site_backup_<ts>.tar.gz)')
        parser.add_argument('--no-compress', action='store_true', help='Create an uncompressed tar archive instead of gzip')
        parser.add_argument('--upload-s3', action='store_true', help='Upload the produced archive to S3 (requires boto3 and s3 bucket config)')
        parser.add_argument('--s3-bucket', type=str, help='S3 bucket to upload the archive to (optional; can use settings.AWS_BACKUP_S3_BUCKET)')
        parser.add_argument('--s3-key-prefix', type=str, default='', help='Optional key prefix to use when uploading to S3')
        parser.add_argument('--remove-local', action='store_true', help='Remove local archive after successful upload')
        parser.add_argument('--max-backups', type=int, help='Maximum number of backups to retain in the backup directory (older removed)')
        parser.add_argument('--retention-days', type=int, help='Remove backups older than this many days (based on mtime)')

    def handle(self, *args, **options):
        apps_opt = options.get('apps')
        exclude_opt = options.get('exclude_apps')
        include_media = options.get('include_media')
        include_static = options.get('include_static')
        output = options.get('output')
        no_compress = options.get('no_compress')

        # Resolve app list
        if apps_opt:
            app_labels = [a.strip() for a in apps_opt.split(',') if a.strip()]
        else:
            app_labels = [c.label for c in django_apps.get_app_configs()]

        if exclude_opt:
            exclude = {a.strip() for a in exclude_opt.split(',') if a.strip()}
            app_labels = [a for a in app_labels if a not in exclude]

        if not app_labels:
            raise CommandError('No apps selected to backup')

        # Output path
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        backups_dir = Path('backups')
        backups_dir.mkdir(parents=True, exist_ok=True)
        if output:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            suffix = 'tar' if no_compress else 'tar.gz'
            out_path = backups_dir / f'site_backup_{ts}.{suffix}'

        self.stdout.write(self.style.NOTICE(f'Creating backup at {out_path}'))

        tmpdir = Path(tempfile.mkdtemp(prefix='site_backup_'))
        try:
            fixtures_dir = tmpdir / 'fixtures'
            fixtures_dir.mkdir()

            # Dump per-app fixtures
            for label in sorted(set(app_labels)):
                try:
                    # Only dump apps that are installed
                    config = django_apps.get_app_config(label)
                except LookupError:
                    # Skip unknown labels but log
                    self.stdout.write(self.style.WARNING(f'App not found or not installed: {label}'))
                    continue

                filename = fixtures_dir / f'{label}.json'
                self.stdout.write(self.style.NOTICE(f'Dumping fixtures for app: {label} -> {filename}'))
                with open(filename, 'w', encoding='utf-8') as fh:
                    # Using call_command('dumpdata', app_label) to write JSON
                    call_command('dumpdata', label, indent=2, stdout=fh, stdout_is_stream=True)

            # Optionally copy media/static
            if include_media:
                media_root = getattr(settings, 'MEDIA_ROOT', None)
                if media_root and Path(media_root).exists():
                    dst = tmpdir / 'media'
                    self.stdout.write(self.style.NOTICE(f'Including media from {media_root}'))
                    shutil.copytree(media_root, dst)
                else:
                    self.stdout.write(self.style.WARNING('MEDIA_ROOT is not set or does not exist; skipping media'))

            if include_static:
                static_root = getattr(settings, 'STATIC_ROOT', None)
                if static_root and Path(static_root).exists():
                    dst = tmpdir / 'static'
                    self.stdout.write(self.style.NOTICE(f'Including static from {static_root}'))
                    shutil.copytree(static_root, dst)
                else:
                    self.stdout.write(self.style.WARNING('STATIC_ROOT is not set or does not exist; skipping static'))

            # Create archive
            mode = 'w' if no_compress else 'w:gz'
            with tarfile.open(out_path, mode) as tar:
                # Add fixtures directory contents into archive root under 'fixtures/'
                for p in tmpdir.iterdir():
                    tar.add(p, arcname=p.name)

            self.stdout.write(self.style.SUCCESS(f'Backup created: {out_path}'))

            # Optional upload to S3
            if options.get('upload_s3'):
                # Determine bucket
                bucket = options.get('s3_bucket') or getattr(settings, 'AWS_BACKUP_S3_BUCKET', None)
                key_prefix = options.get('s3_key_prefix') or ''
                if not bucket:
                    self.stdout.write(self.style.ERROR('S3 upload requested but no bucket configured (use --s3-bucket or set AWS_BACKUP_S3_BUCKET in settings).'))
                elif boto3 is None:
                    self.stdout.write(self.style.ERROR('boto3 is not installed. Install with: pip install boto3'))
                else:
                    client = boto3.client('s3')
                    key_name = f"{key_prefix.rstrip('/')}/{out_path.name}" if key_prefix else out_path.name
                    self.stdout.write(self.style.NOTICE(f'Uploading backup to s3://{bucket}/{key_name}'))
                    try:
                        client.upload_file(str(out_path), bucket, key_name)
                        self.stdout.write(self.style.SUCCESS(f'Uploaded to s3://{bucket}/{key_name}'))
                        if options.get('remove_local'):
                            try:
                                out_path.unlink()
                                self.stdout.write(self.style.NOTICE('Local archive removed after upload.'))
                            except Exception as exc:
                                self.stdout.write(self.style.WARNING(f'Failed to remove local archive: {exc}'))
                    except (BotoCoreError, ClientError) as exc:
                        self.stdout.write(self.style.ERROR(f'Failed to upload backup to S3: {exc}'))

            # Retention: max backups by count and/or retention_days
            try:
                retention_days = options.get('retention_days')
                max_backups = options.get('max_backups')
                backup_dir = out_path.parent
                to_remove: List[Path] = []
                if retention_days is not None:
                    cutoff = datetime.utcnow() - timedelta(days=int(retention_days))
                    for f in sorted(backup_dir.iterdir(), key=lambda p: p.stat().st_mtime):
                        if f.is_file() and datetime.utcfromtimestamp(f.stat().st_mtime) < cutoff:
                            to_remove.append(f)
                if max_backups is not None and max_backups >= 0:
                    files = [p for p in sorted(backup_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if p.is_file()]
                    # keep the newest `max_backups` and remove the rest
                    for old in files[max_backups:]:
                        if old not in to_remove:
                            to_remove.append(old)
                for r in to_remove:
                    try:
                        r.unlink()
                        self.stdout.write(self.style.NOTICE(f'Removed old backup: {r.name}'))
                    except Exception as exc:
                        self.stdout.write(self.style.WARNING(f'Failed to remove {r}: {exc}'))
            except Exception as exc:
                self.stdout.write(self.style.WARNING(f'Error during retention cleanup: {exc}'))
        finally:
            # Clean up tempdir
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
