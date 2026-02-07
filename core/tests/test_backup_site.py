import tarfile
import tempfile
from pathlib import Path
from django.test import TestCase, override_settings
from django.core.management import call_command
from django.conf import settings
from apps.catalog.models import Category


class BackupSiteCommandTests(TestCase):

    def test_backup_sites_creates_archive_with_fixtures(self):
        # Create a sample category so dumpdata has something
        Category.objects.create(name='Backup Test')

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz')
        tmp.close()
        out_path = Path(tmp.name)

        # Run command to backup only categories app
        call_command('backup_site', '--apps=categories', f'--output={out_path}', '--no-compress')

        # Open tar archive and verify fixtures file
        with tarfile.open(out_path, 'r') as tar:
            names = tar.getnames()
        assert 'fixtures/categories.json' in names

    @override_settings(MEDIA_ROOT=tempfile.gettempdir())
    def test_backup_includes_media_when_requested(self):
        # Create a dummy media file
        media_dir = Path(settings.MEDIA_ROOT)
        f = media_dir / 'backup_test_media.txt'
        f.write_text('hello')

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz')
        tmp.close()
        out_path = Path(tmp.name)

        # Run command including media
        call_command('backup_site', '--apps=categories', '--include-media', f'--output={out_path}', '--no-compress')

        with tarfile.open(out_path, 'r') as tar:
            names = tar.getnames()
        assert any(n.startswith('media') for n in names), 'Archive should include media directory'

        # Cleanup file
        f.unlink(missing_ok=True)
