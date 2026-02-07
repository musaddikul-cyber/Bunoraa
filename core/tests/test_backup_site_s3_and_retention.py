import tempfile
from pathlib import Path
from unittest import mock
from django.test import TestCase, override_settings
from django.core.management import call_command


class BackupSiteS3AndRetentionTests(TestCase):

    @override_settings(AWS_BACKUP_S3_BUCKET='test-bucket')
    @mock.patch('core.management.commands.backup_site.boto3')
    def test_uploads_to_s3_when_requested(self, mock_boto):
        # Prepare fake client
        fake_client = mock.Mock()
        mock_boto.client.return_value = fake_client

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz')
        tmp.close()
        out_path = Path(tmp.name)

        # Run command with upload flag (will use settings.AWS_BACKUP_S3_BUCKET)
        call_command('backup_site', '--apps=categories', f'--output={out_path}', '--no-compress', '--upload-s3')

        # Ensure boto3.client was called and upload_file invoked
        mock_boto.client.assert_called_once_with('s3')
        fake_client.upload_file.assert_called_once()

    def test_retention_keeps_max_backups(self):
        tmpdir = Path(tempfile.mkdtemp())
        # create 5 fake backups with different mtimes
        files = []
        for i in range(5):
            p = tmpdir / f'site_backup_{i}.tar'
            p.write_text('x')
            # set mtime to increasing value
            p.utime((i+1, i+1))
            files.append(p)

        # Create a target output in the same dir (simulates creating a new backup)
        out_path = tmpdir / 'site_backup_new.tar'
        out_path.write_text('new')

        # Run command retention logic by invoking backup_site with --output pointing to this dir
        # Use max-backups=2 to retain only two newest files
        call_command('backup_site', '--apps=categories', f'--output={out_path}', '--no-compress', '--max-backups=2')

        # After running, count files in tmpdir
        remaining = [p for p in tmpdir.iterdir() if p.is_file()]
        assert len(remaining) == 2, f'Expected 2 backups retained, found {len(remaining)}'