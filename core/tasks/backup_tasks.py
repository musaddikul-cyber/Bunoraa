"""
Celery background tasks for automated backups.
"""

import logging
import subprocess
from pathlib import Path
from datetime import datetime
from celery import shared_task
from django.conf import settings
import gzip
import shutil

logger = logging.getLogger('bunoraa.tasks.backup')


@shared_task(bind=True, max_retries=3, default_retry_delay=300)
def backup_database_to_r2(self):
    """
    Backup PostgreSQL database to Cloudflare R2 storage.
    Runs as a background task to avoid blocking web workers.
    """
    try:
        logger.info("Starting database backup task")
        
        # Get database connection info
        db_settings = settings.DATABASES['default']
        db_name = db_settings.get('NAME')
        db_user = db_settings.get('USER')
        db_host = db_settings.get('HOST', 'localhost')
        db_port = db_settings.get('PORT', '5432')
        db_password = db_settings.get('PASSWORD', '')
        
        # Create backup filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{db_name}_backup_{timestamp}.sql"
        backup_dir = Path(settings.BASE_DIR) / 'backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / backup_filename
        
        # Run pg_dump
        cmd = [
            'pg_dump',
            '--host', db_host,
            '--port', str(db_port),
            '--username', db_user,
            '--no-password',
            '--format', 'plain',
            '--verbose',
            '--file', str(backup_path),
            db_name
        ]
        
        env = {'PGPASSWORD': db_password}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            raise Exception(f"pg_dump failed: {result.stderr}")
        
        # Compress
        compressed_path = backup_path.with_suffix('.gz')
        with open(backup_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb', compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
        backup_path.unlink()  # Remove uncompressed
        
        # Upload to R2 if configured
        r2_bucket = getattr(settings, 'R2_BUCKET_NAME', None)
        if r2_bucket:
            # Import here to avoid circular imports
            from core.services.r2_storage import upload_file
            s3_key = f"backups/database/{compressed_path.name}"
            upload_file(str(compressed_path), r2_bucket, s3_key)
            logger.info(f"Database backup uploaded to R2: {s3_key}")
        else:
            logger.info(f"Database backup saved locally: {compressed_path}")
        
        return {
            'status': 'success',
            'backup_path': str(compressed_path),
            'size': compressed_path.stat().st_size
        }
        
    except Exception as exc:
        logger.exception("Database backup failed")
        # Retry on failure
        raise self.retry(exc=exc)


@shared_task(bind=True, max_retries=2, default_retry_delay=600)
def backup_media_to_r2(self):
    """
    Backup media files to Cloudflare R2 storage.
    Creates a compressed archive of all media files.
    """
    try:
        logger.info("Starting media backup task")
        
        media_root = Path(settings.MEDIA_ROOT)
        if not media_root.exists():
            logger.warning("Media root not found, skipping backup")
            return {'status': 'skipped', 'reason': 'media_root_not_found'}
        
        # Create backup archive
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"media_backup_{timestamp}.tar.gz"
        backup_dir = Path(settings.BASE_DIR) / 'backups' / 'media'
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / backup_filename
        
        # Create tar.gz archive
        import tarfile
        with tarfile.open(backup_path, 'w:gz', compresslevel=6) as tar:
            tar.add(media_root, arcname='media')
        
        # Upload to R2 if configured
        r2_bucket = getattr(settings, 'R2_BUCKET_NAME', None)
        if r2_bucket:
            from core.services.r2_storage import upload_file
            s3_key = f"backups/media/{backup_path.name}"
            upload_file(str(backup_path), r2_bucket, s3_key)
            logger.info(f"Media backup uploaded to R2: {s3_key}")
            
            # Clean up local file after successful upload
            backup_path.unlink()
        else:
            logger.info(f"Media backup saved locally: {backup_path}")
        
        return {
            'status': 'success',
            'backup_path': str(backup_path),
            'size': backup_path.stat().st_size
        }
        
    except Exception as exc:
        logger.exception("Media backup failed")
        raise self.retry(exc=exc)
