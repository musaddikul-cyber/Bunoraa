"""
Database Backup Command - Production-Ready Automated Backups
Supports PostgreSQL backups with compression, encryption, and cloud storage upload.
"""
import os
import subprocess
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import logging

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

logger = logging.getLogger('bunoraa.backup')


class Command(BaseCommand):
    help = 'Create a database backup with optional compression and upload'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--output-dir',
            type=str,
            default='backups',
            help='Directory to store backup files'
        )
        parser.add_argument(
            '--compress',
            action='store_true',
            default=True,
            help='Compress backup with gzip'
        )
        parser.add_argument(
            '--upload-s3',
            action='store_true',
            help='Upload backup to S3-compatible storage'
        )
        parser.add_argument(
            '--retention-days',
            type=int,
            default=30,
            help='Number of days to retain backups'
        )
        parser.add_argument(
            '--clean-old',
            action='store_true',
            help='Remove backups older than retention period'
        )
    
    def handle(self, *args, **options):
        output_dir = Path(options['output_dir'])
        compress = options['compress']
        upload_s3 = options['upload_s3']
        retention_days = options['retention_days']
        clean_old = options['clean_old']
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate backup filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        db_name = self.get_db_name()
        filename = f"{db_name}_{timestamp}.sql"
        filepath = output_dir / filename
        
        self.stdout.write(f"Starting database backup: {filename}")
        
        try:
            # Create backup
            self.create_backup(filepath)
            
            # Compress if requested
            if compress:
                filepath = self.compress_backup(filepath)
            
            # Upload to S3 if requested
            if upload_s3:
                self.upload_to_s3(filepath)
            
            # Clean old backups
            if clean_old:
                self.clean_old_backups(output_dir, retention_days)
            
            self.stdout.write(
                self.style.SUCCESS(f"Backup completed successfully: {filepath}")
            )
            
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            raise CommandError(f"Backup failed: {str(e)}")
    
    def get_db_name(self) -> str:
        """Get database name from settings."""
        db_config = settings.DATABASES['default']
        return db_config.get('NAME', 'bunoraa')
    
    def create_backup(self, filepath: Path):
        """Create database dump using pg_dump."""
        db_config = settings.DATABASES['default']
        
        # Build connection string from existing config
        env = os.environ.copy()
        if 'PASSWORD' in db_config:
            env['PGPASSWORD'] = db_config['PASSWORD']
        
        cmd = ['pg_dump', '--verbose', '--no-owner', '--no-privileges']
        
        if 'HOST' in db_config and db_config['HOST']:
            cmd.extend(['--host', db_config['HOST']])
        if 'PORT' in db_config and db_config['PORT']:
            cmd.extend(['--port', str(db_config['PORT'])])
        if 'USER' in db_config and db_config['USER']:
            cmd.extend(['--username', db_config['USER']])
        
        # Use custom format for better compression
        cmd.extend(['--format', 'custom', '--file', str(filepath)])
        cmd.append(db_config['NAME'])
        
        self.stdout.write(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"pg_dump failed: {result.stderr}")
        
        logger.info(f"Database backup created: {filepath}")
    
    def compress_backup(self, filepath: Path) -> Path:
        """Compress backup file using gzip."""
        compressed_path = Path(str(filepath) + '.gz')
        
        self.stdout.write(f"Compressing backup to {compressed_path}")
        
        with open(filepath, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb', compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove uncompressed file
        filepath.unlink()
        
        # Log compression stats
        original_size = filepath.stat().st_size if filepath.exists() else 0
        compressed_size = compressed_path.stat().st_size
        ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        self.stdout.write(
            f"Compression: {original_size // 1024}KB -> {compressed_size // 1024}KB ({ratio:.1f}% reduction)"
        )
        
        logger.info(f"Backup compressed: {compressed_path}")
        return compressed_path
    
    def upload_to_s3(self, filepath: Path):
        """Upload backup to S3-compatible storage (Cloudflare R2)."""
        import boto3
        from botocore.exceptions import ClientError
        
        bucket_name = os.environ.get('BACKUP_S3_BUCKET')
        if not bucket_name:
            self.stdout.write(self.style.WARNING("BACKUP_S3_BUCKET not set, skipping upload"))
            return
        
        endpoint_url = os.environ.get('BACKUP_S3_ENDPOINT')
        access_key = os.environ.get('BACKUP_S3_ACCESS_KEY')
        secret_key = os.environ.get('BACKUP_S3_SECRET_KEY')
        
        if not all([endpoint_url, access_key, secret_key]):
            raise Exception("Missing S3 credentials for backup upload")
        
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            
            s3_key = f"backups/database/{filepath.name}"
            
            self.stdout.write(f"Uploading to S3: {s3_key}")
            
            s3.upload_file(
                str(filepath),
                bucket_name,
                s3_key,
                ExtraArgs={
                    'StorageClass': 'STANDARD_IA',  # Infrequent access for cost savings
                    'ServerSideEncryption': 'AES256'
                }
            )
            
            self.stdout.write(self.style.SUCCESS(f"Backup uploaded to S3: {s3_key}"))
            logger.info(f"Backup uploaded to S3: {s3_key}")
            
        except ClientError as e:
            raise Exception(f"S3 upload failed: {str(e)}")
    
    def clean_old_backups(self, output_dir: Path, retention_days: int):
        """Remove backups older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        removed_count = 0
        
        for backup_file in output_dir.glob('*.sql*'):
            if backup_file.is_file():
                file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_mtime < cutoff_date:
                    backup_file.unlink()
                    removed_count += 1
                    logger.info(f"Removed old backup: {backup_file}")
        
        if removed_count > 0:
            self.stdout.write(
                self.style.SUCCESS(f"Cleaned {removed_count} old backup(s)")
            )
