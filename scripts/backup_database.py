#!/usr/bin/env python3
"""
Bunoraa Database Backup Script
===============================

Automates PostgreSQL database backups with compression and rotation.
Supports local storage and cloud upload (AWS S3, R2).

Usage:
    python scripts/backup_database.py
    python scripts/backup_database.py --upload-to-s3
    python scripts/backup_database.py --retention-days 30

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (required)
    BACKUP_DIR: Local backup directory (default: backups/)
    S3_BUCKET: S3 bucket name for cloud storage
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    BACKUP_RETENTION_DAYS: Number of days to keep backups (default: 30)
"""

import os
import sys
import gzip
import shutil
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backup')


def parse_database_url(url: str) -> dict:
    """Parse PostgreSQL database URL into components."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return {
        'host': parsed.hostname or 'localhost',
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password,
    }


def create_backup_filename(database_name: str) -> str:
    """Generate backup filename with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{database_name}_backup_{timestamp}.sql"


def get_backup_dir() -> Path:
    """Get backup directory from environment or default."""
    backup_dir = Path(os.environ.get('BACKUP_DIR', 'backups'))
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def run_pg_dump(db_config: dict, output_path: Path) -> bool:
    """Run pg_dump to create backup."""
    cmd = [
        'pg_dump',
        '--host', db_config['host'],
        '--port', str(db_config['port']),
        '--username', db_config['user'],
        '--no-password',
        '--format', 'plain',
        '--verbose',
        '--file', str(output_path),
        db_config['database']
    ]
    
    # Set PGPASSWORD for passwordless authentication
    env = os.environ.copy()
    env['PGPASSWORD'] = db_config['password'] or ''
    
    try:
        logger.info(f"Starting backup to {output_path}")
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Backup completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Backup failed: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("pg_dump not found. Please install PostgreSQL client tools.")
        return False


def compress_backup(source_path: Path) -> Path:
    """Compress backup file using gzip."""
    compressed_path = source_path.with_suffix('.gz')
    
    logger.info(f"Compressing backup to {compressed_path}")
    with open(source_path, 'rb') as f_in:
        with gzip.open(compressed_path, 'wb', compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove uncompressed file
    source_path.unlink()
    
    # Log compression stats
    original_size = source_path.stat().st_size if source_path.exists() else 0
    compressed_size = compressed_path.stat().st_size
    compression_ratio = (1 - compressed_size / original_size) * 100 if original_size else 0
    logger.info(f"Compression complete: {compression_ratio:.1f}% reduction")
    
    return compressed_path


def upload_to_s3(local_path: Path, bucket_name: str, s3_key: str) -> bool:
    """Upload backup to S3/R2."""
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Get S3 client with endpoint URL for R2 compatibility
        endpoint_url = os.environ.get('S3_ENDPOINT_URL')
        s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        
        logger.info(f"Uploading {local_path.name} to S3 bucket {bucket_name}")
        s3.upload_file(
            str(local_path),
            bucket_name,
            s3_key,
            ExtraArgs={
                'ServerSideEncryption': 'AES256',
                'StorageClass': 'STANDARD_IA'  # Infrequent access for cost savings
            }
        )
        logger.info("Upload completed successfully")
        return True
    except ImportError:
        logger.error("boto3 not installed. Install with: pip install boto3")
        return False
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        return False


def cleanup_old_backups(directory: Path, retention_days: int) -> int:
    """Remove backups older than retention period."""
    if retention_days <= 0:
        return 0
    
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    deleted_count = 0
    
    for backup_file in directory.glob('*_backup_*.sql*'):
        file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
        if file_mtime < cutoff_date:
            try:
                backup_file.unlink()
                logger.info(f"Deleted old backup: {backup_file.name}")
                deleted_count += 1
            except OSError as e:
                logger.warning(f"Failed to delete {backup_file}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} old backups")
    return deleted_count


def verify_backup(backup_path: Path) -> bool:
    """Verify backup file integrity."""
    if not backup_path.exists():
        logger.error(f"Backup file not found: {backup_path}")
        return False
    
    file_size = backup_path.stat().st_size
    if file_size == 0:
        logger.error("Backup file is empty")
        return False
    
    # For compressed files, try to decompress
    if backup_path.suffix == '.gz':
        try:
            with gzip.open(backup_path, 'rb') as f:
                # Read first 1KB to verify it's valid
                f.read(1024)
            logger.info("Backup verification passed")
            return True
        except gzip.BadGzipFile:
            logger.error("Backup file is corrupted (bad gzip)")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Bunoraa Database Backup')
    parser.add_argument('--upload-to-s3', action='store_true', 
                        help='Upload backup to S3')
    parser.add_argument('--retention-days', type=int, 
                        default=int(os.environ.get('BACKUP_RETENTION_DAYS', 30)),
                        help='Days to retain backups (default: 30)')
    parser.add_argument('--no-compression', action='store_true',
                        help='Skip compression')
    args = parser.parse_args()
    
    # Get database URL
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Parse database config
    try:
        db_config = parse_database_url(database_url)
    except Exception as e:
        logger.error(f"Failed to parse DATABASE_URL: {e}")
        sys.exit(1)
    
    # Create backup
    backup_dir = get_backup_dir()
    backup_filename = create_backup_filename(db_config['database'])
    backup_path = backup_dir / backup_filename
    
    # Run backup
    if not run_pg_dump(db_config, backup_path):
        sys.exit(1)
    
    # Verify backup
    if not verify_backup(backup_path):
        sys.exit(1)
    
    # Compress if not disabled
    if not args.no_compression:
        backup_path = compress_backup(backup_path)
    
    # Upload to S3 if requested
    if args.upload_to_s3:
        bucket_name = os.environ.get('S3_BUCKET')
        if not bucket_name:
            logger.error("S3_BUCKET environment variable not set")
        else:
            s3_key = f"backups/postgresql/{backup_path.name}"
            upload_to_s3(backup_path, bucket_name, s3_key)
    
    # Cleanup old backups
    cleanup_old_backups(backup_dir, args.retention_days)
    
    logger.info(f"Backup completed: {backup_path}")
    print(f"BACKUP_PATH={backup_path}")


if __name__ == '__main__':
    main()
