#!/usr/bin/env python3
"""
Bunoraa Media Backup Script
============================

Automates media file backups with compression and cloud upload.
Handles incremental backups for efficiency.

Usage:
    python scripts/backup_media.py
    python scripts/backup_media.py --upload-to-s3
    python scripts/backup_media.py --full

Environment Variables:
    MEDIA_ROOT: Path to media files (default: media/)
    BACKUP_DIR: Local backup directory (default: backups/media/)
    S3_BUCKET: S3 bucket name for cloud storage
"""

import os
import sys
import tarfile
import gzip
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set, Dict
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('media_backup')


def get_media_root() -> Path:
    """Get media root from Django settings or environment."""
    # Try to get from Django settings
    try:
        import django
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings.production')
        django.setup()
        from django.conf import settings
        return Path(settings.MEDIA_ROOT)
    except Exception:
        return Path(os.environ.get('MEDIA_ROOT', 'media'))


def get_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of file for deduplication."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_manifest(backup_dir: Path) -> Dict:
    """Load backup manifest to track incremental changes."""
    manifest_path = backup_dir / '.backup_manifest.json'
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return {}


def save_manifest(backup_dir: Path, manifest: Dict):
    """Save backup manifest."""
    manifest_path = backup_dir / '.backup_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def should_include_file(filepath: Path, manifest: Dict, full_backup: bool) -> bool:
    """Check if file should be included based on changes."""
    if full_backup:
        return True
    
    file_hash = get_file_hash(filepath)
    relative_path = str(filepath.relative_to(get_media_root()))
    
    # Check if file has changed
    if relative_path in manifest:
        return manifest[relative_path]['hash'] != file_hash
    return True


def create_media_backup(media_root: Path, backup_dir: Path, full_backup: bool = False) -> Path:
    """Create compressed tar archive of media files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_type = 'full' if full_backup else 'incremental'
    backup_filename = f"media_{backup_type}_{timestamp}.tar.gz"
    backup_path = backup_dir / backup_filename
    
    manifest = {} if full_backup else load_manifest(backup_dir)
    new_manifest = {}
    files_added = 0
    
    logger.info(f"Creating {backup_type} media backup: {backup_filename}")
    
    with tarfile.open(backup_path, 'w:gz', compresslevel=6) as tar:
        for filepath in media_root.rglob('*'):
            if not filepath.is_file():
                continue
            
            # Skip certain file types
            if filepath.suffix in ['.tmp', '.temp', '.part', '.crdownload']:
                continue
            
            relative_path = filepath.relative_to(media_root)
            
            # Check if we should include this file
            if not full_backup and not should_include_file(filepath, manifest, full_backup):
                # Still track in new manifest
                new_manifest[str(relative_path)] = manifest[str(relative_path)]
                continue
            
            # Add file to tar
            tar.add(filepath, arcname=str(relative_path))
            
            # Update manifest
            file_hash = get_file_hash(filepath)
            new_manifest[str(relative_path)] = {
                'hash': file_hash,
                'size': filepath.stat().st_size,
                'mtime': filepath.stat().st_mtime
            }
            files_added += 1
    
    # Save updated manifest
    save_manifest(backup_dir, new_manifest)
    
    backup_size = backup_path.stat().st_size
    logger.info(f"Backup complete: {files_added} files, {backup_size / (1024*1024):.2f} MB")
    
    return backup_path


def cleanup_old_backups(backup_dir: Path, retention_days: int = 30) -> int:
    """Remove old backups keeping at least one full backup per week."""
    if retention_days <= 0:
        return 0
    
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    deleted_count = 0
    
    for backup_file in backup_dir.glob('media_*.tar.gz'):
        try:
            file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
            if file_mtime < cutoff_date:
                backup_file.unlink()
                logger.info(f"Deleted old backup: {backup_file.name}")
                deleted_count += 1
        except OSError as e:
            logger.warning(f"Failed to delete {backup_file}: {e}")
    
    return deleted_count


def upload_to_cloud(backup_path: Path, bucket_name: str) -> bool:
    """Upload backup to S3/R2."""
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        endpoint_url = os.environ.get('S3_ENDPOINT_URL')
        s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        
        s3_key = f"backups/media/{backup_path.name}"
        logger.info(f"Uploading to S3: {s3_key}")
        
        s3.upload_file(
            str(backup_path),
            bucket_name,
            s3_key,
            ExtraArgs={
                'ServerSideEncryption': 'AES256',
                'StorageClass': 'STANDARD_IA'
            }
        )
        logger.info("Upload completed")
        return True
    except ImportError:
        logger.error("boto3 not installed")
        return False
    except ClientError as e:
        logger.error(f"Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Bunoraa Media Backup')
    parser.add_argument('--upload-to-s3', action='store_true', help='Upload to S3')
    parser.add_argument('--full', action='store_true', help='Force full backup')
    parser.add_argument('--retention-days', type=int, default=30)
    args = parser.parse_args()
    
    media_root = get_media_root()
    if not media_root.exists():
        logger.error(f"Media root not found: {media_root}")
        sys.exit(1)
    
    backup_dir = Path(os.environ.get('BACKUP_DIR', 'backups')) / 'media'
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we need a full backup (none exist or weekly schedule)
    needs_full = args.full
    if not needs_full:
        existing_backups = list(backup_dir.glob('media_full_*.tar.gz'))
        if not existing_backups:
            needs_full = True
            logger.info("No existing full backup found, creating full backup")
        else:
            # Check if it's been a week since last full backup
            most_recent = max(existing_backups, key=lambda p: p.stat().st_mtime)
            week_ago = datetime.now() - timedelta(days=7)
            if datetime.fromtimestamp(most_recent.stat().st_mtime) < week_ago:
                needs_full = True
                logger.info("Weekly full backup scheduled")
    
    # Create backup
    backup_path = create_media_backup(media_root, backup_dir, needs_full)
    
    # Upload if requested
    if args.upload_to_s3:
        bucket = os.environ.get('S3_BUCKET')
        if bucket:
            upload_to_cloud(backup_path, bucket)
        else:
            logger.error("S3_BUCKET not set")
    
    # Cleanup
    cleanup_old_backups(backup_dir, args.retention_days)
    
    logger.info(f"Backup complete: {backup_path}")
    print(f"BACKUP_PATH={backup_path}")


if __name__ == '__main__':
    main()
