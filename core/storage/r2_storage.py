"""
Cloudflare R2 Storage Backend for Bunoraa
Provides optimized storage configuration for R2 with CDN support.
"""
from django.conf import settings
from storages.backends.s3boto3 import S3Boto3Storage


class BunoraR2Storage(S3Boto3Storage):
    """
    Custom storage backend for Cloudflare R2.
    Optimized for media file storage with CDN delivery.
    """
    
    def __init__(self, **kwargs):
        # R2 configuration
        kwargs.setdefault('access_key', getattr(settings, 'R2_ACCESS_KEY_ID', ''))
        kwargs.setdefault('secret_key', getattr(settings, 'R2_SECRET_ACCESS_KEY', ''))
        kwargs.setdefault('bucket_name', getattr(settings, 'R2_BUCKET_NAME', 'bunoraa-media'))
        kwargs.setdefault('region_name', 'auto')
        
        # Build endpoint URL from account ID
        account_id = getattr(settings, 'R2_ACCOUNT_ID', '')
        if account_id:
            kwargs.setdefault('endpoint_url', f'https://{account_id}.r2.cloudflarestorage.com')
        
        # Custom domain for CDN delivery
        custom_domain = getattr(settings, 'R2_CUSTOM_DOMAIN', '')
        if custom_domain:
            kwargs.setdefault('custom_domain', custom_domain)
        
        # R2 doesn't support ACLs
        kwargs.setdefault('default_acl', None)
        kwargs.setdefault('querystring_auth', False)
        
        # Signature and addressing
        kwargs.setdefault('signature_version', 's3v4')
        kwargs.setdefault('addressing_style', 'virtual')
        
        # Cache control for optimal CDN performance
        kwargs.setdefault('object_parameters', {
            'CacheControl': 'max-age=31536000, public',
        })
        
        super().__init__(**kwargs)
    
    def url(self, name):
        """
        Return CDN URL for the file.
        """
        custom_domain = getattr(settings, 'R2_CUSTOM_DOMAIN', '')
        if custom_domain:
            # Use custom domain for faster CDN delivery
            return f'https://{custom_domain}/{name}'
        return super().url(name)


class BunoraR2MediaStorage(BunoraR2Storage):
    """
    Storage backend for media files (user uploads).
    """
    location = 'media'
    file_overwrite = False


class BunoraR2StaticStorage(BunoraR2Storage):
    """
    Storage backend for static files.
    Uses longer cache times for immutable assets.
    """
    location = 'static'
    
    def __init__(self, **kwargs):
        kwargs.setdefault('object_parameters', {
            'CacheControl': 'max-age=31536000, public, immutable',
        })
        super().__init__(**kwargs)


class BunoraR2BackupStorage(S3Boto3Storage):
    """
    Storage backend for backups.
    Uses the backup bucket with encryption.
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault('access_key', getattr(settings, 'R2_ACCESS_KEY_ID', ''))
        kwargs.setdefault('secret_key', getattr(settings, 'R2_SECRET_ACCESS_KEY', ''))
        kwargs.setdefault('bucket_name', getattr(settings, 'R2_BACKUP_BUCKET', 'bunoraa-backups'))
        kwargs.setdefault('region_name', 'auto')
        
        account_id = getattr(settings, 'R2_ACCOUNT_ID', '')
        if account_id:
            kwargs.setdefault('endpoint_url', f'https://{account_id}.r2.cloudflarestorage.com')
        
        kwargs.setdefault('default_acl', None)
        kwargs.setdefault('querystring_auth', True)  # Secure URLs for backups
        kwargs.setdefault('signature_version', 's3v4')
        kwargs.setdefault('addressing_style', 'virtual')
        
        super().__init__(**kwargs)
