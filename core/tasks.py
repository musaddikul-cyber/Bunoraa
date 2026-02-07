"""
Celery tasks for scheduled backups and maintenance
"""
import os
import json
import gzip
import logging
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

import boto3
from botocore.config import Config as BotoConfig
from celery import shared_task
from django.conf import settings
from django.core import management
from django.utils import timezone

logger = logging.getLogger('bunoraa.backups')


def get_r2_client():
    """Get Cloudflare R2 client."""
    return boto3.client(
        's3',
        endpoint_url=getattr(settings, 'AWS_S3_ENDPOINT_URL', ''),
        aws_access_key_id=getattr(settings, 'AWS_ACCESS_KEY_ID', ''),
        aws_secret_access_key=getattr(settings, 'AWS_SECRET_ACCESS_KEY', ''),
        region_name='auto',
        config=BotoConfig(
            signature_version='s3v4',
            retries={'max_attempts': 3}
        )
    )


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    autoretry_for=(Exception,),
    acks_late=True,
)
def backup_database_to_r2(self):
    """
    Backup database to Cloudflare R2.
    Creates a compressed JSON dump of important models.
    """
    logger.info("Starting database backup to R2...")
    
    try:
        backup_bucket = getattr(settings, 'R2_BACKUP_BUCKET', 'bunoraa-backups')
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        
        # Models to backup
        models_to_backup = [
            'accounts.user',
            'accounts.address',
            'catalog.product',
            'catalog.tag',
            'catalog.attribute',
            'catalog.category',
            'orders.order',
            'orders.orderitem',
            'payments.payment',
            'promotions.coupon',
            'reviews.review',
            'pages.sitesettings',
            'i18n.currency',
            'i18n.exchangerate',
        ]
        
        # Create backup
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            backup_file = f.name
            
            # Use Django's dumpdata command
            management.call_command(
                'dumpdata',
                *models_to_backup,
                output=backup_file,
                indent=2,
                verbosity=0,
            )
        
        # Compress the backup
        compressed_file = f"{backup_file}.gz"
        with open(backup_file, 'rb') as f_in:
            with gzip.open(compressed_file, 'wb') as f_out:
                f_out.writelines(f_in)
        
        # Upload to R2
        r2_client = get_r2_client()
        r2_key = f"database/bunoraa_backup_{timestamp}.json.gz"
        
        with open(compressed_file, 'rb') as f:
            r2_client.upload_fileobj(
                f,
                backup_bucket,
                r2_key,
                ExtraArgs={
                    'ContentType': 'application/gzip',
                    'Metadata': {
                        'backup-type': 'database',
                        'timestamp': timestamp,
                    }
                }
            )
        
        # Cleanup local files
        os.unlink(backup_file)
        os.unlink(compressed_file)
        
        logger.info(f"Database backup completed: {r2_key}")
        
        # Cleanup old backups (keep last 30 days)
        cleanup_old_backups.delay(backup_bucket, 'database/', days=30)
        
        return {
            'status': 'success',
            'key': r2_key,
            'timestamp': timestamp,
        }
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        raise


@shared_task(
    bind=True,
    max_retries=3,
)
def backup_media_to_r2(self):
    """
    Sync media files to R2 backup bucket.
    Creates an incremental backup of media files.
    """
    logger.info("Starting media backup to R2...")
    
    try:
        backup_bucket = getattr(settings, 'R2_BACKUP_BUCKET', 'bunoraa-backups')
        media_bucket = getattr(settings, 'AWS_STORAGE_BUCKET_NAME', 'bunoraa-media')
        
        r2_client = get_r2_client()
        
        # List objects in media bucket
        paginator = r2_client.get_paginator('list_objects_v2')
        
        copied_count = 0
        for page in paginator.paginate(Bucket=media_bucket):
            for obj in page.get('Contents', []):
                source_key = obj['Key']
                dest_key = f"media/{source_key}"
                
                # Copy to backup bucket
                try:
                    r2_client.copy_object(
                        Bucket=backup_bucket,
                        Key=dest_key,
                        CopySource={'Bucket': media_bucket, 'Key': source_key},
                    )
                    copied_count += 1
                except Exception as e:
                    logger.warning(f"Failed to copy {source_key}: {e}")
        
        logger.info(f"Media backup completed: {copied_count} files copied")
        
        return {
            'status': 'success',
            'files_copied': copied_count,
        }
        
    except Exception as e:
        logger.error(f"Media backup failed: {e}")
        raise


@shared_task
def cleanup_old_backups(bucket: str, prefix: str, days: int = 30):
    """
    Remove backups older than specified days.
    """
    logger.info(f"Cleaning up old backups from {bucket}/{prefix}...")
    
    try:
        r2_client = get_r2_client()
        cutoff_date = timezone.now() - timedelta(days=days)
        
        paginator = r2_client.get_paginator('list_objects_v2')
        
        deleted_count = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date.replace(tzinfo=None):
                    r2_client.delete_object(Bucket=bucket, Key=obj['Key'])
                    deleted_count += 1
                    logger.debug(f"Deleted old backup: {obj['Key']}")
        
        logger.info(f"Cleanup completed: {deleted_count} old backups deleted")
        
        return {
            'status': 'success',
            'deleted_count': deleted_count,
        }
        
    except Exception as e:
        logger.error(f"Backup cleanup failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def update_exchange_rates():
    """
    Update currency exchange rates from external API.
    """
    logger.info("Updating exchange rates...")
    
    try:
        import requests
        from apps.i18n.models import Currency, ExchangeRate
        
        # Free exchange rate API
        api_key = getattr(settings, 'EXCHANGE_RATE_API_KEY', '')
        base_currency = 'BDT'
        
        # Use exchangerate-api.com or similar
        if api_key:
            url = f'https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}'
        else:
            # Fallback to free API (limited requests)
            url = f'https://api.exchangerate-api.com/v4/latest/{base_currency}'
        
        response = requests.get(url, timeout=30)
        data = response.json()
        
        rates = data.get('rates') or data.get('conversion_rates', {})
        
        # Update exchange rates
        base = Currency.objects.filter(code=base_currency).first()
        
        if not base:
            logger.warning(f"Base currency {base_currency} not found")
            return {'status': 'error', 'message': 'Base currency not found'}
        
        updated = 0
        for code, rate in rates.items():
            target = Currency.objects.filter(code=code).first()
            if target:
                ExchangeRate.objects.update_or_create(
                    from_currency=base,
                    to_currency=target,
                    defaults={
                        'rate': rate,
                        'source': 'exchangerate-api',
                    }
                )
                updated += 1
        
        logger.info(f"Exchange rates updated: {updated} currencies")
        
        return {
            'status': 'success',
            'updated': updated,
        }
        
    except Exception as e:
        logger.error(f"Exchange rate update failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def aggregate_daily_analytics():
    """
    Aggregate daily analytics data.
    """
    logger.info("Aggregating daily analytics...")
    
    try:
        from apps.analytics.models import (
            PageView, ProductView, SearchQuery, CartEvent, DailyStat
        )
        from apps.orders.models import Order
        from apps.accounts.models import User
        
        yesterday = timezone.now().date() - timedelta(days=1)
        
        # Calculate metrics
        page_views = PageView.objects.filter(
            created_at__date=yesterday
        ).count()
        
        unique_visitors = PageView.objects.filter(
            created_at__date=yesterday
        ).values('session_key').distinct().count()
        
        product_views = ProductView.objects.filter(
            created_at__date=yesterday
        ).count()
        
        cart_events = CartEvent.objects.filter(
            created_at__date=yesterday
        )
        
        cart_additions = cart_events.filter(event_type='add').count()
        checkout_starts = cart_events.filter(event_type='checkout_start').count()
        checkout_completions = cart_events.filter(event_type='checkout_complete').count()
        
        orders = Order.objects.filter(created_at__date=yesterday)
        orders_count = orders.count()
        orders_revenue = sum(o.total for o in orders) if orders.exists() else 0
        avg_order_value = orders_revenue / orders_count if orders_count > 0 else 0
        
        new_users = User.objects.filter(
            created_at__date=yesterday
        ).count()
        
        # Calculate conversion rate
        conversion_rate = (checkout_completions / unique_visitors * 100) if unique_visitors > 0 else 0
        
        # Calculate cart abandonment
        abandonment_rate = 0
        if checkout_starts > 0:
            abandonment_rate = ((checkout_starts - checkout_completions) / checkout_starts * 100)
        
        # Create or update daily stat
        DailyStat.objects.update_or_create(
            date=yesterday,
            defaults={
                'page_views': page_views,
                'unique_visitors': unique_visitors,
                'product_views': product_views,
                'products_added_to_cart': cart_additions,
                'orders_count': orders_count,
                'orders_revenue': orders_revenue,
                'average_order_value': avg_order_value,
                'checkout_starts': checkout_starts,
                'checkout_completions': checkout_completions,
                'conversion_rate': conversion_rate,
                'cart_abandonment_rate': abandonment_rate,
                'new_registrations': new_users,
            }
        )
        
        logger.info(f"Daily analytics aggregated for {yesterday}")
        
        return {
            'status': 'success',
            'date': str(yesterday),
            'page_views': page_views,
            'orders': orders_count,
        }
        
    except Exception as e:
        logger.error(f"Analytics aggregation failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def update_user_behavior_profiles():
    """
    Update user behavior profiles for ML recommendations.
    """
    logger.info("Updating user behavior profiles...")
    
    try:
        from apps.accounts.behavior_models import UserBehaviorProfile
        from apps.accounts.models import User
        from apps.analytics.models import ProductView, SearchQuery
        from apps.orders.models import Order
        
        users_updated = 0
        
        for user in User.objects.filter(is_active=True)[:1000]:  # Batch process
            profile, created = UserBehaviorProfile.objects.get_or_create(user=user)
            
            # Update session stats
            profile.total_page_views = user.page_views.count()
            
            # Update product engagement
            profile.products_viewed = ProductView.objects.filter(user=user).count()
            
            # Update order stats
            orders = Order.objects.filter(user=user, is_deleted=False)
            profile.total_orders = orders.count()
            profile.total_spent = sum(o.total for o in orders)
            
            if profile.total_orders > 0:
                profile.avg_order_value = profile.total_spent / profile.total_orders
                profile.last_purchase_date = orders.order_by('-created_at').first().created_at
            
            # Update search stats
            profile.search_count = SearchQuery.objects.filter(user=user).count()
            
            # Update last active
            profile.last_active = timezone.now()
            
            # Recalculate scores
            profile.update_engagement_score()
            profile.update_recency_score()
            
            profile.save()
            users_updated += 1
        
        logger.info(f"Updated {users_updated} user behavior profiles")
        
        return {
            'status': 'success',
            'users_updated': users_updated,
        }
        
    except Exception as e:
        logger.error(f"Behavior profile update failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def cleanup_expired_sessions():
    """
    Clean up expired user sessions.
    """
    logger.info("Cleaning up expired sessions...")
    
    try:
        from django.contrib.sessions.models import Session
        
        expired = Session.objects.filter(expire_date__lt=timezone.now())
        count = expired.count()
        expired.delete()
        
        logger.info(f"Cleaned up {count} expired sessions")
        
        return {
            'status': 'success',
            'deleted': count,
        }
        
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def warm_cache():
    """
    Pre-warm cache for common queries.
    """
    logger.info("Warming cache...")
    
    try:
        from django.core.cache import cache
        from apps.catalog.models import Product, Category
        
        # Cache popular products
        popular = Product.objects.filter(
            is_active=True, is_deleted=False
        ).order_by('-sales_count')[:20]
        
        cache.set('popular_products', list(popular.values_list('id', flat=True)), timeout=3600)
        
        # Cache category tree
        categories = Category.objects.filter(is_visible=True, is_deleted=False).values('id', 'name', 'slug', 'parent_id')
        cache.set('category_tree', list(categories), timeout=3600)
        
        # Cache featured products
        featured = Product.objects.filter(
            is_active=True, is_deleted=False, is_featured=True
        )[:12]
        cache.set('featured_products', list(featured.values_list('id', flat=True)), timeout=3600)
        
        logger.info("Cache warmed successfully")
        
        return {'status': 'success'}
        
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def sync_media_incremental():
    """
    Incrementally sync new/modified media files to backup bucket.
    Only syncs files modified in the last 24 hours.
    """
    logger.info("Starting incremental media sync...")
    
    try:
        backup_bucket = getattr(settings, 'R2_BACKUP_BUCKET', 'bunoraa-backups')
        media_bucket = getattr(settings, 'AWS_STORAGE_BUCKET_NAME', 'bunoraa-media')
        
        r2_client = get_r2_client()
        cutoff_time = timezone.now() - timedelta(hours=24)
        
        # List objects in media bucket modified in last 24 hours
        paginator = r2_client.get_paginator('list_objects_v2')
        
        synced_count = 0
        for page in paginator.paginate(Bucket=media_bucket):
            for obj in page.get('Contents', []):
                # Check if recently modified
                if obj['LastModified'].replace(tzinfo=timezone.utc) > cutoff_time:
                    source_key = obj['Key']
                    dest_key = f"media/{source_key}"
                    
                    try:
                        r2_client.copy_object(
                            Bucket=backup_bucket,
                            Key=dest_key,
                            CopySource={'Bucket': media_bucket, 'Key': source_key},
                        )
                        synced_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to sync {source_key}: {e}")
        
        logger.info(f"Incremental media sync completed: {synced_count} files synced")
        
        return {
            'status': 'success',
            'files_synced': synced_count,
        }
        
    except Exception as e:
        logger.error(f"Incremental media sync failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task(
    bind=True,
    max_retries=3,
)
def update_ml_models(self):
    """
    Update ML models for product recommendations and user segmentation.
    Runs the training pipeline with fresh data.
    """
    logger.info("Starting ML model update...")
    
    try:
        import subprocess
        import sys
        
        # Path to training script
        train_script = Path(settings.BASE_DIR) / 'scripts' / 'train_ml_models.py'
        output_dir = getattr(settings, 'ML_MODELS_DIR', Path(settings.BASE_DIR) / 'outputs' / 'ml_models')
        
        if not train_script.exists():
            logger.warning("ML training script not found")
            return {'status': 'skipped', 'message': 'Training script not found'}
        
        # Run training script
        result = subprocess.run(
            [sys.executable, str(train_script), '--output-dir', str(output_dir)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("ML model training completed successfully")
            
            # Upload models to R2 backup
            backup_models_to_r2.delay(str(output_dir))
            
            return {
                'status': 'success',
                'output': result.stdout[-1000:],  # Last 1000 chars
            }
        else:
            logger.error(f"ML training failed: {result.stderr}")
            return {
                'status': 'error',
                'output': result.stderr[-1000:],
            }
            
    except subprocess.TimeoutExpired:
        logger.error("ML model training timed out")
        return {'status': 'error', 'message': 'Training timed out'}
    except Exception as e:
        logger.error(f"ML model update failed: {e}")
        raise


@shared_task
def backup_models_to_r2(models_dir: str):
    """
    Backup ML models to R2.
    """
    logger.info(f"Backing up ML models from {models_dir}...")
    
    try:
        backup_bucket = getattr(settings, 'R2_BACKUP_BUCKET', 'bunoraa-backups')
        r2_client = get_r2_client()
        
        models_path = Path(models_dir)
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        
        uploaded = 0
        for model_file in models_path.glob('*.pkl'):
            r2_key = f"ml_models/{timestamp}/{model_file.name}"
            
            with open(model_file, 'rb') as f:
                r2_client.upload_fileobj(
                    f,
                    backup_bucket,
                    r2_key,
                    ExtraArgs={
                        'ContentType': 'application/octet-stream',
                        'Metadata': {
                            'model-type': model_file.stem,
                            'timestamp': timestamp,
                        }
                    }
                )
            uploaded += 1
        
        logger.info(f"ML models backup completed: {uploaded} models uploaded")
        
        return {
            'status': 'success',
            'models_uploaded': uploaded,
        }
        
    except Exception as e:
        logger.error(f"ML models backup failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def generate_data_export(user_id: int, export_type: str = 'all'):
    """
    Generate GDPR-compliant data export for a user.
    """
    logger.info(f"Generating data export for user {user_id}...")
    
    try:
        from apps.accounts.models import User
        
        user = User.objects.get(id=user_id)
        
        export_data = {
            'export_date': timezone.now().isoformat(),
            'user_id': user_id,
            'export_type': export_type,
            'data': {}
        }
        
        # Personal information
        if export_type in ['all', 'personal']:
            export_data['data']['personal'] = {
                'email': user.email,
                'phone': user.phone_number,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'created_at': user.created_at.isoformat() if user.created_at else None,
            }
            
            # Addresses
            addresses = []
            for addr in user.addresses.all():
                addresses.append({
                    'label': addr.label,
                    'address_line_1': addr.address_line_1,
                    'address_line_2': addr.address_line_2,
                    'city': addr.city,
                    'division': addr.division,
                    'postal_code': addr.postal_code,
                })
            export_data['data']['addresses'] = addresses
        
        # Orders
        if export_type in ['all', 'orders']:
            from apps.orders.models import Order
            
            orders = []
            for order in Order.objects.filter(user=user):
                order_data = {
                    'order_number': order.order_number,
                    'status': order.status,
                    'total': str(order.total),
                    'created_at': order.created_at.isoformat(),
                    'items': []
                }
                for item in order.items.all():
                    order_data['items'].append({
                        'product': item.product.name if item.product else 'Deleted Product',
                        'quantity': item.quantity,
                        'price': str(item.price),
                    })
                orders.append(order_data)
            export_data['data']['orders'] = orders
        
        # Reviews
        if export_type in ['all', 'reviews']:
            from apps.reviews.models import Review
            
            reviews = []
            for review in Review.objects.filter(user=user):
                reviews.append({
                    'product': review.product.name if review.product else 'Deleted Product',
                    'rating': review.rating,
                    'comment': review.comment,
                    'created_at': review.created_at.isoformat(),
                })
            export_data['data']['reviews'] = reviews
        
        # Create and upload export file
        backup_bucket = getattr(settings, 'R2_BACKUP_BUCKET', 'bunoraa-backups')
        r2_client = get_r2_client()
        
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        r2_key = f"data_exports/user_{user_id}_{timestamp}.json.gz"
        
        # Compress and upload
        export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
        compressed = gzip.compress(export_json.encode('utf-8'))
        
        from io import BytesIO
        r2_client.upload_fileobj(
            BytesIO(compressed),
            backup_bucket,
            r2_key,
            ExtraArgs={
                'ContentType': 'application/gzip',
                'Metadata': {
                    'user-id': str(user_id),
                    'export-type': export_type,
                }
            }
        )
        
        # Generate signed URL for download
        download_url = r2_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': backup_bucket, 'Key': r2_key},
            ExpiresIn=86400,  # 24 hours
        )
        
        logger.info(f"Data export generated for user {user_id}: {r2_key}")
        
        # Notify user
        from apps.notifications.tasks import send_notification
        send_notification.delay(
            user_id=user_id,
            notification_type='data_export_ready',
            context={'download_url': download_url},
        )
        
        return {
            'status': 'success',
            'key': r2_key,
            'download_url': download_url,
        }
        
    except User.DoesNotExist:
        logger.error(f"User {user_id} not found")
        return {'status': 'error', 'message': 'User not found'}
    except Exception as e:
        logger.error(f"Data export failed for user {user_id}: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def full_site_backup():
    """
    Perform a complete site backup including database, media, and configuration.
    This is a comprehensive backup task that should be run weekly.
    """
    logger.info("Starting full site backup...")
    
    results = {
        'database': None,
        'media': None,
        'config': None,
    }
    
    try:
        # Database backup
        db_result = backup_database_to_r2()
        results['database'] = db_result
        
        # Media backup
        media_result = backup_media_to_r2()
        results['media'] = media_result
        
        # Configuration backup (settings, env vars, etc.)
        config_result = backup_configuration()
        results['config'] = config_result
        
        logger.info("Full site backup completed")
        
        return {
            'status': 'success',
            'results': results,
        }
        
    except Exception as e:
        logger.error(f"Full site backup failed: {e}")
        return {
            'status': 'partial',
            'results': results,
            'error': str(e),
        }


def backup_configuration():
    """
    Backup site configuration (non-sensitive).
    """
    logger.info("Backing up configuration...")
    
    try:
        backup_bucket = getattr(settings, 'R2_BACKUP_BUCKET', 'bunoraa-backups')
        r2_client = get_r2_client()
        
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        
        # Collect configuration (non-sensitive)
        config_data = {
            'backup_timestamp': timestamp,
            'django_version': __import__('django').__version__,
            'installed_apps': list(settings.INSTALLED_APPS),
            'middleware': list(settings.MIDDLEWARE),
            'database_engine': settings.DATABASES['default']['ENGINE'],
            'languages': list(settings.LANGUAGES),
            'currencies': getattr(settings, 'SUPPORTED_CURRENCIES', []),
            'cache_backend': list(settings.CACHES.keys()),
            'storage_backend': getattr(settings, 'DEFAULT_FILE_STORAGE', ''),
        }
        
        # Upload
        config_json = json.dumps(config_data, indent=2)
        r2_key = f"config/bunoraa_config_{timestamp}.json"
        
        from io import BytesIO
        r2_client.upload_fileobj(
            BytesIO(config_json.encode('utf-8')),
            backup_bucket,
            r2_key,
            ExtraArgs={'ContentType': 'application/json'}
        )
        
        logger.info(f"Configuration backup completed: {r2_key}")
        
        return {
            'status': 'success',
            'key': r2_key,
        }
        
    except Exception as e:
        logger.error(f"Configuration backup failed: {e}")
        return {'status': 'error', 'message': str(e)}


# =============================================================================
# EMAIL TASKS
# =============================================================================

@shared_task(
    bind=True,
    name='core.send_email',
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def send_email_task(
    self,
    to,
    subject,
    template=None,
    context=None,
    html_body=None,
    text_body=None,
    from_email=None,
    from_name=None,
    reply_to=None,
    cc=None,
    bcc=None,
    attachments=None,
    tags=None,
    **kwargs
):
    """
    Send email asynchronously via Celery.
    
    Args:
        to: Email recipient(s)
        subject: Email subject
        template: Path to email template (optional)
        context: Template context dict (optional)
        html_body: HTML email body (optional)
        text_body: Plain text body (optional)
        from_email: Sender email
        from_name: Sender name
        reply_to: Reply-to address
        cc: CC recipients
        bcc: BCC recipients
        attachments: List of attachments
        tags: Email tags for categorization
        
    Returns:
        dict with success status and message_id
    """
    from core.utils.email_service import EmailService, Email
    
    try:
        # Normalize to list
        if isinstance(to, str):
            to = [to]
        
        # Create email object
        email = Email(
            to=to,
            subject=subject,
            template=template,
            context=context or {},
            html_body=html_body or '',
            text_body=text_body or '',
            from_email=from_email,
            from_name=from_name,
            reply_to=reply_to,
            cc=cc or [],
            bcc=bcc or [],
            attachments=attachments or [],
            tags=tags or [],
        )
        
        result = EmailService.send_email(email)
        
        if not result.success:
            logger.error(
                f"Email send failed: to={to}, subject={subject[:50]}, "
                f"error={result.error}, attempts={result.attempts}"
            )
            raise Exception(result.error)
        
        logger.info(
            f"Email sent successfully: to={to}, subject={subject[:50]}, "
            f"message_id={result.message_id}, provider={result.provider.value}"
        )
        
        return {
            'success': True,
            'message_id': result.message_id,
            'provider': result.provider.value,
            'attempts': result.attempts,
        }
        
    except Exception as e:
        logger.exception(f"Email task failed: {e}")
        raise self.retry(exc=e)


@shared_task(
    bind=True,
    name='core.send_bulk_emails',
    max_retries=2,
    default_retry_delay=120,
)
def send_bulk_emails_task(
    self,
    emails_data: list,
    batch_size: int = 50,
    delay_between_batches: float = 1.0,
):
    """
    Send multiple emails in batches asynchronously.
    
    Args:
        emails_data: List of dicts with email data
        batch_size: Emails per batch
        delay_between_batches: Delay between batches (seconds)
    
    Returns:
        Summary of sent/failed emails
    """
    import time
    from core.utils.email_service import EmailService, Email
    
    results = {
        'total': len(emails_data),
        'sent': 0,
        'failed': 0,
        'errors': [],
    }
    
    for i, email_data in enumerate(emails_data):
        try:
            email = Email(**email_data)
            result = EmailService.send_email(email)
            
            if result.success:
                results['sent'] += 1
            else:
                results['failed'] += 1
                results['errors'].append({
                    'to': email_data.get('to'),
                    'error': result.error,
                })
        except Exception as e:
            results['failed'] += 1
            results['errors'].append({
                'to': email_data.get('to'),
                'error': str(e),
            })
        
        # Add delay between batches
        if (i + 1) % batch_size == 0 and i + 1 < len(emails_data):
            time.sleep(delay_between_batches)
    
    logger.info(
        f"Bulk email complete: sent={results['sent']}, "
        f"failed={results['failed']}, total={results['total']}"
    )
    
    return results


@shared_task(name='core.send_templated_email')
def send_templated_email_task(
    to,
    template_name: str,
    context: dict = None,
    **kwargs
):
    """
    Send a templated email (convenience task).
    
    Template names are mapped to actual template paths.
    
    Args:
        to: Recipient(s)
        template_name: Template identifier (e.g., 'welcome', 'order_confirmation')
        context: Template context
        **kwargs: Additional email parameters
    """
    from core.utils.email_service import EmailService
    
    # Template mapping
    TEMPLATE_MAP = {
        'welcome': ('Welcome to Bunoraa!', 'emails/welcome.html'),
        'order_confirmation': ('Order Confirmation', 'emails/order_confirmation.html'),
        'order_shipped': ('Your Order Has Shipped!', 'emails/order_shipped.html'),
        'password_reset': ('Reset Your Password', 'emails/password_reset.html'),
        'account_verified': ('Account Verified', 'emails/account_verified.html'),
        'newsletter': ('Bunoraa Newsletter', 'emails/newsletter.html'),
        'contact_reply': ('Re: Your Inquiry', 'emails/contact_reply.html'),
        'review_request': ('Share Your Feedback', 'emails/review_request.html'),
        'abandoned_cart': ('You Left Something Behind', 'emails/abandoned_cart.html'),
        'price_drop': ('Price Drop Alert!', 'emails/price_drop.html'),
        'back_in_stock': ('Back in Stock!', 'emails/back_in_stock.html'),
    }
    
    if template_name not in TEMPLATE_MAP:
        raise ValueError(f"Unknown template: {template_name}")
    
    subject, template = TEMPLATE_MAP[template_name]
    
    # Allow subject override
    if 'subject' in kwargs:
        subject = kwargs.pop('subject')
    
    result = EmailService.send(
        to=to,
        subject=subject,
        template=template,
        context=context or {},
        tags=[template_name],
        **kwargs
    )
    
    return {
        'success': result.success,
        'message_id': result.message_id,
        'error': result.error,
    }
