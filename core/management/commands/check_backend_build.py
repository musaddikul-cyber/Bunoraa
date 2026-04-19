"""
Management command to check all types of backend build errors.

Checks:
- Settings validation
- URL configuration
- Middleware loading
- Database connectivity
- Static files configuration
- Template loading
- Model validation
- Celery configuration
- Cache backend
- Email backend
- Storage backend
"""
import sys
import importlib
import traceback
from typing import List, Dict, Any

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.urls import get_resolver, URLResolver
from django.core.checks import registry as checks_registry
from django.db import connection, connections, DatabaseError
from django.core.cache import cache
from django.template import engines
from django.template.loader import get_template
from django.contrib.sites.models import Site


class BuildCheckError:
    """Represents a build check error."""
    
    def __init__(self, category: str, message: str, details: str = "", fix_suggestion: str = ""):
        self.category = category
        self.message = message
        self.details = details
        self.fix_suggestion = fix_suggestion
    
    def __str__(self):
        return f"[{self.category}] {self.message}"


class Command(BaseCommand):
    help = 'Comprehensive backend build validation - checks settings, URLs, middleware, DB, cache, templates, models'

    def add_arguments(self, parser):
        parser.add_argument(
            '--fail-fast',
            action='store_true',
            help='Stop on first error',
        )
        parser.add_argument(
            '--category',
            type=str,
            choices=['settings', 'urls', 'middleware', 'database', 'cache', 'templates', 'models', 'celery', 'email', 'storage', 'all'],
            default='all',
            help='Specific category to check (default: all)',
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed output',
        )

    def handle(self, *args, **options):
        self.fail_fast = options.get('fail_fast', False)
        self.category = options.get('category', 'all')
        self.verbose = options.get('verbose', False)
        self.errors: List[BuildCheckError] = []
        self.warnings: List[BuildCheckError] = []
        
        self.stdout.write(self.style.HTTP_INFO('🔍 Starting backend build validation...\n'))
        
        # Run checks
        check_methods = {
            'settings': self.check_settings,
            'urls': self.check_urls,
            'middleware': self.check_middleware,
            'database': self.check_database,
            'cache': self.check_cache,
            'templates': self.check_templates,
            'models': self.check_models,
            'celery': self.check_celery,
            'email': self.check_email,
            'storage': self.check_storage,
        }
        
        if self.category == 'all':
            categories = list(check_methods.keys())
        else:
            categories = [self.category]
        
        for cat in categories:
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(self.style.HTTP_INFO(f"Checking {cat.upper()}..."))
            self.stdout.write('='*60)
            try:
                check_methods[cat]()
            except Exception as e:
                self.add_error(cat, f"Exception during {cat} check", str(e), traceback.format_exc())
        
        # Report results
        self.report_results()
        
        # Exit with error code if there are errors
        if self.errors:
            sys.exit(1)

    def add_error(self, category: str, message: str, details: str = "", fix_suggestion: str = ""):
        """Add an error to the list."""
        error = BuildCheckError(category, message, details, fix_suggestion)
        self.errors.append(error)
        self.stdout.write(self.style.ERROR(f"  ❌ {message}"))
        if self.verbose and details:
            self.stdout.write(f"     Details: {details}")
        if fix_suggestion:
            self.stdout.write(self.style.WARNING(f"     💡 {fix_suggestion}"))
        
        if self.fail_fast:
            raise CommandError(f"Fail-fast: {message}")

    def add_warning(self, category: str, message: str, details: str = ""):
        """Add a warning to the list."""
        warning = BuildCheckError(category, message, details)
        self.warnings.append(warning)
        self.stdout.write(self.style.WARNING(f"  ⚠️  {message}"))
        if self.verbose and details:
            self.stdout.write(f"     Details: {details}")

    def add_success(self, message: str):
        """Report success."""
        self.stdout.write(self.style.SUCCESS(f"  ✅ {message}"))

    # ========================================================================
    # SETTINGS CHECK
    # ========================================================================
    def check_settings(self):
        """Validate critical Django settings."""
        critical_settings = [
            'SECRET_KEY',
            'DEBUG',
            'ALLOWED_HOSTS',
            'INSTALLED_APPS',
            'DATABASES',
            'STATIC_URL',
            'MEDIA_URL',
        ]
        
        for setting in critical_settings:
            if not hasattr(settings, setting):
                self.add_error('settings', f"Missing required setting: {setting}")
            elif setting == 'SECRET_KEY' and not getattr(settings, setting):
                self.add_error('settings', 'SECRET_KEY is empty', 
                             fix_suggestion='Set SECRET_KEY environment variable')
            elif setting == 'ALLOWED_HOSTS' and not getattr(settings, setting):
                self.add_warning('settings', 'ALLOWED_HOSTS is empty - may cause 400 errors in production')
        
        # Check for common misconfigurations
        if settings.DEBUG and not settings.ALLOWED_HOSTS:
            self.add_warning('settings', 'DEBUG=True but ALLOWED_HOSTS is empty')
        
        # Check database settings
        if 'default' not in settings.DATABASES:
            self.add_error('settings', "DATABASES missing 'default' key")
        elif not settings.DATABASES['default'].get('ENGINE'):
            self.add_error('settings', "DATABASES['default'] missing ENGINE")
        
        self.add_success(f"Settings validation passed ({len(critical_settings)} critical settings checked)")

    # ========================================================================
    # URLS CHECK
    # ========================================================================
    def check_urls(self):
        """Validate URL configuration."""
        try:
            resolver = get_resolver()
            
            # Check for empty URL patterns
            if not resolver.url_patterns:
                self.add_warning('urls', 'No URL patterns defined in root URLconf')
            
            # Collect all URL patterns for checking
            url_errors = []
            self._check_url_patterns(resolver.url_patterns, url_errors)
            
            if url_errors:
                for error in url_errors[:5]:  # Show first 5
                    self.add_error('urls', error['message'], error.get('details', ''))
                if len(url_errors) > 5:
                    self.add_warning('urls', f"... and {len(url_errors) - 5} more URL errors")
            else:
                self.add_success(f"URL configuration valid ({self._count_urls(resolver.url_patterns)} patterns)")
                
        except Exception as e:
            self.add_error('urls', 'Failed to load URL configuration', str(e), traceback.format_exc())

    def _check_url_patterns(self, patterns, errors, prefix=''):
        """Recursively check URL patterns."""
        for pattern in patterns:
            try:
                if isinstance(pattern, URLResolver):
                    new_prefix = prefix + str(pattern.pattern)
                    self._check_url_patterns(pattern.url_patterns, errors, new_prefix)
                else:
                    # Check if view is importable
                    if hasattr(pattern, 'lookup_str') and pattern.lookup_str:
                        try:
                            module_path, view_name = pattern.lookup_str.rsplit('.', 1)
                            module = importlib.import_module(module_path)
                            getattr(module, view_name)
                        except (ImportError, AttributeError) as e:
                            errors.append({
                                'message': f"Cannot import view: {pattern.lookup_str}",
                                'details': str(e),
                                'url': prefix + str(pattern.pattern)
                            })
            except Exception as e:
                errors.append({
                    'message': f"Error checking URL pattern: {pattern}",
                    'details': str(e)
                })

    def _count_urls(self, patterns) -> int:
        """Count total URL patterns."""
        count = 0
        for pattern in patterns:
            if isinstance(pattern, URLResolver):
                count += self._count_urls(pattern.url_patterns)
            else:
                count += 1
        return count

    # ========================================================================
    # MIDDLEWARE CHECK
    # ========================================================================
    def check_middleware(self):
        """Validate middleware configuration."""
        middleware_errors = []
        
        for middleware_path in settings.MIDDLEWARE:
            try:
                module_path, class_name = middleware_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                middleware_class = getattr(module, class_name)
                
                # Check for required methods
                if not callable(getattr(middleware_class, '__init__', None)):
                    middleware_errors.append(f"{middleware_path}: missing __init__")
                    
            except ImportError as e:
                middleware_errors.append(f"Cannot import {middleware_path}: {e}")
            except AttributeError as e:
                middleware_errors.append(f"Cannot find class {middleware_path}: {e}")
            except ValueError:
                # Not a dotted path, might be a callable
                try:
                    module_path, callable_name = middleware_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    getattr(module, callable_name)
                except Exception as e:
                    middleware_errors.append(f"Cannot load middleware {middleware_path}: {e}")
        
        if middleware_errors:
            for error in middleware_errors[:5]:
                self.add_error('middleware', error)
            if len(middleware_errors) > 5:
                self.add_warning('middleware', f"... and {len(middleware_errors) - 5} more middleware errors")
        else:
            self.add_success(f"All {len(settings.MIDDLEWARE)} middleware classes loaded successfully")

    # ========================================================================
    # DATABASE CHECK
    # ========================================================================
    def check_database(self):
        """Validate database connectivity and configuration."""
        errors = []
        
        for db_alias in connections:
            try:
                conn = connections[db_alias]
                # Test connection
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                
                self.add_success(f"Database '{db_alias}' connection successful")
                
                # Check for pending migrations
                try:
                    from django.core.management import call_command
                    from io import StringIO
                    out = StringIO()
                    call_command('showmigrations', '--plan', database=db_alias, stdout=out, stderr=out)
                    output = out.getvalue()
                    if '[ ]' in output:
                        pending = output.count('[ ]')
                        self.add_warning('database', 
                                       f"Database '{db_alias}' has {pending} pending migrations",
                                       fix_suggestion='Run: python manage.py migrate')
                except Exception as e:
                    if self.verbose:
                        self.add_warning('database', f"Could not check migrations for '{db_alias}': {e}")
                        
            except DatabaseError as e:
                errors.append(f"Database '{db_alias}' connection failed: {e}")
            except Exception as e:
                errors.append(f"Database '{db_alias}' error: {e}")
        
        for error in errors:
            self.add_error('database', error)

    # ========================================================================
    # CACHE CHECK
    # ========================================================================
    def check_cache(self):
        """Validate cache backend."""
        try:
            # Test cache connection
            cache_key = '_backend_build_check_'
            cache.set(cache_key, 'test', 1)
            value = cache.get(cache_key)
            cache.delete(cache_key)
            
            if value == 'test':
                cache_backend = settings.CACHES.get('default', {}).get('BACKEND', 'unknown')
                self.add_success(f"Cache backend working ({cache_backend})")
            else:
                self.add_error('cache', 'Cache get/set test failed', 
                             fix_suggestion='Check cache backend configuration in settings')
                
        except Exception as e:
            self.add_error('cache', f'Cache backend error: {e}',
                         fix_suggestion='Verify CACHES configuration and cache server is running')

    # ========================================================================
    # TEMPLATES CHECK
    # ========================================================================
    def check_templates(self):
        """Validate template configuration."""
        errors = []
        warnings = []
        
        # Check template engines
        for engine in engines.all():
            try:
                # Try to get a simple template
                template = engine.get_template('base.html')
                self.add_success(f"Template engine '{engine.name}' configured")
            except Exception as e:
                # base.html might not exist, that's OK
                if 'base.html' in str(e):
                    self.add_success(f"Template engine '{engine.name}' configured (base.html not found)")
                else:
                    errors.append(f"Template engine error: {e}")
        
        # Check template directories exist
        for template_dir in getattr(settings, 'TEMPLATES', []):
            dirs = template_dir.get('DIRS', [])
            for dir_path in dirs:
                import os
                if not os.path.isdir(dir_path):
                    warnings.append(f"Template directory does not exist: {dir_path}")
        
        for error in errors:
            self.add_error('templates', error)
        for warning in warnings:
            self.add_warning('templates', warning)

    # ========================================================================
    # MODELS CHECK
    # ========================================================================
    def check_models(self):
        """Validate model definitions and check for issues."""
        errors = []
        warnings = []
        
        # Run Django's model checks
        from django.core.checks import run_checks
        try:
            issues = run_checks()
            for issue in issues:
                if issue.is_serious():
                    errors.append(f"{issue.msg}: {issue.hint or 'No hint provided'}")
                else:
                    warnings.append(f"{issue.msg}: {issue.hint or 'No hint provided'}")
        except Exception as e:
            errors.append(f"Error running model checks: {e}")
        
        # Check for invalid model configurations
        from django.apps import apps
        for app_config in apps.get_app_configs():
            for model in app_config.get_models():
                # Check for common model issues
                try:
                    # Try to access the model's default manager
                    _ = model._default_manager
                except Exception as e:
                    errors.append(f"Model {model._meta.label}: {e}")
        
        for error in errors:
            self.add_error('models', error)
        for warning in warnings:
            self.add_warning('models', warning)
        
        if not errors and not warnings:
            self.add_success("All models validated successfully")

    # ========================================================================
    # CELERY CHECK
    # ========================================================================
    def check_celery(self):
        """Validate Celery configuration."""
        try:
            from celery import current_app
            
            # Check broker URL
            broker_url = current_app.conf.broker_url
            if not broker_url:
                self.add_error('celery', 'CELERY_BROKER_URL is not set')
                return
            
            self.add_success(f"Celery broker configured: {broker_url.split('://')[0]}://***")
            
            # Check result backend
            result_backend = current_app.conf.result_backend
            if result_backend:
                self.add_success(f"Celery result backend configured: {result_backend.split('://')[0]}://***")
            else:
                self.add_warning('celery', 'CELERY_RESULT_BACKEND not set - task results will not be stored')
            
            # Try to ping the broker (if possible without blocking)
            try:
                with current_app.connection() as conn:
                    conn.connect()
                    self.add_success("Celery broker connection test passed")
            except Exception as e:
                self.add_warning('celery', f'Could not connect to broker: {e}',
                               fix_suggestion='Ensure Redis/Celery broker is running')
                
        except ImportError:
            self.add_warning('celery', 'Celery not installed - async tasks will not work')
        except Exception as e:
            self.add_error('celery', f'Celery configuration error: {e}')

    # ========================================================================
    # EMAIL CHECK
    # ========================================================================
    def check_email(self):
        """Validate email backend configuration."""
        email_backend = getattr(settings, 'EMAIL_BACKEND', 'django.core.mail.backends.console.EmailBackend')
        
        if 'console' in email_backend or 'file' in email_backend or 'locmem' in email_backend:
            self.add_warning('email', f'Using {email_backend} - emails will not be sent in production')
        elif 'smtp' in email_backend:
            required_settings = ['EMAIL_HOST', 'EMAIL_PORT']
            missing = [s for s in required_settings if not getattr(settings, s, None)]
            if missing:
                self.add_error('email', f'SMTP backend configured but missing: {", ".join(missing)}')
            else:
                self.add_success(f"SMTP email configured: {settings.EMAIL_HOST}:{settings.EMAIL_PORT}")
        else:
            self.add_success(f"Email backend: {email_backend}")

    # ========================================================================
    # STORAGE CHECK
    # ========================================================================    
    def check_storage(self):
        """Validate storage backend configuration."""
        storage_backend = getattr(settings, 'DEFAULT_FILE_STORAGE', 
                                 getattr(settings, 'STORAGES', {}).get('default', {}).get('BACKEND', 'unknown'))
        
        if 'FileSystemStorage' in storage_backend:
            import os
            media_root = getattr(settings, 'MEDIA_ROOT', None)
            if media_root and not os.path.isdir(media_root):
                self.add_warning('storage', f'MEDIA_ROOT does not exist: {media_root}')
            else:
                self.add_success(f"Local file storage configured: {media_root}")
        elif 'S3' in storage_backend or 's3' in storage_backend or 'boto' in storage_backend:
            required = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_STORAGE_BUCKET_NAME']
            missing = [s for s in required if not getattr(settings, s, None)]
            if missing:
                self.add_error('storage', f'S3 storage configured but missing: {", ".join(missing)}')
            else:
                self.add_success(f"S3 storage configured: {settings.AWS_STORAGE_BUCKET_NAME}")
        elif 'storages' in storage_backend:
            # django-storages configuration
            self.add_success(f"External storage configured: {storage_backend}")
        else:
            self.add_success(f"Storage backend: {storage_backend}")

    # ========================================================================
    # REPORT RESULTS
    # ========================================================================
    def report_results(self):
        """Print final report."""
        self.stdout.write("\n")
        self.stdout.write("="*60)
        self.stdout.write(self.style.HTTP_INFO("📊 BACKEND BUILD CHECK SUMMARY"))
        self.stdout.write("="*60)
        
        if self.errors:
            self.stdout.write(self.style.ERROR(f"\n❌ ERRORS: {len(self.errors)}"))
            for error in self.errors:
                self.stdout.write(self.style.ERROR(f"   • [{error.category}] {error.message}"))
        
        if self.warnings:
            self.stdout.write(self.style.WARNING(f"\n⚠️  WARNINGS: {len(self.warnings)}"))
            for warning in self.warnings:
                self.stdout.write(self.style.WARNING(f"   • [{warning.category}] {warning.message}"))
        
        if not self.errors and not self.warnings:
            self.stdout.write(self.style.SUCCESS("\n✅ All checks passed! Backend is ready for deployment."))
        elif not self.errors:
            self.stdout.write(self.style.WARNING("\n⚠️  Checks completed with warnings only."))
        else:
            self.stdout.write(self.style.ERROR("\n❌ Checks completed with errors - fix before deploying."))
        
        self.stdout.write("="*60)
