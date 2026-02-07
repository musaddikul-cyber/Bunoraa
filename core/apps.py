from django.apps import AppConfig, apps


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'
    verbose_name = 'Core Configuration'

    def ready(self):
        """Ensure all FileField/ImageField use the configured default storage at runtime.

        Some models (like products) explicitly set an S3 storage instance on their
        fields (via migration). Others rely on Django's DEFAULT_FILE_STORAGE. In
        production the DEFAULT_FILE_STORAGE is set to the S3/R2 backend but
        fields may still use a filesystem storage instance if they were created
        earlier. This hook sets a runtime storage instance for all file fields
        to avoid admin uploads being saved locally.
        """
        try:
            from django.conf import settings
            from django.core.files.storage import FileSystemStorage
            from django.utils.module_loading import import_string
            from django.db.models import FileField, ImageField

            storage_path = getattr(settings, 'DEFAULT_FILE_STORAGE', None)
            media_root = getattr(settings, 'MEDIA_ROOT', None)
            media_url = getattr(settings, 'MEDIA_URL', None)

            if not storage_path:
                # Nothing to do if no storage configured
                return

            storage_cls = import_string(storage_path)
            # If using a filesystem storage class, instantiate with MEDIA_ROOT
            try:
                cls_name = storage_cls.__name__.lower()
            except Exception:
                cls_name = ''

            if 'filesystemstorage' in cls_name or storage_cls is FileSystemStorage:
                if media_root:
                    storage_instance = storage_cls(location=media_root, base_url=media_url)
                else:
                    storage_instance = storage_cls()
            else:
                storage_instance = storage_cls()

            # Iterate all models and set storage for FileField/ImageField where needed
            for model in apps.get_models():
                for field in model._meta.get_fields():
                    # Only handle concrete file/image fields
                    if isinstance(field, (FileField, ImageField)):
                        try:
                            current_storage = getattr(field, 'storage', None)
                            # If field already uses a non-filesystem storage instance, skip
                            if current_storage and not isinstance(current_storage, FileSystemStorage):
                                continue
                            # Set the runtime storage instance
                            field.storage = storage_instance
                        except Exception:
                            # Ignore any fields we can't modify (deprecated/virtual fields)
                            continue
        except Exception:
            # Silently ignore errors during app readiness to avoid blocking startup
            pass
