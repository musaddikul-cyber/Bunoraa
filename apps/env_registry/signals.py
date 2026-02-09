from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import EnvValue
from .runtime import apply_runtime_overrides


@receiver(post_save, sender=EnvValue)
def apply_env_value_runtime(sender, instance, **kwargs):
    env = (settings.ENVIRONMENT or "").lower()
    if not env or instance.environment != env:
        return
    if not getattr(settings, "ENV_REGISTRY_AUTOSYNC_RUNTIME", True):
        return
    apply_runtime_overrides(env)
