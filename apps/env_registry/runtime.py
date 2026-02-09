import json
import os
from typing import Any

from django.conf import settings

from .models import EnvValue


def _cast_value(raw: str, value_type: str) -> Any:
    if raw is None:
        return None
    if value_type == "bool":
        return str(raw).lower() in ("1", "true", "yes", "on")
    if value_type == "int":
        try:
            return int(raw)
        except Exception:
            return 0
    if value_type == "float":
        try:
            return float(raw)
        except Exception:
            return 0.0
    if value_type == "json":
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


def apply_runtime_overrides(environment: str) -> int:
    if not environment:
        return 0

    env_key = environment.lower()
    updated = 0

    values = (
        EnvValue.objects.select_related("variable")
        .filter(
            environment=env_key,
            variable__is_active=True,
            variable__runtime_apply=True,
            variable__restart_required=False,
        )
    )

    for env_value in values:
        value = env_value.get_value()
        if value is None:
            continue
        key = env_value.variable.key
        os.environ[key] = str(value)

        if hasattr(settings, key):
            casted = _cast_value(value, env_value.variable.value_type)
            try:
                setattr(settings, key, casted)
                updated += 1
            except Exception:
                continue

    return updated
