import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from django.db import transaction
from django.utils.text import slugify

from .models import EnvCategory, EnvValue, EnvVariable


RESTART_REQUIRED_KEYS = {
    "SECRET_KEY",
    "DATABASE_URL",
    "DJANGO_SETTINGS_MODULE",
    "ALLOWED_HOSTS",
    "CORS_ALLOWED_ORIGINS",
    "CSRF_TRUSTED_ORIGINS",
    "ENVIRONMENT",
    "DEBUG",
}


def _default_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "env.schema.yml"


def load_schema(schema_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(schema_path) if schema_path else _default_schema_path()
    if not path.exists():
        raise FileNotFoundError(f"Env schema not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "vars" not in data:
        raise ValueError("Env schema missing 'vars' section")
    return data


def _infer_value_type(var: Dict[str, Any]) -> str:
    if var.get("value_type"):
        return str(var["value_type"])
    values = var.get("values") or {}
    sample = None
    if isinstance(values, dict):
        for value in values.values():
            if value is not None:
                sample = value
                break
    if isinstance(sample, bool):
        return EnvVariable.TYPE_BOOL
    if isinstance(sample, int):
        return EnvVariable.TYPE_INT
    if isinstance(sample, float):
        return EnvVariable.TYPE_FLOAT
    if isinstance(sample, (dict, list)):
        return EnvVariable.TYPE_JSON
    return EnvVariable.TYPE_STRING


def _schema_targets(var: Dict[str, Any]) -> Dict[str, Any]:
    targets = var.get("targets") or {}
    if isinstance(targets, dict):
        return targets
    if isinstance(targets, list):
        return {name: {} for name in targets}
    if isinstance(targets, str):
        return {targets: {}}
    return {}


def _get_env_value(var: Dict[str, Any], env: str) -> Optional[str]:
    values = var.get("values") or {}
    if isinstance(values, dict) and env in values:
        return str(values[env])
    if "default" in var:
        return str(var["default"])
    return None


def _sync_variable(
    var: Dict[str, Any],
    schema_version: int,
    env: str,
    force: bool,
) -> Tuple[EnvVariable, bool]:
    scope = var.get("scope") or "misc"
    category_slug = slugify(scope) or "misc"
    category, created = EnvCategory.objects.get_or_create(
        slug=category_slug,
        defaults={"name": scope.replace("_", " ").title()},
    )
    if not created:
        desired_name = scope.replace("_", " ").title()
        if category.name != desired_name:
            category.name = desired_name
            category.save(update_fields=["name"])

    variable, created = EnvVariable.objects.get_or_create(key=var["key"], defaults={"category": category})
    restart_required = bool(var.get("restart_required")) or var["key"] in RESTART_REQUIRED_KEYS
    runtime_apply = var.get("runtime_apply")
    if runtime_apply is None:
        runtime_apply = not restart_required

    value_type = _infer_value_type(var)
    targets = _schema_targets(var)

    changed = False
    updates = {
        "category": category,
        "description": var.get("description", "") or "",
        "is_secret": bool(var.get("secret", False)),
        "required": bool(var.get("required", False)),
        "restart_required": restart_required,
        "runtime_apply": bool(runtime_apply),
        "value_type": value_type,
        "targets": targets,
        "schema_version": schema_version,
        "is_active": True,
    }

    for field, new_value in updates.items():
        if getattr(variable, field) != new_value:
            setattr(variable, field, new_value)
            changed = True

    if changed:
        variable.save()

    if env:
        value = _get_env_value(var, env)
        if value is not None:
            env_value, _ = EnvValue.objects.get_or_create(variable=variable, environment=env)
            if force or not env_value.has_value():
                env_value.set_value(value, is_secret=variable.is_secret)
                env_value.save()

    return variable, created or changed


@transaction.atomic
def sync_from_schema(
    schema_path: Optional[str],
    env: Optional[str],
    force: bool = False,
    prune: bool = False,
) -> Dict[str, int]:
    schema = load_schema(schema_path)
    schema_version = int(schema.get("version") or 1)
    vars_list: Iterable[Dict[str, Any]] = schema.get("vars") or []

    env_name = (env or "").lower() if env else ""
    touched_keys: List[str] = []
    created_or_updated = 0

    for var in vars_list:
        variable, changed = _sync_variable(var, schema_version, env_name, force=force)
        touched_keys.append(variable.key)
        if changed:
            created_or_updated += 1

    if prune:
        inactive = EnvVariable.objects.exclude(key__in=touched_keys)
        inactive.update(is_active=False)

    return {
        "variables_processed": len(touched_keys),
        "variables_changed": created_or_updated,
    }
