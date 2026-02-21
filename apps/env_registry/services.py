import argparse
import os
import re
import secrets
import string
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from dotenv import dotenv_values


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = ROOT_DIR / "config" / "env.schema.yml"


def _load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "targets" not in data or "vars" not in data:
        raise ValueError("Schema must include top-level 'targets' and 'vars'.")
    return data


def _normalize_targets(targets: Any) -> Dict[str, Dict[str, Any]]:
    if targets is None:
        return {}
    if isinstance(targets, str):
        return {targets: {}}
    if isinstance(targets, list):
        return {name: {} for name in targets}
    if isinstance(targets, dict):
        normalized: Dict[str, Dict[str, Any]] = {}
        for name, cfg in targets.items():
            normalized[name] = cfg or {}
        return normalized
    raise ValueError(f"Unsupported targets spec: {targets!r}")


def _extract_scopes(vars_list: List[Dict[str, Any]]) -> List[str]:
    scopes: List[str] = []
    for var in vars_list:
        scope = var.get("scope", "misc")
        if scope not in scopes:
            scopes.append(scope)
    return scopes


def _allowed_keys(vars_list: List[Dict[str, Any]], target_name: str) -> List[str]:
    keys: List[str] = []
    for var in vars_list:
        target_map = _normalize_targets(var.get("targets"))
        if target_name in target_map:
            keys.append(var["key"])
    return keys


def _expand_path(raw: str) -> Tuple[Path | None, bool]:
    expanded = os.path.expandvars(raw)
    unresolved = bool(re.search(r"\$\{[^}]+\}", expanded) or re.search(r"%[^%]+%", expanded))
    if unresolved:
        return None, True
    path = Path(expanded)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path, False


def _parse_targets_arg(raw_targets: List[str] | None) -> List[str]:
    if not raw_targets:
        return []
    names: List[str] = []
    for entry in raw_targets:
        parts = [p.strip() for p in entry.split(",") if p.strip()]
        names.extend(parts)
    return names


def _parse_set_args(raw_sets: List[str] | None) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not raw_sets:
        return overrides
    for item in raw_sets:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value
    return overrides


def _value_is_present(value: Any) -> bool:
    if value is None:
        return False
    return str(value) != ""


def _generate_value(spec: Any) -> str:
    if not spec:
        return ""
    if isinstance(spec, dict):
        spec = spec.get("type") or spec.get("name") or "random"
    if isinstance(spec, str):
        token = spec.lower()
        if token in {"uuid", "uuid4"}:
            return str(uuid.uuid4())
        if token in {"django_secret_key", "secret_key", "random"}:
            alphabet = string.ascii_letters + string.digits
            return "".join(secrets.choice(alphabet) for _ in range(50))
    return str(spec)


def _resolve_value(
    var: Dict[str, Any],
    env: str,
    existing_value: Any,
    overrides: Dict[str, str],
    target_cfg: Dict[str, Any],
) -> Any:
    key = var["key"]
    if key in overrides:
        return overrides[key]

    if var.get("secret") and _value_is_present(existing_value):
        return existing_value

    if "value" in target_cfg:
        return target_cfg["value"]
    if "generateValue" in target_cfg:
        return _generate_value(target_cfg["generateValue"])

    if _value_is_present(existing_value):
        return existing_value

    values = var.get("values") or {}
    if isinstance(values, dict) and env in values:
        return values[env]

    if "default" in var:
        return var["default"]
    return None


def _format_dotenv_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text == "":
        return ""
    if re.search(r"\s|#|\n|\r", text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _scope_title(scope: str) -> str:
    return scope.replace("_", " ").title()


def _read_dotenv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = dotenv_values(path)
    return dict(data)


def _render_envvars_to_map(envvars: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for entry in envvars or []:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if key is None:
            continue
        mapping[str(key)] = entry
    return mapping


def _render_entry_present(entry: Dict[str, Any] | None) -> bool:
    if not entry:
        return False
    if entry.get("sync") is False:
        return True
    if "fromDatabase" in entry or "fromService" in entry:
        return True
    if "value" in entry and _value_is_present(entry.get("value")):
        return True
    return False


def _build_dotenv_content(
    target_name: str,
    env: str,
    vars_list: List[Dict[str, Any]],
    existing: Dict[str, Any],
    overrides: Dict[str, str],
    prune: bool,
    mode: str,
) -> str:
    scopes = _extract_scopes(vars_list)
    allowed = set(_allowed_keys(vars_list, target_name))

    lines: List[str] = []
    for scope in scopes:
        scope_lines: List[str] = []
        for var in vars_list:
            if var.get("scope", "misc") != scope:
                continue
            target_map = _normalize_targets(var.get("targets"))
            if target_name not in target_map:
                continue
            key = var["key"]
            existing_value = existing.get(key)
            if mode == "prune":
                if key not in existing:
                    continue
                value = existing_value
            else:
                value = _resolve_value(
                    var,
                    env,
                    existing_value,
                    overrides,
                    target_map.get(target_name, {}),
                )
            if value is None:
                continue
            scope_lines.append(f"{key}={_format_dotenv_value(value)}")
        if scope_lines:
            lines.append(f"# {_scope_title(scope)}")
            lines.extend(scope_lines)
            lines.append("")

    if not prune:
        unmanaged = []
        for key in sorted(existing.keys()):
            if key in allowed:
                continue
            value = existing.get(key)
            if value is None:
                continue
            unmanaged.append(f"{key}={_format_dotenv_value(value)}")
        if unmanaged:
            lines.append("# Unmanaged")
            lines.extend(unmanaged)
            lines.append("")

    content = "\n".join(lines).rstrip() + "\n"
    return content


def _build_render_envvars(
    target_name: str,
    env: str,
    vars_list: List[Dict[str, Any]],
    existing_envvars: List[Dict[str, Any]],
    overrides: Dict[str, str],
    prune: bool,
    mode: str,
) -> List[Dict[str, Any]]:
    existing_map = _render_envvars_to_map(existing_envvars)
    desired: List[Dict[str, Any]] = []
    allowed = set(_allowed_keys(vars_list, target_name))

    if mode == "prune":
        for entry in existing_envvars:
            key = entry.get("key")
            if key in allowed:
                desired.append(entry)
        return desired

    for var in vars_list:
        target_map = _normalize_targets(var.get("targets"))
        if target_name not in target_map:
            continue
        key = var["key"]
        target_cfg = target_map.get(target_name, {})
        existing_entry = existing_map.get(key)
        existing_value = existing_entry.get("value") if existing_entry else None

        if target_cfg.get("sync") is False:
            desired.append({"key": key, "sync": False})
            continue
        if "fromDatabase" in target_cfg:
            desired.append({"key": key, "fromDatabase": target_cfg["fromDatabase"]})
            continue
        if "fromService" in target_cfg:
            desired.append({"key": key, "fromService": target_cfg["fromService"]})
            continue
        if "generateValue" in target_cfg:
            value = _generate_value(target_cfg["generateValue"])
            desired.append({"key": key, "value": str(value)})
            continue

        if var.get("secret"):
            desired.append({"key": key, "sync": False})
            continue

        value = _resolve_value(var, env, existing_value, overrides, target_cfg)
        if value is None:
            continue
        desired.append({"key": key, "value": str(value)})

    if not prune:
        unmanaged_entries = [entry for k, entry in existing_map.items() if k not in allowed]
        unmanaged_entries.sort(key=lambda e: str(e.get("key", "")))
        desired.extend(unmanaged_entries)

    return desired


def _write_text(path: Path, content: str, dry_run: bool) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing == content:
        return False
    if dry_run:
        print(f"[dry-run] Would update {path}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    print(f"Updated {path}")
    return True


def _dump_yaml(data: Any) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        width=140,
    )


def _sync_dotenv_target(
    target: Dict[str, Any],
    env: str,
    vars_list: List[Dict[str, Any]],
    overrides: Dict[str, str],
    prune: bool,
    dry_run: bool,
    mode: str,
) -> bool:
    path, unresolved = _expand_path(target["path"])
    if unresolved:
        if target.get("optional"):
            print(f"Skipping optional target {target['name']} (path unresolved).")
            return False
        raise ValueError(f"Unresolved path for target {target['name']}")
    if path is None:
        return False
    existing = _read_dotenv(path)
    content = _build_dotenv_content(
        target["name"],
        env,
        vars_list,
        existing,
        overrides,
        prune=prune,
        mode=mode,
    )
    return _write_text(path, content, dry_run=dry_run)


def _sync_render_target_group(
    path: Path,
    targets: List[Dict[str, Any]],
    env: str,
    vars_list: List[Dict[str, Any]],
    overrides: Dict[str, str],
    prune: bool,
    dry_run: bool,
    mode: str,
) -> bool:
    if not path.exists():
        raise FileNotFoundError(f"Render YAML not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    services = data.get("services") or []
    changed = False

    for target in targets:
        service_name = target.get("service")
        if not service_name:
            raise ValueError(f"Render target {target['name']} missing service name.")
        service = next((s for s in services if s.get("name") == service_name), None)
        if service is None:
            raise ValueError(f"Service '{service_name}' not found in {path}")
        existing_envvars = service.get("envVars") or []
        service["envVars"] = _build_render_envvars(
            target["name"],
            env,
            vars_list,
            existing_envvars,
            overrides,
            prune=prune,
            mode=mode,
        )

    new_content = _dump_yaml(data)
    if _write_text(path, new_content, dry_run=dry_run):
        changed = True
    return changed


def _validate_dotenv_target(
    target: Dict[str, Any],
    env: str,
    vars_list: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    path, unresolved = _expand_path(target["path"])
    if unresolved or path is None:
        if target.get("optional"):
            print(f"Skipping optional target {target['name']} (path unresolved).")
            return [], []
        raise ValueError(f"Unresolved path for target {target['name']}")
    if not path.exists():
        if target.get("optional"):
            print(f"Skipping optional target {target['name']} (missing file).")
            return [], []
        raise FileNotFoundError(f"Missing target file: {path}")

    existing = _read_dotenv(path)
    allowed = set(_allowed_keys(vars_list, target["name"]))
    unknown = sorted([key for key in existing.keys() if key not in allowed])

    missing: List[str] = []
    for var in vars_list:
        if not var.get("required"):
            continue
        target_map = _normalize_targets(var.get("targets"))
        if target["name"] not in target_map:
            continue
        value = existing.get(var["key"])
        if not _value_is_present(value):
            missing.append(var["key"])
    return missing, unknown


def _validate_render_target_group(
    path: Path,
    targets: List[Dict[str, Any]],
    vars_list: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Render YAML not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    services = data.get("services") or []
    missing_all: List[str] = []
    unknown_all: List[str] = []

    for target in targets:
        service_name = target.get("service")
        service = next((s for s in services if s.get("name") == service_name), None)
        if service is None:
            raise ValueError(f"Service '{service_name}' not found in {path}")
        envvars = service.get("envVars") or []
        env_map = _render_envvars_to_map(envvars)
        allowed = set(_allowed_keys(vars_list, target["name"]))
        unknown = sorted([key for key in env_map.keys() if key not in allowed])
        if unknown:
            unknown_all.extend([f"{target['name']}:{key}" for key in unknown])

        for var in vars_list:
            if not var.get("required"):
                continue
            target_map = _normalize_targets(var.get("targets"))
            if target["name"] not in target_map:
                continue
            entry = env_map.get(var["key"])
            if not _render_entry_present(entry):
                missing_all.append(f"{target['name']}:{var['key']}")

    return missing_all, unknown_all


def _select_targets(
    schema_targets: List[Dict[str, Any]],
    requested: List[str],
) -> List[Dict[str, Any]]:
    if not requested:
        return schema_targets
    lookup = {t["name"]: t for t in schema_targets}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise ValueError(f"Unknown targets: {', '.join(missing)}")
    return [lookup[name] for name in requested]


def _group_render_targets(targets: List[Dict[str, Any]]) -> Dict[Path, List[Dict[str, Any]]]:
    groups: Dict[Path, List[Dict[str, Any]]] = {}
    for target in targets:
        path, unresolved = _expand_path(target["path"])
        if unresolved:
            if target.get("optional"):
                print(f"Skipping optional target {target['name']} (path unresolved).")
                continue
            raise ValueError(f"Unresolved path for target {target['name']}")
        if path is None:
            continue
        groups.setdefault(path, []).append(target)
    return groups


def _sync(
    schema: Dict[str, Any],
    env: str,
    targets: List[Dict[str, Any]],
    overrides: Dict[str, str],
    prune: bool,
    dry_run: bool,
) -> int:
    vars_list = schema.get("vars", [])
    changed = False

    render_targets = [t for t in targets if t.get("type") == "render_yaml"]
    dotenv_targets = [t for t in targets if t.get("type") == "dotenv"]

    for target in dotenv_targets:
        changed |= _sync_dotenv_target(
            target,
            env,
            vars_list,
            overrides,
            prune=prune,
            dry_run=dry_run,
            mode="sync",
        )

    for path, group in _group_render_targets(render_targets).items():
        changed |= _sync_render_target_group(
            path,
            group,
            env,
            vars_list,
            overrides,
            prune=prune,
            dry_run=dry_run,
            mode="sync",
        )

    if not changed:
        print("No changes required.")
    return 0


def _prune(
    schema: Dict[str, Any],
    env: str,
    targets: List[Dict[str, Any]],
    dry_run: bool,
) -> int:
    vars_list = schema.get("vars", [])
    changed = False

    render_targets = [t for t in targets if t.get("type") == "render_yaml"]
    dotenv_targets = [t for t in targets if t.get("type") == "dotenv"]

    for target in dotenv_targets:
        changed |= _sync_dotenv_target(
            target,
            env,
            vars_list,
            overrides={},
            prune=True,
            dry_run=dry_run,
            mode="prune",
        )

    for path, group in _group_render_targets(render_targets).items():
        changed |= _sync_render_target_group(
            path,
            group,
            env,
            vars_list,
            overrides={},
            prune=True,
            dry_run=dry_run,
            mode="prune",
        )

    if not changed:
        print("No changes required.")
    return 0


def _validate(schema: Dict[str, Any], env: str, targets: List[Dict[str, Any]]) -> int:
    vars_list = schema.get("vars", [])
    missing_all: List[str] = []
    unknown_all: List[str] = []

    render_targets = [t for t in targets if t.get("type") == "render_yaml"]
    dotenv_targets = [t for t in targets if t.get("type") == "dotenv"]

    for target in dotenv_targets:
        missing, unknown = _validate_dotenv_target(target, env, vars_list)
        if missing:
            missing_all.extend([f"{target['name']}:{key}" for key in missing])
        if unknown:
            unknown_all.extend([f"{target['name']}:{key}" for key in unknown])

    for path, group in _group_render_targets(render_targets).items():
        missing, unknown = _validate_render_target_group(path, group, vars_list)
        missing_all.extend(missing)
        unknown_all.extend(unknown)

    if unknown_all:
        print("Unknown keys:")
        for key in sorted(set(unknown_all)):
            print(f"  - {key}")

    if missing_all:
        print("Missing required keys:")
        for key in sorted(set(missing_all)):
            print(f"  - {key}")
        return 1

    print("Validation passed.")
    return 0


def sync_from_schema(
    schema_path: str | None,
    env: str,
    targets: list[str] | None = None,
    overrides: dict[str, str] | None = None,
    prune: bool = False,
    dry_run: bool = False,
) -> int:
    schema = _load_schema(Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH)
    selected_targets = _select_targets(schema.get("targets", []), targets or [])
    return _sync(
        schema,
        env,
        selected_targets,
        overrides or {},
        prune=prune,
        dry_run=dry_run,
    )


def main() -> int:
    return main_with_argv(sys.argv)


def main_with_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Manage Bunoraa env variables from schema.")
    parser.add_argument(
        "--schema",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Path to env schema YAML (default: config/env.schema.yml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Sync env targets from schema.")
    sync_parser.add_argument("--env", required=True, choices=["development", "production"])
    sync_parser.add_argument("--targets", action="append", help="Comma-separated target names.")
    sync_parser.add_argument("--set", action="append", help="Override KEY=VALUE.")
    sync_parser.add_argument("--prune", action="store_true", help="Remove unknown keys.")
    sync_parser.add_argument("--dry-run", action="store_true", help="Do not write changes.")

    validate_parser = subparsers.add_parser("validate", help="Validate env targets.")
    validate_parser.add_argument("--env", required=True, choices=["development", "production"])
    validate_parser.add_argument("--targets", action="append", help="Comma-separated target names.")

    prune_parser = subparsers.add_parser("prune", help="Remove unknown keys from targets.")
    prune_parser.add_argument("--env", required=True, choices=["development", "production"])
    prune_parser.add_argument("--targets", action="append", help="Comma-separated target names.")
    prune_parser.add_argument("--dry-run", action="store_true", help="Do not write changes.")

    args = parser.parse_args(argv[1:])
    schema_path = Path(args.schema)
    schema = _load_schema(schema_path)

    requested_targets = _parse_targets_arg(getattr(args, "targets", None))
    selected_targets = _select_targets(schema.get("targets", []), requested_targets)

    if args.command == "sync":
        overrides = _parse_set_args(getattr(args, "set", None))
        return _sync(
            schema,
            args.env,
            selected_targets,
            overrides,
            prune=bool(args.prune),
            dry_run=bool(args.dry_run),
        )

    if args.command == "prune":
        return _prune(
            schema,
            args.env,
            selected_targets,
            dry_run=bool(args.dry_run),
        )

    if args.command == "validate":
        return _validate(schema, args.env, selected_targets)

    return 1


if __name__ == "__main__":
    sys.exit(main())


# Runtime overrides and env registry sync helpers

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


# Database-backed env schema synchronization

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
def sync_registry_from_schema(
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

