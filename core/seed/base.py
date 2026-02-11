from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from django.db import models
from django.utils import timezone


@dataclass
class SeedResult:
    created: int = 0
    updated: int = 0
    pruned: int = 0
    skipped: int = 0
    errors: int = 0
    notes: list[str] = field(default_factory=list)

    def merge(self, other: "SeedResult") -> "SeedResult":
        self.created += other.created
        self.updated += other.updated
        self.pruned += other.pruned
        self.skipped += other.skipped
        self.errors += other.errors
        self.notes.extend(other.notes)
        return self


@dataclass
class SeedContext:
    base_dir: Path
    dry_run: bool = False
    prune: bool = True
    confirm_prune: bool = False
    environment: str = "development"
    logger: Callable[[str], None] | None = None

    def log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.base_dir / p
        return p


class SeedSpec:
    name: str = ""
    app_label: str = ""
    description: str = ""
    dependencies: list[str] = []
    kind: str = "prod"  # prod or demo
    prune: bool | None = None

    def apply(self, ctx: SeedContext) -> SeedResult:
        raise NotImplementedError


def _normalize_key_value(value: Any) -> Any:
    if isinstance(value, models.Model):
        return value.pk
    return value


class JSONSeedSpec(SeedSpec):
    model: type[models.Model]
    data_path: str | Path
    key_fields: list[str] | None = None
    update_fields: list[str] | None = None
    fk_fields: dict[str, tuple[type[models.Model], str]] | None = None
    m2m_fields: dict[str, tuple[type[models.Model], str]] | None = None
    env_overrides: dict[str, str] | None = None
    record_key_func: Callable[[dict[str, Any]], Any] | None = None
    obj_key_func: Callable[[models.Model], Any] | None = None

    def __init__(
        self,
        *,
        name: str,
        app_label: str,
        model: type[models.Model],
        data_path: str | Path,
        key_fields: list[str] | None = None,
        update_fields: list[str] | None = None,
        fk_fields: dict[str, tuple[type[models.Model], str]] | None = None,
        m2m_fields: dict[str, tuple[type[models.Model], str]] | None = None,
        env_overrides: dict[str, str] | None = None,
        record_key_func: Callable[[dict[str, Any]], Any] | None = None,
        obj_key_func: Callable[[models.Model], Any] | None = None,
        dependencies: list[str] | None = None,
        description: str = "",
        kind: str = "prod",
        prune: bool | None = None,
    ) -> None:
        self.name = name
        self.app_label = app_label
        self.model = model
        self.data_path = data_path
        self.key_fields = key_fields
        self.update_fields = update_fields
        self.fk_fields = fk_fields or {}
        self.m2m_fields = m2m_fields or {}
        self.env_overrides = env_overrides or {}
        self.record_key_func = record_key_func
        self.obj_key_func = obj_key_func
        self.dependencies = dependencies or []
        self.description = description
        self.kind = kind
        self.prune = prune

    def load_records(self, ctx: SeedContext) -> list[dict[str, Any]]:
        import json

        path = ctx.resolve_path(self.data_path)
        if not path.exists():
            ctx.log(f"[seed:{self.name}] data file not found: {path}")
            return []
        with path.open("r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            if "items" in data:
                return list(data["items"])
            if "data" in data:
                return list(data["data"])
        if isinstance(data, list):
            return list(data)
        return []

    def apply(self, ctx: SeedContext) -> SeedResult:
        records = self.load_records(ctx)
        result = SeedResult()
        if not records:
            return result

        desired_keys: set[Any] = set()

        for record in records:
            record = dict(record)
            self._apply_env_overrides(record)
            key = self._record_key(record)
            desired_keys.add(key)

            lookup = self._build_lookup(record)
            values, m2m_values = self._build_values(record)
            if ctx.dry_run:
                obj = self.model.objects.filter(**lookup).first()
                if obj is None:
                    result.created += 1
                    continue
                updated = self._update_instance(obj, values, dry_run=True)
                m2m_changed = False
                if m2m_values:
                    m2m_changed = self._m2m_would_change(obj, m2m_values)
                if updated or m2m_changed:
                    result.updated += 1
                continue

            obj, created = self.model.objects.get_or_create(**lookup, defaults=values)
            if created:
                result.created += 1
            else:
                updated = self._update_instance(obj, values, dry_run=False)
                if updated:
                    result.updated += 1

            if m2m_values:
                self._apply_m2m(obj, m2m_values, ctx)

        if self._should_prune(ctx):
            pruned = self._prune(ctx, desired_keys)
            result.pruned += pruned

        return result

    def _apply_env_overrides(self, record: dict[str, Any]) -> None:
        if not self.env_overrides:
            return
        import os

        for field, env_key in self.env_overrides.items():
            if env_key in os.environ:
                record[field] = os.environ.get(env_key)

    def _record_key(self, record: dict[str, Any]) -> Any:
        if self.record_key_func:
            return self.record_key_func(record)
        if not self.key_fields:
            raise ValueError(f"SeedSpec {self.name} missing key_fields")
        parts = []
        for field in self.key_fields:
            if field in record:
                parts.append(record[field])
                continue
            if "__" in field:
                base = field.split("__", 1)[0]
                if base in record:
                    parts.append(record[base])
                    continue
            parts.append(None)
        return tuple(parts)

    def _obj_key(self, obj: models.Model) -> Any:
        if self.obj_key_func:
            return self.obj_key_func(obj)
        if not self.key_fields:
            raise ValueError(f"SeedSpec {self.name} missing key_fields")
        values = []
        for field in self.key_fields:
            if "__" in field:
                values.append(self._resolve_attr(obj, field))
            else:
                values.append(getattr(obj, field))
        return tuple(values)

    def _build_lookup(self, record: dict[str, Any]) -> dict[str, Any]:
        if not self.key_fields:
            raise ValueError(f"SeedSpec {self.name} missing key_fields")
        lookup: dict[str, Any] = {}
        for field in self.key_fields:
            if field in record:
                lookup[field] = record[field]
                continue
            if "__" in field:
                base = field.split("__", 1)[0]
                if base in record:
                    lookup[field] = record[base]
        return lookup

    def _build_values(self, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        values = {}
        m2m_values = {}
        for key, value in record.items():
            if key in self.key_fields:
                continue
            if key in self.m2m_fields:
                m2m_values[key] = value
                continue
            values[key] = value

        for field, (model, lookup_field) in self.fk_fields.items():
            if field not in values:
                continue
            ref = values[field]
            if ref is None or ref == "":
                values[field] = None
                continue
            if isinstance(ref, model):
                values[field] = ref
                continue
            values[field] = model.objects.get(**{lookup_field: ref})

        return values, m2m_values

    def _apply_m2m(self, obj: models.Model, m2m_values: dict[str, Any], ctx: SeedContext) -> None:
        for field, (model, lookup_field) in self.m2m_fields.items():
            items = m2m_values.get(field, [])
            if items is None:
                continue
            qs = model.objects.filter(**{f"{lookup_field}__in": items})
            getattr(obj, field).set(qs)

    def _m2m_would_change(self, obj: models.Model, m2m_values: dict[str, Any]) -> bool:
        for field, (model, lookup_field) in self.m2m_fields.items():
            items = m2m_values.get(field, [])
            if items is None:
                continue
            qs = model.objects.filter(**{f"{lookup_field}__in": items})
            desired_ids = set(qs.values_list("pk", flat=True))
            current_ids = set(getattr(obj, field).values_list("pk", flat=True))
            if desired_ids != current_ids:
                return True
        return False

    def _update_instance(self, obj: models.Model, values: dict[str, Any], *, dry_run: bool) -> bool:
        if not values:
            return False
        changed = False
        fields = self.update_fields or list(values.keys())
        for field in fields:
            if field not in values:
                continue
            current = getattr(obj, field)
            new = values[field]
            if _normalize_key_value(current) != _normalize_key_value(new):
                setattr(obj, field, new)
                changed = True
        if changed and not dry_run:
            obj.save(update_fields=fields)
        return changed

    def _resolve_attr(self, obj: models.Model, path: str) -> Any:
        value = obj
        for part in path.split("__"):
            value = getattr(value, part, None)
            if value is None:
                break
        return value

    def _should_prune(self, ctx: SeedContext) -> bool:
        if self.prune is False:
            return False
        if self.prune is True:
            return ctx.prune
        return ctx.prune

    def _prune(self, ctx: SeedContext, desired_keys: set[Any]) -> int:
        qs = self.model.objects.all()
        to_prune = []
        for obj in qs:
            key = self._obj_key(obj)
            if key not in desired_keys:
                to_prune.append(obj)

        if not to_prune:
            return 0

        if ctx.dry_run:
            ctx.log(f"[seed:{self.name}] prune {len(to_prune)} items (dry-run)")
            return 0

        count = 0
        now = timezone.now()
        for obj in to_prune:
            if hasattr(obj, "is_deleted"):
                obj.is_deleted = True
                if hasattr(obj, "deleted_at"):
                    obj.deleted_at = now
                obj.save(update_fields=[f for f in ["is_deleted", "deleted_at"] if hasattr(obj, f)])
            elif hasattr(obj, "is_active"):
                obj.is_active = False
                obj.save(update_fields=["is_active"])
            else:
                obj.delete()
            count += 1
        return count
