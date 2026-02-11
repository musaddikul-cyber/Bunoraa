from __future__ import annotations

import json
import os
from typing import Any

from django.contrib.auth import get_user_model

from core.seed.base import SeedContext, SeedResult, SeedSpec
from core.seed.registry import register_seed
from apps.email_service.models import SenderDomain, SenderIdentity, EmailTemplate, UnsubscribeGroup

User = get_user_model()


def _load_json(ctx: SeedContext, path: str) -> dict[str, Any]:
    p = ctx.resolve_path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def _resolve_user(user_ref: str | None) -> Any | None:
    if not user_ref:
        return None
    try:
        return User.objects.get(email=user_ref)
    except User.DoesNotExist:
        try:
            return User.objects.get(id=user_ref)
        except Exception:
            return None


def _default_user_fallback() -> Any | None:
    return User.objects.filter(is_superuser=True).first() or User.objects.first()


class EmailServiceSeedBase(SeedSpec):
    data_path: str = ""

    def _load_records(self, ctx: SeedContext) -> tuple[list[dict[str, Any]], str | None]:
        data = _load_json(ctx, self.data_path)
        records = []
        if isinstance(data, dict):
            if "items" in data:
                records = list(data["items"])
            elif "data" in data:
                records = list(data["data"])
        elif isinstance(data, list):
            records = list(data)
        default_user = None
        if isinstance(data, dict):
            default_user = data.get("default_user")
        if not default_user:
            default_user = os.environ.get("SEED_DEFAULT_USER_EMAIL") or os.environ.get("SEED_USER_EMAIL")
        return records, default_user

    def _resolve_record_user(self, record: dict[str, Any], default_user: str | None) -> Any | None:
        user_ref = record.pop("user", None) or record.pop("user_email", None) or record.pop("user_id", None)
        user = _resolve_user(user_ref) or _resolve_user(default_user)
        return user or _default_user_fallback()


class SenderDomainSeedSpec(EmailServiceSeedBase):
    name = "email_service.sender_domains"
    app_label = "email_service"
    kind = "prod"
    data_path = "apps/email_service/data/sender_domains.json"

    def apply(self, ctx: SeedContext) -> SeedResult:
        records, default_user = self._load_records(ctx)
        if not records:
            return SeedResult()

        result = SeedResult()
        desired = set()
        for record in records:
            payload = dict(record)
            domain = payload.pop("domain", None)
            if not domain:
                result.skipped += 1
                continue
            user = self._resolve_record_user(payload, default_user)
            if not user:
                ctx.log(f"[seed:{self.name}] skipping domain {domain}: no user found")
                result.skipped += 1
                continue
            payload["user"] = user

            if ctx.dry_run:
                obj = SenderDomain.objects.filter(domain=domain).first()
                if not obj:
                    result.created += 1
                    continue
                changed = False
                for field, value in payload.items():
                    if getattr(obj, field) != value:
                        changed = True
                        break
                if changed:
                    result.updated += 1
                continue

            obj, created = SenderDomain.objects.get_or_create(domain=domain, defaults=payload)
            if created:
                result.created += 1
            else:
                changed = False
                for field, value in payload.items():
                    if getattr(obj, field) != value:
                        setattr(obj, field, value)
                        changed = True
                if changed:
                    obj.save()
                    result.updated += 1
            desired.add(domain)

        if ctx.prune and not ctx.dry_run and desired:
            for obj in SenderDomain.objects.exclude(domain__in=desired):
                if hasattr(obj, "is_active"):
                    obj.is_active = False
                    obj.save(update_fields=["is_active"])
                    result.pruned += 1
                else:
                    obj.delete()
                    result.pruned += 1
        return result


class SenderIdentitySeedSpec(EmailServiceSeedBase):
    name = "email_service.sender_identities"
    app_label = "email_service"
    kind = "prod"
    data_path = "apps/email_service/data/sender_identities.json"
    dependencies = ["email_service.sender_domains"]

    def apply(self, ctx: SeedContext) -> SeedResult:
        records, default_user = self._load_records(ctx)
        if not records:
            return SeedResult()

        result = SeedResult()
        desired = set()
        for record in records:
            payload = dict(record)
            email = payload.pop("email", None)
            domain_ref = payload.pop("domain", None)
            if not email:
                result.skipped += 1
                continue
            user = self._resolve_record_user(payload, default_user)
            if not user:
                ctx.log(f"[seed:{self.name}] skipping identity {email}: no user found")
                result.skipped += 1
                continue
            payload["user"] = user
            if domain_ref:
                payload["domain"] = SenderDomain.objects.filter(domain=domain_ref).first()

            if ctx.dry_run:
                obj = SenderIdentity.objects.filter(email=email).first()
                if not obj:
                    result.created += 1
                    continue
                changed = False
                for field, value in payload.items():
                    if getattr(obj, field) != value:
                        changed = True
                        break
                if changed:
                    result.updated += 1
                continue

            obj, created = SenderIdentity.objects.get_or_create(email=email, defaults=payload)
            if created:
                result.created += 1
            else:
                changed = False
                for field, value in payload.items():
                    if getattr(obj, field) != value:
                        setattr(obj, field, value)
                        changed = True
                if changed:
                    obj.save()
                    result.updated += 1
            desired.add(email)

        if ctx.prune and not ctx.dry_run and desired:
            for obj in SenderIdentity.objects.exclude(email__in=desired):
                if hasattr(obj, "is_active"):
                    obj.is_active = False
                    obj.save(update_fields=["is_active"])
                    result.pruned += 1
                else:
                    obj.delete()
                    result.pruned += 1
        return result


class EmailTemplateSeedSpec(EmailServiceSeedBase):
    name = "email_service.email_templates"
    app_label = "email_service"
    kind = "prod"
    data_path = "apps/email_service/data/email_templates.json"

    def apply(self, ctx: SeedContext) -> SeedResult:
        records, default_user = self._load_records(ctx)
        if not records:
            return SeedResult()

        result = SeedResult()
        desired = set()
        for record in records:
            payload = dict(record)
            template_id = payload.pop("template_id", None)
            if not template_id:
                result.skipped += 1
                continue
            user = self._resolve_record_user(payload, default_user)
            if not user:
                ctx.log(f"[seed:{self.name}] skipping template {template_id}: no user found")
                result.skipped += 1
                continue
            payload["user"] = user

            if ctx.dry_run:
                obj = EmailTemplate.objects.filter(template_id=template_id).first()
                if not obj:
                    result.created += 1
                    continue
                changed = False
                for field, value in payload.items():
                    if getattr(obj, field) != value:
                        changed = True
                        break
                if changed:
                    result.updated += 1
                continue

            obj, created = EmailTemplate.objects.get_or_create(template_id=template_id, defaults=payload)
            if created:
                result.created += 1
            else:
                changed = False
                for field, value in payload.items():
                    if getattr(obj, field) != value:
                        setattr(obj, field, value)
                        changed = True
                if changed:
                    obj.save()
                    result.updated += 1
            desired.add(template_id)

        if ctx.prune and not ctx.dry_run and desired:
            for obj in EmailTemplate.objects.exclude(template_id__in=desired):
                if hasattr(obj, "is_active"):
                    obj.is_active = False
                    obj.save(update_fields=["is_active"])
                    result.pruned += 1
                else:
                    obj.delete()
                    result.pruned += 1
        return result


class UnsubscribeGroupSeedSpec(EmailServiceSeedBase):
    name = "email_service.unsubscribe_groups"
    app_label = "email_service"
    kind = "prod"
    data_path = "apps/email_service/data/unsubscribe_groups.json"

    def apply(self, ctx: SeedContext) -> SeedResult:
        records, default_user = self._load_records(ctx)
        if not records:
            return SeedResult()

        result = SeedResult()
        desired = set()
        for record in records:
            payload = dict(record)
            name = payload.pop("name", None)
            if not name:
                result.skipped += 1
                continue
            user = self._resolve_record_user(payload, default_user)
            if not user:
                ctx.log(f"[seed:{self.name}] skipping unsubscribe group {name}: no user found")
                result.skipped += 1
                continue
            payload["user"] = user

            if ctx.dry_run:
                obj = UnsubscribeGroup.objects.filter(name=name, user=user).first()
                if not obj:
                    result.created += 1
                    continue
                changed = False
                for field, value in payload.items():
                    if getattr(obj, field) != value:
                        changed = True
                        break
                if changed:
                    result.updated += 1
                continue

            obj, created = UnsubscribeGroup.objects.get_or_create(name=name, user=user, defaults=payload)
            if created:
                result.created += 1
            else:
                changed = False
                for field, value in payload.items():
                    if getattr(obj, field) != value:
                        setattr(obj, field, value)
                        changed = True
                if changed:
                    obj.save()
                    result.updated += 1
            desired.add((user.id, name))

        if ctx.prune and not ctx.dry_run and desired:
            for obj in UnsubscribeGroup.objects.all():
                key = (obj.user_id, obj.name)
                if key not in desired:
                    if hasattr(obj, "is_active"):
                        obj.is_active = False
                        obj.save(update_fields=["is_active"])
                        result.pruned += 1
                    else:
                        obj.delete()
                        result.pruned += 1
        return result


register_seed(SenderDomainSeedSpec())
register_seed(SenderIdentitySeedSpec())
register_seed(EmailTemplateSeedSpec())
register_seed(UnsubscribeGroupSeedSpec())
