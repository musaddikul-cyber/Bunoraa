from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any, Iterable

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.core.serializers.json import DjangoJSONEncoder

from core.seed.runner import SeedRunner


DEFAULT_CATALOG_TAXONOMY_PATH = "apps/catalog/data/taxonomy.json"
DEFAULT_PAGES_TAXONOMY_PATH = "apps/pages/data/taxonomy.json"
DEFAULT_PREORDER_TAXONOMY_PATH = "apps/preorders/data/taxonomy.json"

DOMAIN_ORDER = ("catalog", "pages", "preorders")

DOMAIN_SEED_SPECS: dict[str, list[str]] = {
    "catalog": [
        "catalog.categories",
        "catalog.facets",
        "catalog.category_facets",
    ],
    "pages": [
        "i18n.currencies",
        "pages.site_settings",
        "pages.newsletter_incentives",
        "pages.blog_categories",
        "pages.blog_tags",
        "pages.faqs",
        "pages.social_links",
    ],
    "preorders": [
        "preorders.categories",
        "preorders.options",
        "preorders.option_choices",
        "preorders.templates",
    ],
}


class Command(BaseCommand):
    help = (
        "Sync taxonomy-based seeds into DB and write current DB state back to taxonomy files "
        "(catalog/pages/preorders) in one command."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--domains",
            type=str,
            default="catalog,pages,preorders",
            help="Comma-separated domains to process: catalog,pages,preorders",
        )
        parser.add_argument(
            "--catalog-file",
            type=str,
            help="Override catalog taxonomy path (defaults to apps/catalog/data/taxonomy.json).",
        )
        parser.add_argument(
            "--pages-file",
            type=str,
            help="Override pages taxonomy path (defaults to apps/pages/data/taxonomy.json).",
        )
        parser.add_argument(
            "--preorders-file",
            type=str,
            help="Override preorders taxonomy path (defaults to apps/preorders/data/taxonomy.json).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Run sync in dry-run mode. File writes are skipped.",
        )
        parser.add_argument(
            "--no-prune",
            action="store_true",
            help="Disable pruning during seed sync.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Allow prune in production (equivalent to confirm-prune).",
        )
        parser.add_argument(
            "--skip-sync",
            action="store_true",
            help="Skip DB sync and only export current DB state to taxonomy files.",
        )
        parser.add_argument(
            "--no-save",
            action="store_true",
            help="Run DB sync but do not write taxonomy files.",
        )
        parser.add_argument(
            "--backup",
            action="store_true",
            help="Create a timestamped backup before overwriting each taxonomy file.",
        )
        parser.add_argument(
            "--indent",
            type=int,
            default=2,
            help="JSON indent level for saved taxonomy files (default: 2).",
        )

    def handle(self, *args, **options):
        domains = self._parse_domains(options.get("domains"))
        self._apply_path_overrides(options)

        if not options.get("skip_sync"):
            run_specs = self._build_seed_spec_selection(domains)
            runner = SeedRunner(
                dry_run=options.get("dry_run", False),
                prune=not options.get("no_prune", False),
                confirm_prune=options.get("force", False),
                logger=self.stdout.write,
            )
            result = runner.run(only=run_specs, kind="prod")

            self.stdout.write("")
            self.stdout.write("Seed sync summary:")
            self.stdout.write(f"  Created: {result.created}")
            self.stdout.write(f"  Updated: {result.updated}")
            self.stdout.write(f"  Pruned:  {result.pruned}")
            self.stdout.write(f"  Skipped: {result.skipped}")
            self.stdout.write(f"  Errors:  {result.errors}")

            if result.errors:
                raise CommandError(
                    f"Seed sync finished with {result.errors} error(s). "
                    "Fix sync errors before writing taxonomy files."
                )
        else:
            self.stdout.write("Skipped DB sync (--skip-sync). Exporting from current DB state.")

        if options.get("no_save"):
            self.stdout.write(self.style.SUCCESS("Completed without writing taxonomy files (--no-save)."))
            return

        if options.get("dry_run"):
            self.stdout.write(self.style.WARNING("Dry-run enabled; taxonomy files were not written."))
            return

        saved_paths: list[Path] = []
        backups: list[Path] = []
        indent = int(options.get("indent") or 2)
        backup = bool(options.get("backup", False))

        if "catalog" in domains:
            path = self._resolve_catalog_path()
            payload = self._build_catalog_taxonomy_payload(path)
            saved_path, backup_path = self._atomic_write_json(path, payload, indent=indent, backup=backup)
            saved_paths.append(saved_path)
            if backup_path:
                backups.append(backup_path)

        if "pages" in domains:
            path = self._resolve_pages_path()
            payload = self._build_pages_taxonomy_payload(path)
            saved_path, backup_path = self._atomic_write_json(path, payload, indent=indent, backup=backup)
            saved_paths.append(saved_path)
            if backup_path:
                backups.append(backup_path)

        if "preorders" in domains:
            path = self._resolve_preorders_path()
            payload = self._build_preorders_taxonomy_payload(path)
            saved_path, backup_path = self._atomic_write_json(path, payload, indent=indent, backup=backup)
            saved_paths.append(saved_path)
            if backup_path:
                backups.append(backup_path)

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Taxonomy sync-and-save completed."))
        for saved in saved_paths:
            self.stdout.write(f"  Saved:  {saved}")
        for backup_path in backups:
            self.stdout.write(f"  Backup: {backup_path}")

    def _parse_domains(self, raw: str | None) -> list[str]:
        if not raw:
            return list(DOMAIN_ORDER)
        parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
        invalid = [part for part in parts if part not in DOMAIN_SEED_SPECS]
        if invalid:
            raise CommandError(
                f"Invalid --domains value(s): {', '.join(invalid)}. "
                f"Allowed: {', '.join(DOMAIN_ORDER)}."
            )

        seen: set[str] = set()
        ordered: list[str] = []
        for domain in DOMAIN_ORDER:
            if domain in parts and domain not in seen:
                ordered.append(domain)
                seen.add(domain)
        return ordered

    def _apply_path_overrides(self, options: dict[str, Any]) -> None:
        if options.get("catalog_file"):
            os.environ["SEED_TAXONOMY_PATH"] = str(options["catalog_file"])
        if options.get("pages_file"):
            os.environ["SEED_PAGES_TAXONOMY_PATH"] = str(options["pages_file"])
        if options.get("preorders_file"):
            os.environ["SEED_PREORDER_TAXONOMY_PATH"] = str(options["preorders_file"])

    def _build_seed_spec_selection(self, domains: Iterable[str]) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        for domain in domains:
            for name in DOMAIN_SEED_SPECS[domain]:
                if name in seen:
                    continue
                selected.append(name)
                seen.add(name)
        return selected

    def _resolve_catalog_path(self) -> Path:
        env_path = os.environ.get("SEED_TAXONOMY_PATH") or os.environ.get("CATALOG_TAXONOMY_PATH")
        return self._resolve_path(env_path or DEFAULT_CATALOG_TAXONOMY_PATH)

    def _resolve_pages_path(self) -> Path:
        env_path = os.environ.get("SEED_PAGES_TAXONOMY_PATH") or os.environ.get("PAGES_TAXONOMY_PATH")
        return self._resolve_path(env_path or DEFAULT_PAGES_TAXONOMY_PATH)

    def _resolve_preorders_path(self) -> Path:
        env_path = os.environ.get("SEED_PREORDER_TAXONOMY_PATH") or os.environ.get("PREORDER_TAXONOMY_PATH")
        return self._resolve_path(env_path or DEFAULT_PREORDER_TAXONOMY_PATH)

    def _resolve_path(self, path: str | Path) -> Path:
        value = Path(path)
        if value.is_absolute():
            return value
        return Path(settings.BASE_DIR) / value

    def _safe_load_json(self, path: Path) -> Any:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8-sig") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _atomic_write_json(
        self,
        path: Path,
        payload: Any,
        *,
        indent: int,
        backup: bool,
    ) -> tuple[Path, Path | None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        backup_path: Path | None = None

        if backup and path.exists():
            stamp = datetime.now(dt_timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = path.with_name(f"{path.name}.{stamp}.bak")
            backup_path.write_bytes(path.read_bytes())

        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(payload, fh, indent=indent, ensure_ascii=False, cls=DjangoJSONEncoder)
                fh.write("\n")
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return path, backup_path

    def _build_catalog_taxonomy_payload(self, path: Path) -> dict[str, Any]:
        from apps.catalog.models import Category

        existing = self._safe_load_json(path)
        payload = deepcopy(existing) if isinstance(existing, dict) else {}

        existing_nodes_by_path: dict[tuple[str, ...], dict[str, Any]] = {}
        self._index_catalog_nodes((payload.get("categories") or []), (), existing_nodes_by_path)

        categories = list(
            Category.objects.select_related("parent")
            .prefetch_related("category_facets__facet")
            .order_by("path")
        )
        children_by_parent: dict[Any, list[Any]] = {}
        for category in categories:
            children_by_parent.setdefault(category.parent_id, []).append(category)

        def build_node(category: Any, slug_path: tuple[str, ...]) -> dict[str, Any]:
            next_path = slug_path + (category.slug,)
            existing_node = existing_nodes_by_path.get(next_path, {})

            # Keep unknown keys from existing taxonomy (e.g. `code`) while refreshing canonical fields.
            node: dict[str, Any] = {}
            for key, value in existing_node.items():
                if key in {
                    "name",
                    "slug",
                    "children",
                    "facets",
                    "is_visible",
                    "meta_title",
                    "meta_description",
                    "aspect_ratio",
                }:
                    continue
                node[key] = deepcopy(value)

            node["name"] = category.name
            node["slug"] = category.slug

            facets = sorted(
                {
                    cf.facet.slug
                    for cf in category.category_facets.all()
                    if getattr(cf, "facet", None) and cf.facet.slug
                }
            )
            if facets:
                node["facets"] = facets
            if not category.is_visible:
                node["is_visible"] = False
            if category.meta_title:
                node["meta_title"] = category.meta_title
            if category.meta_description:
                node["meta_description"] = category.meta_description
            if category.aspect_ratio and category.aspect_ratio != "1:1":
                node["aspect_ratio"] = category.aspect_ratio

            children = [build_node(child, next_path) for child in children_by_parent.get(category.id, [])]
            if children:
                node["children"] = children
            return node

        roots = children_by_parent.get(None, [])
        payload["categories"] = [build_node(root, ()) for root in roots]
        if "version" not in payload:
            payload["version"] = 1
        return payload

    def _index_catalog_nodes(
        self,
        nodes: Iterable[dict[str, Any]],
        slug_path: tuple[str, ...],
        out: dict[tuple[str, ...], dict[str, Any]],
    ) -> None:
        for node in nodes:
            slug = str(node.get("slug") or "").strip()
            if not slug:
                continue
            next_path = slug_path + (slug,)
            out[next_path] = node
            self._index_catalog_nodes(node.get("children") or [], next_path, out)

    def _build_pages_taxonomy_payload(self, path: Path) -> dict[str, Any]:
        from apps.pages.models import SiteSettings, NewsletterIncentive, BlogCategory, BlogTag, FAQ, SocialLink

        existing = self._safe_load_json(path)
        payload = deepcopy(existing) if isinstance(existing, dict) else {}

        site_settings_fields = [
            "site_name",
            "site_tagline",
            "site_description",
            "contact_email",
            "contact_phone",
            "contact_address",
            "support_reply_time_note",
            "tax_rate",
            "default_meta_title",
            "default_meta_description",
            "footer_text",
            "copyright_text",
        ]

        site_settings = SiteSettings.objects.first()
        if site_settings:
            site_payload = {field: getattr(site_settings, field) for field in site_settings_fields}
            site_payload["currency"] = getattr(site_settings, "currency_id", None) or "BDT"
        else:
            site_payload = {}
        payload["site_settings"] = site_payload

        payload["newsletter_incentives"] = [
            {
                "title": item.title,
                "description": item.description,
                "discount_percentage": item.discount_percentage,
                "discount_code": item.discount_code,
                "min_order_amount": item.min_order_amount,
                "max_uses": item.max_uses,
                "is_active": item.is_active,
                "valid_until": item.valid_until,
            }
            for item in NewsletterIncentive.objects.all().order_by("discount_code", "title")
        ]

        payload["blog_categories"] = [
            {
                "name": item.name,
                "slug": item.slug,
                "description": item.description,
                "icon": item.icon,
            }
            for item in BlogCategory.objects.all().order_by("slug", "name")
        ]

        payload["blog_tags"] = [
            {
                "name": item.name,
                "slug": item.slug,
            }
            for item in BlogTag.objects.all().order_by("slug", "name")
        ]

        payload["faqs"] = [
            {
                "question": item.question,
                "answer": item.answer,
                "category": item.category,
                "sort_order": item.sort_order,
                "is_active": item.is_active,
            }
            for item in FAQ.objects.all().order_by("sort_order", "question")
        ]

        payload["social_links"] = [
            {
                "site": item.site_id or 1,
                "name": item.name,
                "url": item.url,
                "order": item.order,
                "is_active": item.is_active,
            }
            for item in SocialLink.objects.select_related("site").all().order_by("order", "name")
        ]
        return payload

    def _build_preorders_taxonomy_payload(self, path: Path) -> dict[str, Any]:
        from apps.preorders.models import (
            PreOrderCategory,
            PreOrderOption,
            PreOrderOptionChoice,
            PreOrderTemplate,
        )

        existing = self._safe_load_json(path)
        payload = deepcopy(existing) if isinstance(existing, dict) else {}
        payload.setdefault("version", datetime.now(dt_timezone.utc).date().isoformat())

        payload["categories"] = [
            {
                "name": item.name,
                "slug": item.slug,
                "description": item.description,
                "icon": item.icon,
                "base_price": item.base_price,
                "deposit_percentage": item.deposit_percentage,
                "min_production_days": item.min_production_days,
                "max_production_days": item.max_production_days,
                "requires_design": item.requires_design,
                "requires_approval": item.requires_approval,
                "allow_rush_order": item.allow_rush_order,
                "rush_order_fee_percentage": item.rush_order_fee_percentage,
                "min_quantity": item.min_quantity,
                "max_quantity": item.max_quantity,
                "is_active": item.is_active,
                "order": item.order,
            }
            for item in PreOrderCategory.objects.all().order_by("order", "name")
        ]

        payload["options"] = [
            {
                "category": item.category.slug,
                "name": item.name,
                "description": item.description,
                "option_type": item.option_type,
                "is_required": item.is_required,
                "min_length": item.min_length,
                "max_length": item.max_length,
                "price_modifier": item.price_modifier,
                "placeholder": item.placeholder,
                "help_text": item.help_text,
                "order": item.order,
                "is_active": item.is_active,
            }
            for item in PreOrderOption.objects.select_related("category").all().order_by(
                "category__slug",
                "order",
                "name",
            )
        ]

        payload["option_choices"] = [
            {
                "category": item.option.category.slug,
                "option": item.option.name,
                "value": item.value,
                "display_name": item.display_name,
                "price_modifier": item.price_modifier,
                "color_code": item.color_code,
                "order": item.order,
                "is_active": item.is_active,
            }
            for item in PreOrderOptionChoice.objects.select_related("option__category").all().order_by(
                "option__category__slug",
                "option__name",
                "order",
                "value",
            )
        ]

        payload["templates"] = [
            {
                "name": item.name,
                "slug": item.slug,
                "description": item.description,
                "category": item.category.slug,
                "default_quantity": item.default_quantity,
                "base_price": item.base_price,
                "estimated_days": item.estimated_days,
                "default_options": item.default_options,
                "is_active": item.is_active,
                "is_featured": item.is_featured,
                "order": item.order,
            }
            for item in PreOrderTemplate.objects.select_related("category").all().order_by("order", "name")
        ]

        return payload
