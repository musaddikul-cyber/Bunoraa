from __future__ import annotations

import json
import tempfile
from pathlib import Path

from django.core.management import call_command
from django.test import TestCase

from apps.catalog.models import Category, CategoryFacet, Facet
from apps.preorders.models import (
    PreOrderCategory,
    PreOrderOption,
    PreOrderOptionChoice,
    PreOrderTemplate,
)


class SyncTaxonomySeedsCommandTests(TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_preorders_sync_and_save_runs_in_one_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            taxonomy_path = Path(tmpdir) / "preorders_taxonomy.json"
            self._write_json(
                taxonomy_path,
                {
                    "version": "2026-02-15",
                    "categories": [
                        {
                            "name": "Custom Embroidery",
                            "slug": "custom-embroidery",
                            "description": "Custom embroidery orders.",
                        }
                    ],
                    "options": [
                        {
                            "category": "custom-embroidery",
                            "name": "Material",
                            "option_type": "select",
                            "is_required": True,
                        }
                    ],
                    "option_choices": [
                        {
                            "category": "custom-embroidery",
                            "option": "Material",
                            "value": "cotton",
                            "display_name": "Cotton",
                        }
                    ],
                    "templates": [
                        {
                            "name": "Embroidery Basic",
                            "slug": "embroidery-basic",
                            "description": "Starter package.",
                            "category": "custom-embroidery",
                            "default_options": {"Material": "cotton"},
                        }
                    ],
                },
            )

            call_command(
                "sync_taxonomy_seeds",
                domains="preorders",
                preorders_file=str(taxonomy_path),
                force=True,
            )

            self.assertEqual(PreOrderCategory.objects.count(), 1)
            self.assertEqual(PreOrderOption.objects.count(), 1)
            self.assertEqual(PreOrderOptionChoice.objects.count(), 1)
            self.assertEqual(PreOrderTemplate.objects.count(), 1)

            saved = self._read_json(taxonomy_path)
            self.assertIn("categories", saved)
            self.assertIn("options", saved)
            self.assertIn("option_choices", saved)
            self.assertIn("templates", saved)
            self.assertEqual(saved["categories"][0]["slug"], "custom-embroidery")
            self.assertEqual(saved["options"][0]["category"], "custom-embroidery")
            self.assertEqual(saved["option_choices"][0]["option"], "Material")
            self.assertEqual(saved["templates"][0]["slug"], "embroidery-basic")

    def test_catalog_save_preserves_existing_taxonomy_codes(self):
        root = Category.objects.create(name="Home & Living", slug="home-living")
        child = Category.objects.create(name="Decor", slug="decor", parent=root)
        facet = Facet.objects.create(name="Material", slug="material", type="choice", values=["wood"])
        CategoryFacet.objects.create(category=root, facet=facet)

        with tempfile.TemporaryDirectory() as tmpdir:
            taxonomy_path = Path(tmpdir) / "catalog_taxonomy.json"
            self._write_json(
                taxonomy_path,
                {
                    "version": 1,
                    "categories": [
                        {
                            "code": "CAT_HOME",
                            "name": "Old Home Name",
                            "slug": "home-living",
                            "children": [
                                {
                                    "code": "CAT_HOME_DECOR",
                                    "name": "Old Decor Name",
                                    "slug": "decor",
                                }
                            ],
                        }
                    ],
                },
            )

            call_command(
                "sync_taxonomy_seeds",
                domains="catalog",
                catalog_file=str(taxonomy_path),
                skip_sync=True,
            )

            saved = self._read_json(taxonomy_path)
            root_node = saved["categories"][0]
            self.assertEqual(root_node["slug"], "home-living")
            self.assertEqual(root_node["name"], "Home & Living")
            self.assertEqual(root_node["code"], "CAT_HOME")
            self.assertEqual(root_node["facets"], ["material"])

            child_node = root_node["children"][0]
            self.assertEqual(child_node["slug"], "decor")
            self.assertEqual(child_node["name"], "Decor")
            self.assertEqual(child_node["code"], "CAT_HOME_DECOR")
