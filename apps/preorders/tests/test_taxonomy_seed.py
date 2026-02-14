from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from django.test import TestCase

from core.seed.runner import SeedRunner
from apps.preorders.models import (
    PreOrderCategory,
    PreOrderOption,
    PreOrderOptionChoice,
    PreOrderTemplate,
)


PREORDER_SEED_SPECS = [
    "preorders.categories",
    "preorders.options",
    "preorders.option_choices",
    "preorders.templates",
]


class PreorderTaxonomySeedTests(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.base_dir = Path(self.tmpdir.name)
        self._old_taxonomy_env = os.environ.get("SEED_PREORDER_TAXONOMY_PATH")

    def tearDown(self):
        if self._old_taxonomy_env is None:
            os.environ.pop("SEED_PREORDER_TAXONOMY_PATH", None)
        else:
            os.environ["SEED_PREORDER_TAXONOMY_PATH"] = self._old_taxonomy_env

    def _write_taxonomy(self, payload: dict) -> Path:
        path = self.base_dir / "taxonomy.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _run_seed(self, taxonomy_path: Path, *, prune: bool = True):
        os.environ["SEED_PREORDER_TAXONOMY_PATH"] = str(taxonomy_path)
        runner = SeedRunner(
            base_dir=self.base_dir,
            dry_run=False,
            prune=prune,
            confirm_prune=True,
            logger=lambda *_args, **_kwargs: None,
        )
        return runner.run(only=PREORDER_SEED_SPECS, kind="prod")

    def test_seed_handles_duplicate_option_names_across_categories(self):
        payload = {
            "categories": [
                {"name": "Embroidery", "slug": "embroidery"},
                {"name": "Printing", "slug": "printing"},
            ],
            "options": [
                {"category": "embroidery", "name": "Material", "option_type": "select"},
                {"category": "printing", "name": "Material", "option_type": "select"},
            ],
            "option_choices": [
                {
                    "category": "embroidery",
                    "option": "Material",
                    "value": "cotton",
                    "display_name": "Cotton",
                },
                {
                    "category": "printing",
                    "option": "Material",
                    "value": "cotton",
                    "display_name": "Cotton",
                },
            ],
            "templates": [],
        }
        taxonomy_path = self._write_taxonomy(payload)

        first = self._run_seed(taxonomy_path)
        self.assertEqual(first.errors, 0)
        self.assertEqual(PreOrderCategory.objects.count(), 2)
        self.assertEqual(PreOrderOption.objects.count(), 2)
        self.assertEqual(PreOrderOptionChoice.objects.count(), 2)
        self.assertTrue(
            PreOrderOptionChoice.objects.filter(
                option__category__slug="embroidery",
                option__name="Material",
                value="cotton",
            ).exists()
        )
        self.assertTrue(
            PreOrderOptionChoice.objects.filter(
                option__category__slug="printing",
                option__name="Material",
                value="cotton",
            ).exists()
        )

        second = self._run_seed(taxonomy_path)
        self.assertEqual(second.errors, 0)
        self.assertEqual(second.created, 0)
        self.assertEqual(second.updated, 0)

    def test_prune_marks_missing_option_related_records_inactive(self):
        full_payload = {
            "categories": [{"name": "Embroidery", "slug": "embroidery"}],
            "options": [
                {"category": "embroidery", "name": "Material", "option_type": "select"}
            ],
            "option_choices": [
                {
                    "category": "embroidery",
                    "option": "Material",
                    "value": "cotton",
                    "display_name": "Cotton",
                }
            ],
            "templates": [
                {
                    "name": "Embroidery Basic",
                    "slug": "embroidery-basic",
                    "description": "Starter embroidery package.",
                    "category": "embroidery",
                    "default_options": {"Material": "cotton"},
                }
            ],
        }
        taxonomy_full = self._write_taxonomy(full_payload)
        first = self._run_seed(taxonomy_full)
        self.assertEqual(first.errors, 0)
        self.assertEqual(PreOrderTemplate.objects.filter(slug="embroidery-basic").count(), 1)

        reduced_payload = {
            "categories": [{"name": "Embroidery", "slug": "embroidery"}],
            "options": [],
            "option_choices": [],
            "templates": [],
        }
        taxonomy_reduced = self._write_taxonomy(reduced_payload)
        second = self._run_seed(taxonomy_reduced, prune=True)
        self.assertEqual(second.errors, 0)
        self.assertGreaterEqual(second.pruned, 3)

        option = PreOrderOption.objects.get(category__slug="embroidery", name="Material")
        self.assertFalse(option.is_active)

        choice = PreOrderOptionChoice.objects.get(option=option, value="cotton")
        self.assertFalse(choice.is_active)

        template = PreOrderTemplate.objects.get(slug="embroidery-basic")
        self.assertFalse(template.is_active)
