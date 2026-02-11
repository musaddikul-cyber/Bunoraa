import json
import tempfile
from pathlib import Path

from django.test import TestCase

from core.seed.base import SeedContext, JSONSeedSpec
from apps.catalog.models import Tag


class JSONSeedSpecTests(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.base_dir = Path(self.tmpdir.name)

    def _write_tags(self, names):
        path = self.base_dir / "tags.json"
        data = {"items": [{"name": name} for name in names]}
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _spec(self, path: Path) -> JSONSeedSpec:
        return JSONSeedSpec(
            name="test.tags",
            app_label="catalog",
            model=Tag,
            data_path=str(path),
            key_fields=["name"],
            update_fields=["name"],
        )

    def test_idempotent_seed(self):
        path = self._write_tags(["Alpha", "Beta"])
        spec = self._spec(path)
        ctx = SeedContext(base_dir=self.base_dir, dry_run=False, prune=True)

        first = spec.apply(ctx)
        self.assertEqual(first.created, 2)
        self.assertEqual(Tag.objects.count(), 2)

        second = spec.apply(ctx)
        self.assertEqual(second.created, 0)
        self.assertEqual(second.updated, 0)
        self.assertEqual(Tag.objects.count(), 2)

    def test_prune_missing_records(self):
        path = self._write_tags(["Alpha", "Beta"])
        spec = self._spec(path)
        ctx = SeedContext(base_dir=self.base_dir, dry_run=False, prune=True)

        spec.apply(ctx)
        self.assertEqual(Tag.objects.count(), 2)

        path = self._write_tags(["Alpha"])
        spec = self._spec(path)
        spec.apply(ctx)
        self.assertEqual(Tag.objects.count(), 1)
        self.assertTrue(Tag.objects.filter(name="Alpha").exists())

    def test_dry_run_no_writes(self):
        path = self._write_tags(["Alpha"])
        spec = self._spec(path)
        ctx = SeedContext(base_dir=self.base_dir, dry_run=True, prune=True)

        result = spec.apply(ctx)
        self.assertEqual(result.created, 1)
        self.assertEqual(Tag.objects.count(), 0)
