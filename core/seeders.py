from __future__ import annotations

from django.contrib.sites.models import Site

from core.seed.base import JSONSeedSpec
from core.seed.registry import register_seed


register_seed(
    JSONSeedSpec(
        name="core.sites",
        app_label="core",
        model=Site,
        data_path="core/data/sites.json",
        key_fields=["id"],
        update_fields=["domain", "name"],
        description="Seed django.contrib.sites Site records",
        kind="prod",
    )
)
