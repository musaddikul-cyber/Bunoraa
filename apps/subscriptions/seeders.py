from __future__ import annotations

from core.seed.base import JSONSeedSpec
from core.seed.registry import register_seed
from apps.subscriptions.models import Plan


register_seed(
    JSONSeedSpec(
        name="subscriptions.plans",
        app_label="subscriptions",
        model=Plan,
        data_path="apps/subscriptions/data/plans.json",
        key_fields=["name", "interval"],
        update_fields=[
            "stripe_price_id",
            "name",
            "description",
            "interval",
            "price_amount",
            "currency",
            "active",
            "trial_period_days",
            "metadata",
        ],
    )
)
