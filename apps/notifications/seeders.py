from __future__ import annotations

import random
import uuid

from django.contrib.auth import get_user_model

from core.seed.base import JSONSeedSpec, SeedContext, SeedResult, SeedSpec
from core.seed.registry import register_seed
from apps.notifications.models import EmailTemplate, BackInStockNotification
from apps.catalog.models import Product

User = get_user_model()


register_seed(
    JSONSeedSpec(
        name="notifications.email_templates",
        app_label="notifications",
        model=EmailTemplate,
        data_path="apps/notifications/data/email_templates.json",
        key_fields=["notification_type"],
        update_fields=[
            "name",
            "notification_type",
            "subject",
            "html_template",
            "text_template",
            "is_active",
        ],
    )
)


class BackInStockDemoSeedSpec(SeedSpec):
    name = "notifications.demo"
    app_label = "notifications"
    kind = "demo"
    description = "Seed demo back-in-stock notifications"

    def apply(self, ctx: SeedContext) -> SeedResult:
        result = SeedResult()
        products = list(Product.objects.all())
        users = list(User.objects.filter(is_superuser=False))

        if not products:
            result.skipped += 1
            return result

        for product in random.sample(products, min(5, len(products))):
            variant = None
            if product.variants.exists() and random.random() > 0.5:
                variant = random.choice(list(product.variants.all()))

            if users and random.random() > 0.5:
                user = random.choice(users)
                email = ""
            else:
                user = None
                email = f"anon_{uuid.uuid4().hex[:8]}@example.com"

            if ctx.dry_run:
                lookup = {"product": product, "variant": variant, "user": user}
                if not user:
                    lookup["email"] = email
                exists = BackInStockNotification.objects.filter(**lookup).exists()
                if not exists:
                    result.created += 1
                continue

            lookup = {"product": product, "variant": variant, "user": user}
            if not user:
                lookup["email"] = email
            obj, created = BackInStockNotification.objects.get_or_create(
                **lookup,
                defaults={"email": email, "is_notified": False},
            )
            if created:
                result.created += 1

        return result


register_seed(BackInStockDemoSeedSpec())
