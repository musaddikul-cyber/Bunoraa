from __future__ import annotations

import random
from decimal import Decimal

from django.utils import timezone
from django.contrib.auth import get_user_model

from core.seed.base import SeedContext, SeedResult, SeedSpec
from core.seed.registry import register_seed
from apps.referral.models import ReferralCode, ReferralReward

User = get_user_model()


class ReferralDemoSeedSpec(SeedSpec):
    name = "referral.demo"
    app_label = "referral"
    kind = "demo"
    description = "Seed demo referral data"

    def apply(self, ctx: SeedContext) -> SeedResult:
        result = SeedResult()
        users = list(User.objects.filter(is_superuser=False))
        if len(users) < 2:
            result.skipped += 1
            return result

        sample_users = random.sample(users, min(5, len(users)))
        for user in sample_users:
            if ctx.dry_run:
                if not ReferralCode.objects.filter(user=user).exists():
                    result.created += 1
                continue

            code, created = ReferralCode.objects.get_or_create(user=user)
            if created:
                result.created += 1

            if random.random() > 0.5:
                referee = random.choice([u for u in users if u != user])
                reward_type = random.choice(["discount", "store_credit"])
                value = Decimal(random.randint(5, 20))
                if ctx.dry_run:
                    result.created += 1
                    continue
                reward, created_reward = ReferralReward.objects.get_or_create(
                    referral_code=code,
                    referrer_user=user,
                    referee_user=referee,
                    reward_type=reward_type,
                    defaults={
                        "value": value,
                        "description": f"{value}% off or {value} store credit",
                        "status": random.choice(["pending", "earned", "applied"]),
                        "earned_at": timezone.now() if random.random() > 0.5 else None,
                        "applied_at": timezone.now() if random.random() > 0.5 else None,
                    },
                )
                if created_reward:
                    result.created += 1

        return result


register_seed(ReferralDemoSeedSpec())
