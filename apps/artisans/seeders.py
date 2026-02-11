from __future__ import annotations

from core.seed.base import SeedContext, SeedResult, SeedSpec
from core.seed.registry import register_seed
from apps.artisans.models import Artisan


class ArtisanDemoSeedSpec(SeedSpec):
    name = "artisans.demo"
    app_label = "artisans"
    kind = "demo"
    description = "Seed demo artisans"

    def apply(self, ctx: SeedContext) -> SeedResult:
        result = SeedResult()
        payloads = [
            {
                "name": "Bunoraa",
                "bio": "Specializes in traditional Nakshi Kantha embroidery.",
                "website": "https://bunoraa.com",
                "instagram": "https://instagram.com/bunoraa_bd",
                "is_active": True,
            }
        ]

        for data in payloads:
            name = data["name"]
            if ctx.dry_run:
                if not Artisan.objects.filter(name=name).exists():
                    result.created += 1
                continue
            obj, created = Artisan.objects.get_or_create(name=name, defaults=data)
            if created:
                result.created += 1
            else:
                changed = False
                for field, value in data.items():
                    if getattr(obj, field) != value:
                        setattr(obj, field, value)
                        changed = True
                if changed:
                    obj.save()
                    result.updated += 1

        return result


register_seed(ArtisanDemoSeedSpec())
