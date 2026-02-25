from __future__ import annotations

from django.db import migrations
from django.db.models import Avg, Count, Q


def migrate_legacy_reviews(apps, schema_editor):
    LegacyReview = apps.get_model("reviews", "Review")
    LegacyReviewImage = apps.get_model("reviews", "ReviewImage")
    LegacyReviewVote = apps.get_model("reviews", "ReviewVote")

    CatalogReview = apps.get_model("catalog", "Review")
    CatalogReviewImage = apps.get_model("catalog", "ReviewImage")
    CatalogReviewVote = apps.get_model("catalog", "ReviewVote")
    Product = apps.get_model("catalog", "Product")

    review_id_map: dict[str, str] = {}
    touched_product_ids: set[str] = set()

    # 1) Migrate legacy reviews into catalog reviews when a user/product review
    # does not already exist.
    for legacy in LegacyReview.objects.filter(is_deleted=False).iterator():
        touched_product_ids.add(str(legacy.product_id))
        existing = CatalogReview.objects.filter(
            user_id=legacy.user_id,
            product_id=legacy.product_id,
        ).first()
        if existing:
            review_id_map[str(legacy.id)] = str(existing.id)
            continue

        created = CatalogReview.objects.create(
            user_id=legacy.user_id,
            product_id=legacy.product_id,
            rating=legacy.rating,
            title=legacy.title or "",
            body=legacy.content or "",
            verified_purchase=legacy.is_verified_purchase,
            helpful_votes=legacy.helpful_count or 0,
            not_helpful_votes=legacy.not_helpful_count or 0,
            moderation_status=legacy.status or "pending",
            moderation_notes=legacy.moderation_notes or "",
        )
        # Preserve timestamps from legacy rows.
        CatalogReview.objects.filter(pk=created.pk).update(
            created_at=legacy.created_at,
            updated_at=legacy.updated_at,
        )
        review_id_map[str(legacy.id)] = str(created.id)

    if not review_id_map:
        return

    # 2) Migrate legacy review images.
    for legacy_image in LegacyReviewImage.objects.iterator():
        target_review_id = review_id_map.get(str(legacy_image.review_id))
        if not target_review_id:
            continue
        exists = CatalogReviewImage.objects.filter(
            review_id=target_review_id,
            image=legacy_image.image,
            caption=legacy_image.caption or "",
            sort_order=legacy_image.sort_order or 0,
        ).exists()
        if exists:
            continue
        new_image = CatalogReviewImage.objects.create(
            review_id=target_review_id,
            image=legacy_image.image,
            caption=legacy_image.caption or "",
            sort_order=legacy_image.sort_order or 0,
        )
        CatalogReviewImage.objects.filter(pk=new_image.pk).update(created_at=legacy_image.created_at)

    # 3) Migrate legacy votes.
    for legacy_vote in LegacyReviewVote.objects.iterator():
        target_review_id = review_id_map.get(str(legacy_vote.review_id))
        if not target_review_id:
            continue
        vote, created = CatalogReviewVote.objects.get_or_create(
            review_id=target_review_id,
            user_id=legacy_vote.user_id,
            defaults={"is_helpful": legacy_vote.is_helpful},
        )
        if created:
            CatalogReviewVote.objects.filter(pk=vote.pk).update(
                created_at=legacy_vote.created_at,
                updated_at=legacy_vote.created_at,
            )
        elif vote.is_helpful != legacy_vote.is_helpful:
            vote.is_helpful = legacy_vote.is_helpful
            vote.save(update_fields=["is_helpful", "updated_at"])

    # 4) Recompute review counters and product aggregates.
    for target_review_id in set(review_id_map.values()):
        helpful = CatalogReviewVote.objects.filter(review_id=target_review_id, is_helpful=True).count()
        not_helpful = CatalogReviewVote.objects.filter(review_id=target_review_id, is_helpful=False).count()
        CatalogReview.objects.filter(pk=target_review_id).update(
            helpful_votes=helpful,
            not_helpful_votes=not_helpful,
        )

    for product_id in touched_product_ids:
        stats = CatalogReview.objects.filter(product_id=product_id).aggregate(
            total=Count("id"),
            approved=Count("id", filter=Q(moderation_status="approved")),
            avg=Avg("rating", filter=Q(moderation_status="approved")),
        )
        Product.objects.filter(pk=product_id).update(
            reviews_count=stats.get("total") or 0,
            rating_count=stats.get("approved") or 0,
            average_rating=float(stats.get("avg") or 0.0),
        )


class Migration(migrations.Migration):

    dependencies = [
        ("catalog", "0008_historicalreviewreport_historicalreviewvote_and_more"),
        ("reviews", "0002_historicalreview_historicalreviewimage_and_more"),
    ]

    operations = [
        migrations.RunPython(migrate_legacy_reviews, migrations.RunPython.noop),
    ]

