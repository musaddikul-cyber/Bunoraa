from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("catalog", "0009_migrate_legacy_reviews"),
        ("reviews", "0002_historicalreview_historicalreviewimage_and_more"),
    ]

    operations = [
        # Drop legacy historical models first; they reference reviews.Review.
        migrations.DeleteModel(name="HistoricalReviewImage"),
        migrations.DeleteModel(name="HistoricalReviewReply"),
        migrations.DeleteModel(name="HistoricalReviewVote"),
        migrations.DeleteModel(name="HistoricalReview"),
        # Drop legacy primary tables after catalog migration has copied data.
        migrations.DeleteModel(name="ReviewVote"),
        migrations.DeleteModel(name="ReviewReply"),
        migrations.DeleteModel(name="ReviewImage"),
        migrations.DeleteModel(name="Review"),
    ]
