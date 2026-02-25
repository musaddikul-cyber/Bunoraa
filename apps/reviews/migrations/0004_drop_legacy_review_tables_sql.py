from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("reviews", "0003_deprecate_legacy_review_tables"),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
            DROP TABLE IF EXISTS reviews_historicalreviewvote;
            DROP TABLE IF EXISTS reviews_historicalreviewreply;
            DROP TABLE IF EXISTS reviews_historicalreviewimage;
            DROP TABLE IF EXISTS reviews_historicalreview;
            DROP TABLE IF EXISTS reviews_reviewvote;
            DROP TABLE IF EXISTS reviews_reviewreply;
            DROP TABLE IF EXISTS reviews_reviewimage;
            DROP TABLE IF EXISTS reviews_review;
            """,
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]

