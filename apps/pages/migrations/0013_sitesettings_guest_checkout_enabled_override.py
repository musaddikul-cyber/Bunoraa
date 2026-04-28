from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("pages", "0012_alter_banner_background_image_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="historicalsitesettings",
            name="guest_checkout_enabled_override",
            field=models.BooleanField(
                blank=True,
                default=None,
                help_text="Override guest checkout availability. Leave blank to use the core settings default.",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="sitesettings",
            name="guest_checkout_enabled_override",
            field=models.BooleanField(
                blank=True,
                default=None,
                help_text="Override guest checkout availability. Leave blank to use the core settings default.",
                null=True,
            ),
        ),
    ]
