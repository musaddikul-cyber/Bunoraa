import uuid
import django.db.models.deletion
from django.db import migrations, models
from django.utils import timezone


class Migration(migrations.Migration):

    dependencies = [
        ("notifications", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="notification",
            name="channels_requested",
            field=models.JSONField(default=list, help_text="Channels requested for delivery"),
        ),
        migrations.AddField(
            model_name="notification",
            name="category",
            field=models.CharField(
                choices=[
                    ("marketing", "Marketing"),
                    ("transactional", "Transactional"),
                    ("system", "System"),
                ],
                default="transactional",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="notification",
            name="priority",
            field=models.CharField(
                choices=[
                    ("low", "Low"),
                    ("normal", "Normal"),
                    ("high", "High"),
                    ("urgent", "Urgent"),
                ],
                default="normal",
                max_length=10,
            ),
        ),
        migrations.AddField(
            model_name="notification",
            name="status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("processing", "Processing"),
                    ("sent", "Sent"),
                    ("partial", "Partial"),
                    ("failed", "Failed"),
                    ("skipped", "Skipped"),
                ],
                default="pending",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="notification",
            name="dedupe_key",
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name="notification",
            name="expires_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="notification",
            name="updated_at",
            field=models.DateTimeField(auto_now=True, default=timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="notificationpreference",
            name="email_enabled",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="notificationpreference",
            name="digest_frequency",
            field=models.CharField(
                choices=[
                    ("immediate", "Immediate"),
                    ("hourly", "Hourly"),
                    ("daily", "Daily"),
                    ("weekly", "Weekly"),
                    ("never", "Never"),
                ],
                default="immediate",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="notificationpreference",
            name="quiet_hours_start",
            field=models.TimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="notificationpreference",
            name="quiet_hours_end",
            field=models.TimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="notificationpreference",
            name="timezone",
            field=models.CharField(default="Asia/Dhaka", max_length=50),
        ),
        migrations.AddField(
            model_name="notificationpreference",
            name="marketing_opt_in",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="notificationpreference",
            name="transactional_opt_in",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="notificationpreference",
            name="per_type_overrides",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="pushtoken",
            name="platform",
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name="pushtoken",
            name="app_version",
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name="pushtoken",
            name="locale",
            field=models.CharField(blank=True, max_length=10, null=True),
        ),
        migrations.AddField(
            model_name="pushtoken",
            name="timezone",
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name="pushtoken",
            name="browser",
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name="pushtoken",
            name="user_agent",
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
        migrations.AddField(
            model_name="pushtoken",
            name="last_ip",
            field=models.GenericIPAddressField(blank=True, null=True),
        ),
        migrations.CreateModel(
            name="NotificationTemplate",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=100)),
                (
                    "notification_type",
                    models.CharField(
                        choices=[
                            ("order_placed", "Order Placed"),
                            ("order_confirmed", "Order Confirmed"),
                            ("order_shipped", "Order Shipped"),
                            ("order_delivered", "Order Delivered"),
                            ("order_cancelled", "Order Cancelled"),
                            ("order_refunded", "Order Refunded"),
                            ("payment_received", "Payment Received"),
                            ("payment_failed", "Payment Failed"),
                            ("review_approved", "Review Approved"),
                            ("review_rejected", "Review Rejected"),
                            ("price_drop", "Price Drop"),
                            ("back_in_stock", "Back In Stock"),
                            ("wishlist_sale", "Wishlist Item On Sale"),
                            ("account_created", "Account Created"),
                            ("password_reset", "Password Reset"),
                            ("promo_code", "Promo Code"),
                            ("general", "General"),
                        ],
                        max_length=50,
                    ),
                ),
                (
                    "channel",
                    models.CharField(
                        choices=[
                            ("email", "Email"),
                            ("sms", "SMS"),
                            ("push", "Push Notification"),
                            ("in_app", "In-App Notification"),
                        ],
                        max_length=20,
                    ),
                ),
                ("language", models.CharField(default="en", max_length=10)),
                ("subject", models.CharField(blank=True, max_length=200)),
                ("body", models.TextField(blank=True, help_text="Body for SMS/push/in-app")),
                ("html_template", models.TextField(blank=True, help_text="HTML template for email")),
                ("text_template", models.TextField(blank=True, help_text="Plain text template for email")),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["name"],
            },
        ),
        migrations.CreateModel(
            name="NotificationDelivery",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                (
                    "channel",
                    models.CharField(
                        choices=[
                            ("email", "Email"),
                            ("sms", "SMS"),
                            ("push", "Push Notification"),
                            ("in_app", "In-App Notification"),
                        ],
                        max_length=20,
                    ),
                ),
                ("provider", models.CharField(blank=True, max_length=50, null=True)),
                ("external_id", models.CharField(blank=True, max_length=255, null=True)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("queued", "Queued"),
                            ("batched", "Batched"),
                            ("sent", "Sent"),
                            ("failed", "Failed"),
                            ("skipped", "Skipped"),
                        ],
                        default="pending",
                        max_length=20,
                    ),
                ),
                ("attempts", models.PositiveSmallIntegerField(default=0)),
                ("error", models.TextField(blank=True, null=True)),
                ("scheduled_for", models.DateTimeField(blank=True, null=True)),
                ("sent_at", models.DateTimeField(blank=True, null=True)),
                ("delivered_at", models.DateTimeField(blank=True, null=True)),
                ("opened_at", models.DateTimeField(blank=True, null=True)),
                ("clicked_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "notification",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="deliveries",
                        to="notifications.notification",
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddField(
            model_name="emaillog",
            name="notification",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="email_logs",
                to="notifications.notification",
            ),
        ),
        migrations.AddField(
            model_name="emaillog",
            name="delivery",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="email_logs",
                to="notifications.notificationdelivery",
            ),
        ),
        migrations.AddIndex(
            model_name="notification",
            index=models.Index(fields=["dedupe_key"], name="notificatio_dedupe_4f8a2f_idx"),
        ),
        migrations.AddIndex(
            model_name="notification",
            index=models.Index(fields=["category"], name="notificatio_categor_764ad0_idx"),
        ),
        migrations.AddIndex(
            model_name="notificationtemplate",
            index=models.Index(fields=["notification_type", "channel", "language"], name="notificatio_notific_0f7f7f_idx"),
        ),
        migrations.AddIndex(
            model_name="notificationdelivery",
            index=models.Index(fields=["channel", "status"], name="notificatio_channel_0b08c6_idx"),
        ),
        migrations.AddIndex(
            model_name="notificationdelivery",
            index=models.Index(fields=["notification", "channel"], name="notificatio_notific_6a2c7e_idx"),
        ),
        migrations.AlterUniqueTogether(
            name="notificationtemplate",
            unique_together={("notification_type", "channel", "language")},
        ),
    ]
