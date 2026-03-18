from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="AdminAuditLog",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("action", models.CharField(max_length=100)),
                ("resource_type", models.CharField(blank=True, max_length=100)),
                ("resource_id", models.CharField(blank=True, max_length=100)),
                ("path", models.CharField(max_length=255)),
                ("method", models.CharField(max_length=10)),
                ("status_code", models.PositiveIntegerField()),
                ("ip_address", models.GenericIPAddressField(blank=True, null=True)),
                ("user_agent", models.TextField(blank=True)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                (
                    "actor",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="admin_audit_logs",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="adminauditlog",
            index=models.Index(fields=["action", "-created_at"], name="admin_api_a_action_4b9a22_idx"),
        ),
        migrations.AddIndex(
            model_name="adminauditlog",
            index=models.Index(fields=["resource_type", "resource_id"], name="admin_api_a_resource_1f6d5f_idx"),
        ),
    ]
