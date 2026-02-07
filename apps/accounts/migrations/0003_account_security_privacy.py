from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0002_behavior_models"),
    ]

    operations = [
        migrations.AddField(
            model_name="usersession",
            name="session_type",
            field=models.CharField(
                choices=[("behavior", "Behavior"), ("auth", "Auth")],
                db_index=True,
                default="behavior",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="usersession",
            name="access_jti",
            field=models.CharField(blank=True, db_index=True, max_length=255),
        ),
        migrations.AddField(
            model_name="usersession",
            name="refresh_jti",
            field=models.CharField(blank=True, db_index=True, max_length=255),
        ),
        migrations.AddField(
            model_name="usersession",
            name="revoked_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="usersession",
            name="is_active",
            field=models.BooleanField(db_index=True, default=True),
        ),
        migrations.AddIndex(
            model_name="usersession",
            index=models.Index(fields=["user", "session_type", "-started_at"], name="accounts_us_user_id_8b7b5e_idx"),
        ),
        migrations.CreateModel(
            name="WebAuthnCredential",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("credential_id", models.BinaryField(unique=True)),
                ("public_key", models.BinaryField()),
                ("sign_count", models.PositiveIntegerField(default=0)),
                ("transports", models.JSONField(blank=True, default=list)),
                ("nickname", models.CharField(blank=True, max_length=100)),
                ("is_active", models.BooleanField(default=True)),
                ("last_used_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="webauthn_credentials", to="accounts.user")),
            ],
            options={
                "verbose_name": "WebAuthn credential",
                "verbose_name_plural": "WebAuthn credentials",
            },
        ),
        migrations.CreateModel(
            name="WebAuthnChallenge",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("challenge", models.CharField(db_index=True, max_length=255)),
                ("challenge_type", models.CharField(choices=[("register", "Register"), ("login", "Login"), ("mfa", "MFA")], max_length=20)),
                ("expires_at", models.DateTimeField()),
                ("consumed", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("user", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="webauthn_challenges", to="accounts.user")),
            ],
            options={
                "verbose_name": "WebAuthn challenge",
                "verbose_name_plural": "WebAuthn challenges",
            },
        ),
        migrations.CreateModel(
            name="DataExportJob",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("status", models.CharField(choices=[("pending", "Pending"), ("processing", "Processing"), ("completed", "Completed"), ("failed", "Failed")], default="pending", max_length=20)),
                ("file", models.FileField(blank=True, null=True, upload_to="exports/")),
                ("error_message", models.TextField(blank=True)),
                ("requested_at", models.DateTimeField(auto_now_add=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("expires_at", models.DateTimeField(blank=True, null=True)),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="data_exports", to="accounts.user")),
            ],
            options={
                "verbose_name": "data export job",
                "verbose_name_plural": "data export jobs",
                "ordering": ["-requested_at"],
            },
        ),
        migrations.CreateModel(
            name="AccountDeletionRequest",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("status", models.CharField(choices=[("pending", "Pending"), ("cancelled", "Cancelled"), ("completed", "Completed")], default="pending", max_length=20)),
                ("requested_at", models.DateTimeField(auto_now_add=True)),
                ("scheduled_for", models.DateTimeField()),
                ("processed_at", models.DateTimeField(blank=True, null=True)),
                ("cancelled_at", models.DateTimeField(blank=True, null=True)),
                ("reason", models.CharField(blank=True, max_length=255)),
                ("user", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="deletion_request", to="accounts.user")),
            ],
            options={
                "verbose_name": "account deletion request",
                "verbose_name_plural": "account deletion requests",
            },
        ),
        migrations.AddIndex(
            model_name="webauthncredential",
            index=models.Index(fields=["user", "created_at"], name="accounts_we_user_id_585f6c_idx"),
        ),
        migrations.AddIndex(
            model_name="webauthnchallenge",
            index=models.Index(fields=["challenge", "challenge_type"], name="accounts_we_challen_5d64e5_idx"),
        ),
        migrations.AddIndex(
            model_name="webauthnchallenge",
            index=models.Index(fields=["user", "created_at"], name="accounts_we_user_id_79a4bf_idx"),
        ),
        migrations.AddIndex(
            model_name="dataexportjob",
            index=models.Index(fields=["user", "-requested_at"], name="accounts_da_user_id_eb927a_idx"),
        ),
        migrations.AddIndex(
            model_name="dataexportjob",
            index=models.Index(fields=["status"], name="accounts_da_status_1b9c6a_idx"),
        ),
        migrations.AddIndex(
            model_name="accountdeletionrequest",
            index=models.Index(fields=["status"], name="accounts_ac_status_7bcb60_idx"),
        ),
        migrations.AddIndex(
            model_name="accountdeletionrequest",
            index=models.Index(fields=["scheduled_for"], name="accounts_ac_schedule_0f9d50_idx"),
        ),
    ]
