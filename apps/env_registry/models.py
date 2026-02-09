import base64
from typing import Optional

from django.conf import settings
from django.db import models

from apps.accounts.services import CredentialEncryptionService


class EnvCategory(models.Model):
    slug = models.SlugField(unique=True)
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True)
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "Env Category"
        verbose_name_plural = "Env Categories"
        ordering = ["order", "name"]

    def __str__(self) -> str:
        return self.name


class EnvVariable(models.Model):
    TYPE_STRING = "string"
    TYPE_BOOL = "bool"
    TYPE_INT = "int"
    TYPE_FLOAT = "float"
    TYPE_JSON = "json"
    TYPE_CHOICES = [
        (TYPE_STRING, "String"),
        (TYPE_BOOL, "Boolean"),
        (TYPE_INT, "Integer"),
        (TYPE_FLOAT, "Float"),
        (TYPE_JSON, "JSON"),
    ]

    key = models.CharField(max_length=200, unique=True)
    category = models.ForeignKey(EnvCategory, on_delete=models.PROTECT, related_name="variables")
    description = models.TextField(blank=True)
    is_secret = models.BooleanField(default=False)
    required = models.BooleanField(default=False)
    restart_required = models.BooleanField(default=False)
    runtime_apply = models.BooleanField(default=True)
    value_type = models.CharField(max_length=20, choices=TYPE_CHOICES, default=TYPE_STRING)
    targets = models.JSONField(default=dict, blank=True)
    schema_version = models.PositiveIntegerField(default=1)
    is_active = models.BooleanField(default=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Env Variable"
        verbose_name_plural = "Env Variables"
        ordering = ["key"]

    def __str__(self) -> str:
        return self.key


class EnvValue(models.Model):
    ENV_DEVELOPMENT = "development"
    ENV_PRODUCTION = "production"
    ENV_CHOICES = [
        (ENV_DEVELOPMENT, "Development"),
        (ENV_PRODUCTION, "Production"),
    ]

    variable = models.ForeignKey(EnvVariable, on_delete=models.CASCADE, related_name="values")
    environment = models.CharField(max_length=20, choices=ENV_CHOICES)
    value_plain = models.TextField(blank=True)
    value_encrypted = models.BinaryField(blank=True, null=True)
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="env_values_updated",
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Env Value"
        verbose_name_plural = "Env Values"
        unique_together = ("variable", "environment")
        ordering = ["variable__key", "environment"]

    def __str__(self) -> str:
        return f"{self.variable.key} ({self.environment})"

    def has_value(self) -> bool:
        if self.variable.is_secret:
            return bool(self.value_encrypted or self.value_plain)
        return bool(self.value_plain)

    def get_value(self) -> Optional[str]:
        if self.variable.is_secret:
            if self.value_encrypted:
                decrypted = CredentialEncryptionService.decrypt_password(self.value_encrypted)
                if decrypted is not None:
                    return decrypted
            return self.value_plain or None
        return self.value_plain or None

    def set_value(self, value: str, is_secret: bool) -> None:
        if is_secret:
            encrypted = CredentialEncryptionService.encrypt_password(value)
            if encrypted:
                self.value_encrypted = encrypted
                self.value_plain = ""
                return
            # Fallback if encryption unavailable
            self.value_plain = value
            self.value_encrypted = None
            return
        self.value_plain = value
        self.value_encrypted = None

    def masked_value(self) -> str:
        if self.variable.is_secret and self.has_value():
            return "********"
        return self.value_plain or ""
