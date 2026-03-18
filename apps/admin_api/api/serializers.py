from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group, Permission
from rest_framework import serializers

from apps.admin_api.models import AdminAuditLog

User = get_user_model()


class AdminAuditLogSerializer(serializers.ModelSerializer):
    actor_email = serializers.EmailField(source="actor.email", read_only=True)

    class Meta:
        model = AdminAuditLog
        fields = (
            "id",
            "created_at",
            "actor",
            "actor_email",
            "action",
            "resource_type",
            "resource_id",
            "path",
            "method",
            "status_code",
            "ip_address",
            "user_agent",
            "metadata",
        )
        read_only_fields = fields


class AdminUserSerializer(serializers.ModelSerializer):
    groups = serializers.SlugRelatedField(slug_field="name", many=True, read_only=True)

    class Meta:
        model = User
        fields = (
            "id",
            "email",
            "first_name",
            "last_name",
            "is_active",
            "is_staff",
            "is_superuser",
            "last_login",
            "date_joined",
            "groups",
        )
        read_only_fields = fields


class AdminPermissionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Permission
        fields = ("id", "codename", "name")
        read_only_fields = fields


class AdminGroupSerializer(serializers.ModelSerializer):
    permissions = AdminPermissionSerializer(many=True, read_only=True)

    class Meta:
        model = Group
        fields = ("id", "name", "permissions")
        read_only_fields = fields
