from django.conf import settings
from django_otp import user_has_device
from rest_framework.permissions import BasePermission


def _is_verified(user) -> bool:
    checker = getattr(user, "is_verified", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return False


class IsStaffWithMfa(BasePermission):
    message = "Admin access requires staff role and MFA verification."

    def has_permission(self, request, view) -> bool:
        user = getattr(request, "user", None)
        if not user or not user.is_authenticated:
            return False
        if not getattr(user, "is_staff", False):
            return False
        if not getattr(settings, "ADMIN_MFA_REQUIRED", True):
            return True
        if not user_has_device(user):
            return False
        return _is_verified(user)
