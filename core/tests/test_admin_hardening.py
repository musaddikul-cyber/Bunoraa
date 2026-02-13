import pytest

from django.conf import settings
from django.contrib import admin


@pytest.mark.django_db
def test_admin_site_requires_otp():
    import core.urls  # noqa: F401 - triggers admin site patch
    from two_factor.admin import AdminSiteOTPRequired

    assert isinstance(admin.site, AdminSiteOTPRequired)


def test_axes_settings_configured():
    assert "axes.backends.AxesBackend" in settings.AUTHENTICATION_BACKENDS
    assert "axes.middleware.AxesMiddleware" in settings.MIDDLEWARE
    assert getattr(settings, "AXES_CLIENT_IP_CALLABLE", "") == "core.utils.axes.get_client_ip"


@pytest.mark.django_db
def test_simple_history_registered_for_user(django_user_model):
    user = django_user_model.objects.create_user(email="history@test.com", password="secret123")
    assert hasattr(user, "history")
    assert user.history.count() >= 1


def test_import_export_resource_excludes_sensitive_fields():
    from core.admin_mixins import SafeModelResource
    from apps.accounts.models import User

    resource_cls = SafeModelResource.for_model(User)
    excluded = set(getattr(resource_cls.Meta, "exclude", []))
    assert "password" in excluded


def test_export_sanitizes_formula_injection():
    from core.admin_mixins import sanitize_export_value

    assert sanitize_export_value("=1+1") == "'=1+1"
    assert sanitize_export_value("+SUM(A1:A2)") == "'+SUM(A1:A2)"
    assert sanitize_export_value("@cmd") == "'@cmd"
    assert sanitize_export_value("normal") == "normal"
