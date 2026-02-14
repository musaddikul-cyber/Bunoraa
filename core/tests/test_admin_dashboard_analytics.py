import pytest
from django.urls import reverse

from core.admin_dashboard import get_date_range


@pytest.fixture(autouse=True)
def allow_testserver_host(settings):
    allowed = list(getattr(settings, "ALLOWED_HOSTS", []))
    if "testserver" not in allowed:
        allowed.append("testserver")
    settings.ALLOWED_HOSTS = allowed


@pytest.mark.django_db
def test_admin_dashboard_stats_endpoint_shape(client, django_user_model):
    staff = django_user_model.objects.create_user(
        email="staff-analytics@test.com",
        password="secret123",
        is_staff=True,
        is_superuser=True,
    )
    client.force_login(staff)

    response = client.get(reverse("dashboard_stats_api"), {"period": "7d"})
    assert response.status_code == 200

    payload = response.json()
    assert "total_revenue" in payload
    assert "total_orders" in payload
    assert "unique_visitors" in payload
    assert "timeseries" in payload
    assert "top_products" in payload
    assert "top_categories" in payload
    assert "generated_at" in payload


@pytest.mark.django_db
def test_admin_dashboard_page_renders(client, django_user_model):
    staff = django_user_model.objects.create_user(
        email="staff-dashboard@test.com",
        password="secret123",
        is_staff=True,
        is_superuser=True,
    )
    client.force_login(staff)

    response = client.get(reverse("admin_dashboard"))
    assert response.status_code == 200
    assert b"Analytics Dashboard" in response.content


def test_get_date_range_custom_period():
    start, end, period = get_date_range("custom", "2026-01-01", "2026-01-07")
    assert period == "custom"
    assert start.date().isoformat() == "2026-01-01"
    assert end.date().isoformat() == "2026-01-08"
