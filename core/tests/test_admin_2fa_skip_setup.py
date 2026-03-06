import re

import pytest
from django.urls import reverse

ADMIN_2FA_SETUP_SKIPPED_SESSION_KEY = "admin_2fa_setup_skipped"


@pytest.mark.django_db
def test_skip_setup_redirects_anonymous_users_to_admin_login(client):
    response = client.get("/admin/2fa/skip-setup/")

    assert response.status_code == 302
    assert response["Location"].startswith("/admin/login/?next=/admin/2fa/skip-setup/")


@pytest.mark.django_db
def test_skip_setup_sets_session_flag_and_redirects_to_requested_page(client, django_user_model):
    user = django_user_model.objects.create_superuser(
        email="admin-skip-setup@test.com",
        password="secret12345",
    )
    client.force_login(user)

    session = client.session
    session["next"] = "/admin/catalog/product/add/"
    session.save()

    response = client.get("/admin/2fa/skip-setup/")

    assert response.status_code == 302
    assert response["Location"] == "/admin/catalog/product/add/"

    session = client.session
    assert session.get(ADMIN_2FA_SETUP_SKIPPED_SESSION_KEY) is True
    assert "next" not in session


@pytest.mark.django_db
def test_skip_setup_falls_back_to_admin_index_for_unsafe_next(client, django_user_model):
    user = django_user_model.objects.create_superuser(
        email="admin-skip-setup-unsafe@test.com",
        password="secret12345",
    )
    client.force_login(user)

    response = client.get("/admin/2fa/skip-setup/?next=https://evil.example/phish")

    assert response.status_code == 302
    assert response["Location"] == reverse("admin:index")


@pytest.mark.django_db
def test_admin_login_routes_no_device_user_to_setup_instead_of_login_loop(client, django_user_model):
    user = django_user_model.objects.create_superuser(
        email="admin-login-no-device@test.com",
        password="secret12345",
    )

    login_url = "/admin/login/?next=/admin/catalog/product/add/"
    response = client.get(login_url)
    assert response.status_code == 200

    html = response.content.decode("utf-8", "ignore")
    hidden = dict(
        re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', html)
    )
    payload = {
        **hidden,
        "auth-username": user.email,
        "auth-password": "secret12345",
        "next": "/admin/catalog/product/add/",
    }

    post = client.post(login_url, data=payload)
    assert post.status_code == 302
    assert post["Location"].startswith("/admin/2fa/setup/?next=%2Fadmin%2Fcatalog%2Fproduct%2Fadd%2F")
