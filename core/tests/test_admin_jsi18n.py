import pytest


@pytest.fixture(autouse=True)
def allow_testserver_host(settings):
    allowed = list(getattr(settings, "ALLOWED_HOSTS", []))
    if "testserver" not in allowed:
        allowed.append("testserver")
    settings.ALLOWED_HOSTS = allowed


@pytest.mark.django_db
def test_admin_jsi18n_serves_javascript_without_admin_auth(client):
    response = client.get("/admin/jsi18n/")

    assert response.status_code == 200
    content_type = str(response.get("Content-Type", "")).lower()
    assert "javascript" in content_type
    assert b"const globals = this" in response.content[:200]
