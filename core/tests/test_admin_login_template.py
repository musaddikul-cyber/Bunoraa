import pytest


@pytest.fixture(autouse=True)
def allow_testserver_host(settings):
    allowed = list(getattr(settings, "ALLOWED_HOSTS", []))
    if "testserver" not in allowed:
        allowed.append("testserver")
    settings.ALLOWED_HOSTS = allowed


@pytest.mark.django_db
def test_admin_login_uses_custom_two_factor_template(client):
    response = client.get("/admin/login/")
    assert response.status_code == 200

    template_names = [template.name for template in response.templates if template.name]
    assert "two_factor/core/login.html" in template_names

    content = response.content.decode("utf-8", "ignore")
    assert "Bunoraa Control Center" in content
    assert "Provide a template named" not in content
