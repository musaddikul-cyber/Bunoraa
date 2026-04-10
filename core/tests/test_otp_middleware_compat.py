import pytest


@pytest.mark.django_db
def test_otp_middleware_adapter_keeps_boolean_is_verified_field_saveable(django_user_model):
    from core.middleware.otp import CompatibleOTPMiddleware

    user = django_user_model.objects.create_user(
        email="otp-compat@test.com",
        password="secret12345",
        is_verified=True,
    )

    CompatibleOTPMiddleware._init_user_fields(user)

    assert callable(user.is_verified)
    assert bool(user.is_verified) is True
    assert user.is_verified() is False

    user.first_name = "Updated"
    user.save()
    user.refresh_from_db()

    assert user.is_verified is True
