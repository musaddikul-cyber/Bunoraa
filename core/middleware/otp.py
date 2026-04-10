"""
OTP middleware compatibility layer for custom user models.

This project uses ``User.is_verified`` as a BooleanField (email verification),
while django-otp normally injects ``user.is_verified()`` as an OTP status
callable. This middleware keeps OTP callable behavior without breaking boolean
field saves.
"""

from __future__ import annotations

import functools

from asgiref.sync import iscoroutinefunction, markcoroutinefunction
from django.db import models
from django.utils.functional import SimpleLazyObject

from django_otp import DEVICE_ID_SESSION_KEY


def _otp_is_verified(user) -> bool:
    return user.otp_device is not None


class _OTPVerificationAdapter:
    """Callable OTP checker that also behaves like the original bool field."""

    __slots__ = ("_user", "_field_value")

    def __init__(self, user, field_value: bool):
        self._user = user
        self._field_value = bool(field_value)

    def __call__(self) -> bool:
        return _otp_is_verified(self._user)

    def __bool__(self) -> bool:
        return self._field_value

    def __eq__(self, other):
        if isinstance(other, bool):
            return bool(self) is other
        return NotImplemented

    def __repr__(self) -> str:
        return f"<OTPVerificationAdapter field_value={self._field_value}>"


class CompatibleOTPMiddleware:
    """
    OTP middleware with ``is_verified`` BooleanField compatibility.

    Mirrors django_otp.middleware.OTPMiddleware behavior while avoiding model
    field corruption on custom user models that already define ``is_verified``.
    """

    sync_capable = True
    async_capable = True

    def __init__(self, get_response):
        self.get_response = get_response
        self._is_async = iscoroutinefunction(get_response)
        if self._is_async:
            markcoroutinefunction(self)

    def __call__(self, request):
        if self._is_async:
            return self.__acall__(request)

        self._install_lazy_accessors(request)
        return self.get_response(request)

    async def __acall__(self, request):
        self._install_lazy_accessors(request)
        return await self.get_response(request)

    def _install_lazy_accessors(self, request):
        user = getattr(request, "user", None)
        if user is not None:
            request.user = SimpleLazyObject(
                functools.partial(self._verify_user_sync, request, user)
            )

        auser = getattr(request, "auser", None)
        if auser is not None:
            request.auser = functools.partial(
                self._verify_user_async_via_auser, request, auser
            )

    @staticmethod
    def _has_boolean_is_verified_field(user) -> bool:
        meta = getattr(user, "_meta", None)
        if meta is None:
            return False
        try:
            field = meta.get_field("is_verified")
        except Exception:
            return False
        return isinstance(field, models.BooleanField)

    @classmethod
    def _init_user_fields(cls, user):
        user.otp_device = None

        if cls._has_boolean_is_verified_field(user):
            field_value = bool(getattr(user, "is_verified", False))
            user.is_verified = _OTPVerificationAdapter(user, field_value)
            return

        # Default django-otp behavior for user models without this field.
        user.is_verified = functools.partial(_otp_is_verified, user)

    @staticmethod
    def _normalize_persistent_id(persistent_id: str) -> str:
        if persistent_id.count(".") > 1:
            parts = persistent_id.split(".")
            return ".".join((parts[-3], parts[-1]))
        return persistent_id

    @staticmethod
    def _finalize_device(request, user, device):
        if (device is not None) and (device.user_id != user.pk):
            device = None

        if (device is None) and (DEVICE_ID_SESSION_KEY in request.session):
            del request.session[DEVICE_ID_SESSION_KEY]

        return device

    def _verify_user_sync(self, request, user):
        self._init_user_fields(user)

        if user.is_authenticated:
            persistent_id = request.session.get(DEVICE_ID_SESSION_KEY)
            device = (
                self._device_from_persistent_id(persistent_id)
                if persistent_id
                else None
            )
            user.otp_device = self._finalize_device(request, user, device)

        return user

    def _device_from_persistent_id(self, persistent_id: str):
        from django_otp.models import Device

        persistent_id = self._normalize_persistent_id(persistent_id)
        return Device.from_persistent_id(persistent_id)

    async def _verify_user_async_via_auser(self, request, auser):
        user = await auser()
        self._init_user_fields(user)

        if user.is_authenticated:
            persistent_id = request.session.get(DEVICE_ID_SESSION_KEY)
            device = (
                await self._adevice_from_persistent_id(persistent_id)
                if persistent_id
                else None
            )
            user.otp_device = self._finalize_device(request, user, device)

        return user

    async def _adevice_from_persistent_id(self, persistent_id: str):
        from django_otp.models import Device

        persistent_id = self._normalize_persistent_id(persistent_id)
        return await Device.afrom_persistent_id(persistent_id)
