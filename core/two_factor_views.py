"""
Project-specific two-factor auth view customizations.
"""

from urllib.parse import parse_qs, urlencode, urlparse

from django.contrib.auth import REDIRECT_FIELD_NAME, login
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.shortcuts import resolve_url
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.utils.http import url_has_allowed_host_and_scheme
from django.views import View
from two_factor.utils import default_device
from two_factor.views import LoginView as BaseTwoFactorLoginView, SetupView

ADMIN_2FA_SETUP_SKIPPED_SESSION_KEY = "admin_2fa_setup_skipped"


def _is_admin_2fa_interstitial(path: str) -> bool:
    normalized = (path or "").strip().lower()
    if not normalized:
        return False
    if normalized in {"/admin/login", "/admin/login/"}:
        return True
    return normalized.startswith("/admin/2fa/")


def _unwrap_next_target(redirect_to: str | None) -> str | None:
    """
    Resolve nested `next=` chains that point to admin login/setup/skip pages.

    This prevents loops like:
    /admin/login/?next=/admin/2fa/skip-setup/?next=...
    """
    current = (redirect_to or "").strip()
    if not current:
        return None

    seen: set[str] = set()
    for _ in range(8):
        if not current or current in seen:
            return None
        seen.add(current)

        parsed = urlparse(current)
        if not _is_admin_2fa_interstitial(parsed.path):
            return current

        nested_next = parse_qs(parsed.query, keep_blank_values=True).get(REDIRECT_FIELD_NAME, [None])[0]
        if not nested_next:
            return None
        current = nested_next.strip()

    return None


def get_safe_admin_redirect_target(request, redirect_to: str | None) -> str | None:
    candidate = _unwrap_next_target(redirect_to)
    if not candidate:
        return None
    if url_has_allowed_host_and_scheme(
        url=candidate,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return candidate
    return None


class AdminLoginView(BaseTwoFactorLoginView):
    """
    Ensure admin destinations route through 2FA setup when the user has no OTP device.
    """

    def _is_admin_target(self, redirect_to: str) -> bool:
        return redirect_to == "/admin" or redirect_to.startswith("/admin/")

    def get_success_url(self):
        redirect_to = get_safe_admin_redirect_target(self.request, super().get_success_url())
        return redirect_to or resolve_url("admin:index")

    def done(self, form_list, **kwargs):
        user = self.get_user()
        login(self.request, user)
        redirect_to = self.get_success_url()

        # Prevent admin login loops for users without an OTP device.
        if not default_device(user) and self._is_admin_target(redirect_to):
            self.request.session["next"] = redirect_to
            self.request.session.modified = True
            setup_url = resolve_url("two_factor:setup")
            query = urlencode({REDIRECT_FIELD_NAME: redirect_to})
            return HttpResponseRedirect(f"{setup_url}?{query}")

        return super().done(form_list, **kwargs)


class AdminSetupView(SetupView):
    """
    Use an explicit skip endpoint for admin 2FA onboarding.
    """

    def get_context_data(self, form, **kwargs):
        context = super().get_context_data(form, **kwargs)
        cancel_url = resolve_url("two_factor:skip_setup")
        request_next = get_safe_admin_redirect_target(
            self.request,
            self.request.GET.get("next") or self.request.session.get("next"),
        )
        if request_next:
            cancel_url = f"{cancel_url}?{urlencode({'next': request_next})}"
        context["cancel_url"] = cancel_url
        return context


@method_decorator(login_required(login_url="/admin/login/"), name="dispatch")
class SkipAdminSetupView(View):
    """
    Allow staff without any OTP device to continue for the current session.
    """

    http_method_names = ["get"]

    def get(self, request, *args, **kwargs):
        if default_device(request.user):
            return HttpResponseRedirect(resolve_url("admin:index"))

        request.session[ADMIN_2FA_SETUP_SKIPPED_SESSION_KEY] = True
        request.session.modified = True

        redirect_to = get_safe_admin_redirect_target(
            request,
            request.GET.get("next") or request.session.get("next"),
        )
        request.session.pop("next", None)
        request.session.modified = True
        return HttpResponseRedirect(redirect_to or reverse("admin:index"))
