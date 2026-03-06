"""
Project-specific two-factor auth view customizations.
"""

from django.contrib.auth import REDIRECT_FIELD_NAME, login
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.shortcuts import resolve_url
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.utils.http import url_has_allowed_host_and_scheme
from django.views import View
from urllib.parse import urlencode
from two_factor.utils import default_device
from two_factor.views import LoginView as BaseTwoFactorLoginView, SetupView

ADMIN_2FA_SETUP_SKIPPED_SESSION_KEY = "admin_2fa_setup_skipped"


class AdminLoginView(BaseTwoFactorLoginView):
    """
    Ensure admin destinations route through 2FA setup when the user has no OTP device.
    """

    def _is_admin_target(self, redirect_to: str) -> bool:
        return redirect_to == "/admin" or redirect_to.startswith("/admin/")

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
        redirect_to = request_next = self.request.GET.get("next") or self.request.session.get("next")
        if redirect_to and url_has_allowed_host_and_scheme(
            url=redirect_to,
            allowed_hosts={self.request.get_host()},
            require_https=self.request.is_secure(),
        ):
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

        redirect_to = request.GET.get("next") or request.session.get("next")
        if redirect_to and url_has_allowed_host_and_scheme(
            url=redirect_to,
            allowed_hosts={request.get_host()},
            require_https=request.is_secure(),
        ):
            request.session.pop("next", None)
            request.session.modified = True
            return HttpResponseRedirect(redirect_to)

        request.session.pop("next", None)
        request.session.modified = True
        return HttpResponseRedirect(reverse("admin:index"))
