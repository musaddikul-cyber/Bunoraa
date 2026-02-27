"""
Project-specific two-factor auth view customizations.
"""

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.shortcuts import resolve_url
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.utils.http import url_has_allowed_host_and_scheme
from django.views import View
from two_factor.utils import default_device
from two_factor.views import SetupView

ADMIN_2FA_SETUP_SKIPPED_SESSION_KEY = "admin_2fa_setup_skipped"


class AdminSetupView(SetupView):
    """
    Use an explicit skip endpoint for admin 2FA onboarding.
    """

    def get_context_data(self, form, **kwargs):
        context = super().get_context_data(form, **kwargs)
        context["cancel_url"] = resolve_url("two_factor:skip_setup")
        return context


@method_decorator(login_required, name="dispatch")
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

        redirect_to = request.session.get("next") or request.GET.get("next")
        if redirect_to and url_has_allowed_host_and_scheme(
            url=redirect_to,
            allowed_hosts={request.get_host()},
            require_https=request.is_secure(),
        ):
            return HttpResponseRedirect(redirect_to)

        return HttpResponseRedirect(reverse("admin:index"))
