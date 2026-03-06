from django.urls import path
from django.utils.module_loading import import_string


def _lazy_view(path):
    def _view(request, *args, **kwargs):
        view = import_string(path)
        return view.as_view()(request, *args, **kwargs)

    return _view

app_name = "two_factor"

urlpatterns = [
    path("admin/login/", _lazy_view("core.two_factor_views.AdminLoginView"), name="login"),
    path("admin/2fa/", _lazy_view("two_factor.views.ProfileView"), name="profile"),
    path("admin/2fa/setup/", _lazy_view("core.two_factor_views.AdminSetupView"), name="setup"),
    path("admin/2fa/skip-setup/", _lazy_view("core.two_factor_views.SkipAdminSetupView"), name="skip_setup"),
    path("admin/2fa/qrcode/", _lazy_view("two_factor.views.QRGeneratorView"), name="qr"),
    path("admin/2fa/setup/complete/", _lazy_view("two_factor.views.SetupCompleteView"), name="setup_complete"),
    path("admin/2fa/backup/tokens/", _lazy_view("two_factor.views.BackupTokensView"), name="backup_tokens"),
    path("admin/2fa/disable/", _lazy_view("two_factor.views.DisableView"), name="disable"),
]
