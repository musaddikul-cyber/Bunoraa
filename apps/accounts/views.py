"""
Account views - Frontend pages
"""
from django.shortcuts import redirect, get_object_or_404
from django.views.generic import TemplateView, View
from django.views.generic.edit import FormView
from django.contrib.auth import REDIRECT_FIELD_NAME, login, logout, update_session_auth_hash
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.forms import PasswordChangeForm
from django.http import HttpResponseBadRequest, HttpResponseForbidden
from django.contrib import messages
from django.urls import reverse, reverse_lazy
from django.conf import settings
from django.utils.http import url_has_allowed_host_and_scheme
from django.core.exceptions import ValidationError
from urllib.parse import urlencode, urlparse
from social_core.actions import do_auth
from social_core.exceptions import MissingBackend
from social_django.utils import load_backend, load_strategy
from .services import UserService, AddressService
from .models import Address
from apps.i18n.services import GeoService as CountryService
from .forms import LoginForm, RegistrationForm


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}


def _normalize_origin(value):
    raw = (value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _host_to_hostname(host_value):
    raw = (host_value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}", scheme="http")
    return (parsed.hostname or "").lower().strip()


def _is_local_host(host_value):
    host = _host_to_hostname(host_value)
    return host in _LOCAL_HOSTS or host.endswith(".local")


def _build_absolute_redirect_uri(request, uri):
    uri = (uri or "").strip()
    if not uri:
        return ""
    if uri.startswith("//"):
        uri = f"{request.scheme}:{uri}"
    elif not uri.startswith("http://") and not uri.startswith("https://"):
        uri = request.build_absolute_uri(uri)

    if getattr(settings, "SOCIAL_AUTH_REDIRECT_IS_HTTPS", False):
        parsed = urlparse(uri)
        if parsed.scheme != "https":
            uri = parsed._replace(scheme="https").geturl()
    return uri


def _normalize_social_complete_redirect_uri(backend, uri):
    """Normalize OAuth callback path so begin/complete use the same redirect URI.

    Social auth complete views resolve to `/oauth/complete/<backend>/`. If a
    configured URI omits this trailing slash, auth can succeed but token
    exchange fails with `redirect_uri_mismatch` because complete() uses the
    canonical slash form.
    """
    uri = (uri or "").strip()
    if not uri:
        return ""

    canonical_path = reverse("social:complete", args=(backend,))
    canonical_without_slash = canonical_path[:-1] if canonical_path.endswith("/") else canonical_path
    parsed = urlparse(uri)
    if parsed.path != canonical_without_slash:
        return uri
    return parsed._replace(path=canonical_path).geturl()


def _has_local_redirect_port_mismatch(request, uri):
    if not uri:
        return False
    parsed = urlparse(uri if "://" in uri else f"{request.scheme}://{request.get_host()}{uri}")
    if not parsed.hostname or not _is_local_host(request.get_host()):
        return False
    request_host = urlparse(f"{request.scheme}://{request.get_host()}")
    return _is_local_host(parsed.hostname) and parsed.port != request_host.port


def _infer_local_frontend_origin(request):
    for header in ("HTTP_ORIGIN", "HTTP_REFERER"):
        candidate = (request.META.get(header) or "").strip()
        if not candidate:
            continue
        parsed = urlparse(candidate)
        if parsed.scheme and parsed.netloc and _is_local_host(parsed.netloc):
            return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")

    if _is_local_host(request.get_host()):
        return "http://localhost:3000"
    return ""


class OAuthBeginView(View):
    """Start social login and reset any existing auth session for login flows."""

    def _begin(self, request, backend):
        process = (request.GET.get("process") or "").strip().lower()
        if request.user.is_authenticated and process != "connect":
            logout(request)

        redirect_uri = reverse("social:complete", args=(backend,))
        configured_google_redirect_uri = (
            getattr(settings, "SOCIAL_AUTH_GOOGLE_OAUTH2_REDIRECT_URI", "").strip()
        )
        if backend == "google-oauth2" and configured_google_redirect_uri:
            if _has_local_redirect_port_mismatch(request, configured_google_redirect_uri):
                redirect_uri = _build_absolute_redirect_uri(request, redirect_uri)
            else:
                redirect_uri = _build_absolute_redirect_uri(request, configured_google_redirect_uri)
        else:
            redirect_uri = _build_absolute_redirect_uri(request, redirect_uri)
        redirect_uri = _normalize_social_complete_redirect_uri(backend, redirect_uri)

        request.session["social_login_flow"] = backend
        request.session.modified = True

        try:
            strategy = load_strategy(request)
            social_backend = load_backend(
                strategy,
                backend,
                redirect_uri=redirect_uri,
            )
        except MissingBackend:
            return HttpResponseBadRequest("Unsupported social backend.")

        return do_auth(social_backend, redirect_name=REDIRECT_FIELD_NAME)

    def get(self, request, backend):
        return self._begin(request, backend)

    def post(self, request, backend):
        return self._begin(request, backend)


class OAuthCallbackRedirectView(View):
    """Redirect API-domain OAuth callback to the frontend callback page."""

    def get(self, request):
        frontend_origin = _get_frontend_origin(request=request)
        if not frontend_origin:
            return redirect("/")
        target = f"{frontend_origin}{request.get_full_path()}"
        return redirect(target)


def _get_frontend_origin(request=None):
    configured_origin = _normalize_origin(
        getattr(settings, "NEXT_FRONTEND_ORIGIN", "").strip()
        or getattr(settings, "SITE_URL", "").strip()
    )
    if request and _is_local_host(request.get_host()):
        if not configured_origin or not _is_local_host(configured_origin):
            inferred_origin = _infer_local_frontend_origin(request)
            if inferred_origin:
                return inferred_origin
    return configured_origin


def _build_frontend_url(path, next_url=None, request=None):
    origin = _get_frontend_origin(request=request)
    if not origin:
        return None
    if next_url:
        return f"{origin}{path}?{urlencode({'next': next_url})}"
    return f"{origin}{path}"


def _get_safe_next_url_for_request(request):
    next_url = request.POST.get('next') or request.GET.get('next')
    if not next_url:
        return None

    allowed_hosts = {request.get_host()}
    allowed_hosts.update(getattr(settings, 'ALLOWED_HOSTS', []))

    if url_has_allowed_host_and_scheme(
        next_url,
        allowed_hosts=allowed_hosts,
        require_https=request.is_secure(),
    ):
        return next_url
    return None


class AccountDashboardView(LoginRequiredMixin, TemplateView):
    """User account dashboard."""
    template_name = 'accounts/dashboard.html'
    login_url = '/account/login/'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'My Account'
        return context


class ProfileView(LoginRequiredMixin, TemplateView):
    """User profile page."""
    template_name = 'accounts/profile.html'
    login_url = '/account/login/'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'My Profile'
        return context


class ChangePasswordView(LoginRequiredMixin, View):
    """Handle password change submission from profile page."""
    login_url = '/account/login/'

    def post(self, request):
        form = PasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            messages.success(request, 'Password updated successfully.')
        else:
            messages.error(request, 'Please correct the errors below and try again.')
        return redirect('accounts:profile')

    def get(self, request):
        return redirect('accounts:profile')


class DeleteAccountView(LoginRequiredMixin, View):
    """Handle account deletion from profile modal."""
    login_url = '/account/login/'

    def post(self, request):
        password = request.POST.get('password', '')
        user = request.user

        if not user.check_password(password):
            messages.error(request, 'Incorrect password. Account not deleted.')
            return redirect('accounts:profile')

        user.delete()
        messages.success(request, 'Your account has been deleted. Sorry to see you go.')
        return redirect('home')

    def get(self, request):
        return redirect('accounts:profile')


class AddressListView(LoginRequiredMixin, TemplateView):
    """User addresses page."""
    template_name = 'accounts/addresses.html'
    login_url = '/account/login/'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'My Addresses'
        context['addresses'] = AddressService.get_user_addresses(self.request.user)
        context['countries'] = CountryService.get_shipping_countries()
        return context


class AddAddressView(LoginRequiredMixin, View):
    """Create a new address from the modal form."""
    login_url = '/account/login/'

    def post(self, request):
        data = self._extract_address_data(request)
        try:
            AddressService.create_address(user=request.user, **data)
            messages.success(request, 'Address added successfully.')
        except ValidationError as exc:
            message = exc.message if hasattr(exc, 'message') else ''
            if not message and hasattr(exc, 'messages') and exc.messages:
                message = exc.messages[0]
            messages.error(request, message or 'You can save up to 4 addresses.')
        return redirect('accounts:addresses')

    def get(self, request):
        return redirect('accounts:addresses')

    def _extract_address_data(self, request):
        return {
            'address_type': request.POST.get('address_type') or Address.AddressType.BOTH,
            'full_name': request.POST.get('full_name', '').strip(),
            'phone': request.POST.get('phone', '').strip(),
            'address_line_1': request.POST.get('address_line_1', '').strip(),
            'address_line_2': request.POST.get('address_line_2', '').strip(),
            'city': request.POST.get('city', '').strip(),
            'state': request.POST.get('state', '').strip(),
            'postal_code': request.POST.get('postal_code', '').strip(),
            'country': request.POST.get('country', '').strip(),
            'is_default': bool(request.POST.get('is_default')),
        }


class EditAddressView(LoginRequiredMixin, View):
    """Update an existing address for the current user."""
    login_url = '/account/login/'

    def post(self, request, pk):
        address = get_object_or_404(Address, pk=pk, user=request.user, is_deleted=False)
        data = {
            'full_name': request.POST.get('full_name', '').strip(),
            'phone': request.POST.get('phone', '').strip(),
            'address_line_1': request.POST.get('address_line_1', '').strip(),
            'address_line_2': request.POST.get('address_line_2', '').strip(),
            'city': request.POST.get('city', '').strip(),
            'state': request.POST.get('state', '').strip(),
            'postal_code': request.POST.get('postal_code', '').strip(),
            'country': request.POST.get('country', '').strip(),
            'is_default': bool(request.POST.get('is_default')),
        }
        AddressService.update_address(address, **data)
        messages.success(request, 'Address updated successfully.')
        return redirect('accounts:addresses')

    def get(self, request, pk):
        return redirect('accounts:addresses')


class DeleteAddressView(LoginRequiredMixin, View):
    """Soft delete a user's address."""
    login_url = '/account/login/'

    def post(self, request, pk):
        address = get_object_or_404(Address, pk=pk, user=request.user, is_deleted=False)
        AddressService.delete_address(address)
        messages.success(request, 'Address deleted successfully.')
        return redirect('accounts:addresses')

    def get(self, request, pk):
        return redirect('accounts:addresses')


class SetDefaultAddressView(LoginRequiredMixin, View):
    """Mark an address as the user's default."""
    login_url = '/account/login/'

    def post(self, request, pk):
        address = get_object_or_404(Address, pk=pk, user=request.user, is_deleted=False)
        address.is_default = True
        address.save(update_fields=['is_default', 'updated_at'])
        messages.success(request, 'Default address updated.')
        return redirect('accounts:addresses')

    def get(self, request, pk):
        return redirect('accounts:addresses')


class LoginView(FormView):
    """Login page."""
    template_name = 'accounts/login.html'
    form_class = LoginForm
    success_url = reverse_lazy('accounts:dashboard')

    def dispatch(self, request, *args, **kwargs):
        if request.method == "GET":
            safe_next = self._get_safe_next_url()
            frontend_login = _build_frontend_url("/account/login/", safe_next, request=request)
            if frontend_login:
                return redirect(frontend_login)
        if request.user.is_authenticated:
            redirect_url = self._get_safe_next_url()
            return redirect(redirect_url or self.get_success_url())
        return super().dispatch(request, *args, **kwargs)

    def _get_safe_next_url(self):
        """Return a safe `next` parameter if provided."""
        return _get_safe_next_url_for_request(self.request)

    def get_success_url(self):
        return self._get_safe_next_url() or super().get_success_url()

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['request'] = self.request
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Login'
        context['next'] = self.request.POST.get('next') or self.request.GET.get('next')
        return context

    def form_valid(self, form):
        user = form.get_user()
        login(self.request, user)

        remember = form.cleaned_data.get('remember')
        if remember:
            self.request.session.set_expiry(60 * 60 * 24 * 30)
        else:
            self.request.session.set_expiry(0)

        messages.success(self.request, 'Welcome back! You are now logged in.')
        return super().form_valid(form)

    def form_invalid(self, form):
        messages.error(self.request, 'Unable to log in with the provided credentials.')
        return super().form_invalid(form)


class RegisterView(FormView):
    """Registration page."""
    template_name = 'accounts/register.html'
    form_class = RegistrationForm
    success_url = reverse_lazy('accounts:dashboard')

    def dispatch(self, request, *args, **kwargs):
        if request.method == "GET":
            safe_next = _get_safe_next_url_for_request(request)
            frontend_register = _build_frontend_url("/account/register/", safe_next, request=request)
            if frontend_register:
                return redirect(frontend_register)
        if request.user.is_authenticated:
            return redirect('accounts:dashboard')
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Create Account'
        return context

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['request'] = self.request
        return kwargs

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        messages.success(self.request, 'Welcome aboard! Your account is ready.')
        return super().form_valid(form)

    def form_invalid(self, form):
        messages.error(self.request, 'We could not create your account. Fix the issues below and try again.')
        return super().form_invalid(form)


class LogoutView(View):
    """Logout handler."""
    
    def get(self, request):
        logout(request)
        return redirect('home')
    
    def post(self, request):
        logout(request)
        return redirect('home')


class VerifyEmailView(View):
    """Email verification handler."""
    
    def get(self, request, token):
        user = UserService.verify_email(token)
        if user:
            messages.success(request, 'Your email has been verified successfully!')
            if request.user.is_authenticated:
                return redirect('accounts:dashboard')
            return redirect('accounts:login')
        messages.error(request, 'Invalid or expired verification link.')
        return redirect('home')


class ForgotPasswordView(TemplateView):
    """Forgot password page."""
    template_name = 'accounts/forgot_password.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Forgot Password'
        return context
    
    def post(self, request, *args, **kwargs):
        """Handle password reset request form submission."""
        email = request.POST.get('email', '').strip()
        
        if not email:
            messages.error(request, 'Please enter your email address.')
            return self.get(request, *args, **kwargs)
        
        try:
            # Request password reset
            UserService.request_password_reset(email, request=request)
            messages.success(request, 'If an account exists with this email, a password reset link will be sent.')
            return redirect('accounts:login')
        except Exception as e:
            messages.error(request, 'An error occurred. Please try again.')
            return self.get(request, *args, **kwargs)


class ResetPasswordView(TemplateView):
    """Reset password page."""
    template_name = 'accounts/reset_password.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Reset Password'
        context['token'] = self.kwargs.get('token')
        return context
