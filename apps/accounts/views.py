"""
Account views - Frontend pages
"""
from django.shortcuts import redirect, get_object_or_404
from django.views.generic import TemplateView, View
from django.views.generic.edit import FormView
from django.contrib.auth import login, logout, update_session_auth_hash
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.forms import PasswordChangeForm
from django.http import HttpResponseForbidden
from django.contrib import messages
from django.urls import reverse_lazy
from django.conf import settings
from django.utils.http import url_has_allowed_host_and_scheme
from django.core.exceptions import ValidationError
from .services import UserService, AddressService
from .models import Address
from apps.i18n.services import GeoService as CountryService
from .forms import LoginForm, RegistrationForm


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
        if request.user.is_authenticated:
            redirect_url = self._get_safe_next_url()
            return redirect(redirect_url or self.get_success_url())
        return super().dispatch(request, *args, **kwargs)

    def _get_safe_next_url(self):
        """Return a safe `next` parameter if provided."""
        next_url = self.request.POST.get('next') or self.request.GET.get('next')
        if not next_url:
            return None

        allowed_hosts = {self.request.get_host()}
        allowed_hosts.update(getattr(settings, 'ALLOWED_HOSTS', []))

        if url_has_allowed_host_and_scheme(next_url, allowed_hosts=allowed_hosts, require_https=self.request.is_secure()):
            return next_url
        return None

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
        if request.user.is_authenticated:
            return redirect('accounts:dashboard')
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Create Account'
        return context

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
            UserService.request_password_reset(email)
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
