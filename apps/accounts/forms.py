"""Account forms."""
from django import forms
from django.contrib.auth import authenticate, get_user_model
from django.utils.translation import gettext_lazy as _

User = get_user_model()


class LoginForm(forms.Form):
    """Simple email-based login form."""

    email = forms.EmailField(label=_('Email address'))
    password = forms.CharField(label=_('Password'), widget=forms.PasswordInput)
    remember = forms.BooleanField(label=_('Remember me'), required=False)

    error_messages = {
        'invalid_login': _('We could not match that email and password.'),
        'inactive': _('This account is inactive.'),
    }

    def __init__(self, request=None, *args, **kwargs):
        self.request = request
        self.user_cache = None
        super().__init__(*args, **kwargs)

    def clean(self):
        cleaned_data = super().clean()
        email = cleaned_data.get('email')
        password = cleaned_data.get('password')

        if email and password:
            self.user_cache = authenticate(self.request, email=email, password=password)
            if self.user_cache is None:
                raise forms.ValidationError(self.error_messages['invalid_login'], code='invalid_login')
            self.confirm_login_allowed(self.user_cache)

        return cleaned_data

    def confirm_login_allowed(self, user):
        """Hook for additional login checks."""
        if not user.is_active:
            raise forms.ValidationError(self.error_messages['inactive'], code='inactive')

    def get_user(self):
        return self.user_cache


class RegistrationForm(forms.Form):
    """Customer-facing account creation form."""

    first_name = forms.CharField(label=_('First name'), max_length=150)
    last_name = forms.CharField(label=_('Last name'), max_length=150)
    email = forms.EmailField(label=_('Email address'))
    password1 = forms.CharField(label=_('Password'), widget=forms.PasswordInput)
    password2 = forms.CharField(label=_('Confirm password'), widget=forms.PasswordInput)
    newsletter = forms.BooleanField(label=_('Subscribe to newsletter'), required=False)
    terms = forms.BooleanField(label=_('Agree to terms'), required=True, error_messages={
        'required': _('You must agree to continue.'),
    })

    def clean_email(self):
        email = self.cleaned_data['email'].lower()
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError(_('An account with this email already exists.'))
        return email

    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get('password1')
        password2 = cleaned_data.get('password2')

        if password1 and password2 and password1 != password2:
            self.add_error('password2', _('Passwords do not match.'))

        return cleaned_data

    def save(self):
        data = self.cleaned_data
        # Use UserService to create the user so verification email is sent
        from .services import UserService
        user = UserService.create_user(
            email=data['email'],
            password=data['password1'],
            first_name=data.get('first_name', '').strip(),
            last_name=data.get('last_name', '').strip(),
            newsletter_subscribed=data.get('newsletter', False)
        )
        return user
