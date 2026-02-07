"""
Account URL configuration - Frontend views
"""
from django.urls import path
from django.views.generic import RedirectView
from . import views

app_name = 'accounts'

urlpatterns = [
    path('', RedirectView.as_view(pattern_name='accounts:login', permanent=False)),
    path('dashboard/', views.AccountDashboardView.as_view(), name='dashboard'),
    path('profile/', views.ProfileView.as_view(), name='profile'),
    path('addresses/', views.AddressListView.as_view(), name='addresses'),
    path('addresses/add/', views.AddAddressView.as_view(), name='add_address'),
    path('addresses/<uuid:pk>/edit/', views.EditAddressView.as_view(), name='edit_address'),
    path('addresses/<uuid:pk>/delete/', views.DeleteAddressView.as_view(), name='delete_address'),
    path('addresses/<uuid:pk>/set-default/', views.SetDefaultAddressView.as_view(), name='set_default_address'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('register/', views.RegisterView.as_view(), name='register'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
    path('verify-email/<str:token>/', views.VerifyEmailView.as_view(), name='verify_email'),
    path('forgot-password/', views.ForgotPasswordView.as_view(), name='forgot_password'),
    path('password-reset/', views.ForgotPasswordView.as_view(), name='password_reset'),  # Alias
    path('reset-password/<str:token>/', views.ResetPasswordView.as_view(), name='reset_password'),
    path('change-password/', views.ChangePasswordView.as_view(), name='change_password'),
    path('delete-account/', views.DeleteAccountView.as_view(), name='delete_account'),
    
    # Placeholder URLs - implement views as needed
    # path('profile/change-password/', views.ChangePasswordView.as_view(), name='change_password'),
    # path('profile/delete-account/', views.DeleteAccountView.as_view(), name='delete_account'),
    # path('notifications/', views.NotificationsView.as_view(), name='notifications'),
]
