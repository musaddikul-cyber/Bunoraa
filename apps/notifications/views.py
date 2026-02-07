"""
Frontend views for notifications.
"""
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView


class NotificationListView(LoginRequiredMixin, TemplateView):
    """List a user's notifications (placeholder)."""
    template_name = 'notifications/list.html'
    login_url = '/account/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Notifications'
        return context
