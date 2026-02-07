"""
Frontend views for payments.
"""
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin


class PaymentMethodsView(LoginRequiredMixin, TemplateView):
    """Payment methods management page."""
    template_name = 'payments/payment_methods.html'
    login_url = '/account/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Payment Methods'
        return context


class GatewayIPNView(TemplateView):
    """Basic handler that can be extended to render a response for gateway callbacks.

    Gateways like SSLCommerz use server-to-server callbacks (IPN). See API views for
    JSON webhook handlers used by frontend APIs.
    """
    template_name = 'payments/ipn_ack.html'

    def post(self, request, *args, **kwargs):
        # Acknowledge immediately; real processing should be done by API endpoint
        return self.render_to_response({'status': 'ok'})