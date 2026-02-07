"""
Views for the artisans app.
"""
from django.views.generic import ListView, DetailView
from .models import Artisan

class ArtisanListView(ListView):
    """
    View to display a list of all active artisans.
    """
    model = Artisan
    context_object_name = 'artisans'
    template_name = 'artisans/artisan_list.html'

    def get_queryset(self):
        return Artisan.objects.filter(is_active=True)

class ArtisanDetailView(DetailView):
    """
    View to display the profile of a single artisan.
    """
    model = Artisan
    context_object_name = 'artisan'
    template_name = 'artisans/artisan_detail.html'

    def get_queryset(self):
        return Artisan.objects.filter(is_active=True)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        artisan = self.get_object()
        context['products'] = artisan.products.filter(is_active=True, is_deleted=False).prefetch_for_list()[:12]
        return context