"""
Core views
"""
from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import JsonResponse
from django.db import connection


class HomeView(TemplateView):
    """Home page view."""
    template_name = 'home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Welcome to Bunoraa'
        context['meta_description'] = 'Discover premium products at Bunoraa. Shop our curated collection of high-quality items.'
        return context


def health_check(request):
    """Health check endpoint for load balancers and monitoring."""
    try:
        # Check database connection
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
        db_status = 'healthy'
    except Exception:
        db_status = 'unhealthy'
    
    return JsonResponse({
        'status': 'healthy' if db_status == 'healthy' else 'unhealthy',
        'database': db_status,
    })
