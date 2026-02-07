"""
Pages views
"""
from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import TemplateView, ListView, DetailView
from django.contrib import messages
from django.http import JsonResponse

from .models import Page, FAQ, ContactMessage, SiteSettings, Subscriber


class HomeView(TemplateView):
    """Homepage view."""
    template_name = 'pages/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Home'
        
        # Get featured products
        from apps.products.services import ProductService
        context['featured_products'] = ProductService.get_featured_products()[:8]
        context['new_arrivals'] = ProductService.get_new_arrivals()[:8]
        
        # Get banners
        from apps.promotions.services import BannerService
        context['hero_banners'] = BannerService.get_hero_banners()
        context['secondary_banners'] = BannerService.get_secondary_banners()
        
        # Get featured categories
        from apps.categories.services import CategoryService
        context['featured_categories'] = CategoryService.get_featured_categories()[:6]
        
        return context


class PageDetailView(DetailView):
    """Static page view."""
    model = Page
    template_name = 'pages/page.html'
    context_object_name = 'page'
    
    def get_queryset(self):
        return Page.objects.filter(is_published=True)
    
    def get_template_names(self):
        # Use specific template if available
        template_map = {
            Page.TEMPLATE_LANDING: 'pages/home.html',
            Page.TEMPLATE_CONTACT: 'pages/contact.html',
            Page.TEMPLATE_ABOUT: 'pages/about.html',
            Page.TEMPLATE_FAQ: 'pages/faq.html',
        }
        return [template_map.get(self.object.template, self.template_name)]
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = self.object.meta_title or self.object.title
        
        # For FAQ page, include FAQs
        if self.object.template == Page.TEMPLATE_FAQ:
            context['faqs'] = FAQ.objects.filter(is_active=True)
        
        return context


class ContactView(TemplateView):
    """Contact page view."""
    template_name = 'pages/contact.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Contact Us'
        context['settings'] = SiteSettings.get_settings()
        return context
    
    def post(self, request, *args, **kwargs):
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip()
        phone = request.POST.get('phone', '').strip()
        subject = request.POST.get('subject', '').strip()
        message = request.POST.get('message', '').strip()
        
        # Validation
        errors = []
        if not name:
            errors.append('Name is required')
        if not email:
            errors.append('Email is required')
        if not subject:
            errors.append('Subject is required')
        if not message:
            errors.append('Message is required')
        
        if errors:
            for error in errors:
                messages.error(request, error)
            return self.get(request, *args, **kwargs)
        
        ContactMessage.objects.create(
            name=name,
            email=email,
            phone=phone,
            subject=subject,
            message=message
        )
        
        messages.success(request, 'Your message has been sent. We will get back to you soon.')
        return redirect('pages:contact')


class FAQListView(ListView):
    """FAQ page view."""
    model = FAQ
    template_name = 'pages/faq.html'
    context_object_name = 'faqs'
    
    def get_queryset(self):
        return FAQ.objects.filter(is_active=True).order_by('sort_order')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Frequently Asked Questions'
        
        # Group by category
        categories = {}
        for faq in context['faqs']:
            cat = faq.category or 'General'
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(faq)
        context['faq_categories'] = categories
        
        return context


class AboutView(TemplateView):
    """About page view."""
    template_name = 'pages/about.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'About Us'
        context['settings'] = SiteSettings.get_settings()
        return context


def subscribe_newsletter(request):
    """Newsletter subscription endpoint."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'Method not allowed'}, status=405)
    
    email = request.POST.get('email', '').strip()
    name = request.POST.get('name', '').strip()
    
    if not email:
        return JsonResponse({'success': False, 'message': 'Email is required'})
    
    # Check if already subscribed
    existing = Subscriber.objects.filter(email=email).first()
    if existing:
        if existing.is_active:
            return JsonResponse({'success': False, 'message': 'You are already subscribed'})
        else:
            # Reactivate
            existing.is_active = True
            existing.unsubscribed_at = None
            existing.save()
            return JsonResponse({'success': True, 'message': 'Welcome back! You have been re-subscribed.'})
    
    Subscriber.objects.create(
        email=email,
        name=name,
        source='website'
    )
    
    return JsonResponse({'success': True, 'message': 'Thank you for subscribing!'})
