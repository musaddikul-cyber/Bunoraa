"""
Admin configuration for the artisans app.
"""
from django.contrib import admin
from .models import Artisan

@admin.register(Artisan)
class ArtisanAdmin(admin.ModelAdmin):
    """
    Admin interface for the Artisan model.
    """
    list_display = ('name', 'is_active', 'created_at')
    list_filter = ('is_active',)
    search_fields = ('name', 'bio')
    prepopulated_fields = {'slug': ('name',)}
    
    fieldsets = (
        (None, {
            'fields': ('name', 'slug', 'bio', 'photo')
        }),
        ('Social Media', {
            'fields': ('website', 'instagram', 'facebook'),
            'classes': ('collapse',)
        }),
        ('Status', {
            'fields': ('is_active',)
        }),
    )