"""
Models for the artisans app.
"""
import uuid
from django.db import models
from django.urls import reverse
from django.utils.text import slugify

class Artisan(models.Model):
    """
    Represents a creator of hand-embroidered products.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    bio = models.TextField(blank=True)
    photo = models.ImageField(upload_to='artisans/photos/', blank=True, null=True)
    
    # Social links
    website = models.URLField(blank=True)
    instagram = models.URLField(blank=True)
    facebook = models.URLField(blank=True)
    
    # Status
    is_active = models.BooleanField(default=True, db_index=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        verbose_name = 'Artisan'
        verbose_name_plural = 'Artisans'

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('artisans:artisan_detail', kwargs={'slug': self.slug})