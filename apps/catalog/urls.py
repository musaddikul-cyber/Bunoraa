"""
Catalog URL Configuration - Web views
"""
from django.urls import path, include
from . import views

app_name = 'catalog'

urlpatterns = [
    # Homepage
    path('', views.HomepageView.as_view(), name='homepage'),
    
    # Categories
    path('categories/', views.CategoryListView.as_view(), name='category-list'),
    path('category/<path:slug>/', views.CategoryDetailView.as_view(), name='category-detail'),
    
    # Products
    path('products/', views.ProductListView.as_view(), name='product-list'),
    path('products/<slug:slug>/', views.ProductDetailView.as_view(), name='product-detail'),
    path('products/<slug:slug>/quick-view/', views.ProductQuickView.as_view(), name='product-quick-view'),
    path('products/<slug:slug>/reviews/', views.ProductReviewsView.as_view(), name='product-reviews'),
    path('products/<slug:slug>/review/', views.CreateReviewView.as_view(), name='create-review'),
    
    # Collections
    path('collections/', views.CollectionListView.as_view(), name='collection-list'),
    path('collections/<slug:slug>/', views.CollectionDetailView.as_view(), name='collection-detail'),
    
    # Bundles
    path('bundles/', views.BundleListView.as_view(), name='bundle-list'),
    path('bundles/<slug:slug>/', views.BundleDetailView.as_view(), name='bundle-detail'),
    
    # Search
    path('search/', views.SearchView.as_view(), name='search'),
    
    # API routes
    path('api/', include('apps.catalog.api.urls')),
]
