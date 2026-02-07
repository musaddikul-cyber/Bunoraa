"""
Commerce URL Configuration
"""
from django.urls import path, include

from . import views

app_name = 'commerce'

# Cart URLs
cart_patterns = [
    path('', views.CartView.as_view(), name='cart'),
    path('shared/<str:token>/', views.SharedCartView.as_view(), name='shared_cart'),
    path('add/', views.CartAddView.as_view(), name='cart_add'),
    path('update/', views.CartUpdateView.as_view(), name='cart_update'),
    path('remove/', views.CartRemoveView.as_view(), name='cart_remove'),
    path('clear/', views.CartClearView.as_view(), name='cart_clear'),
    path('coupon/apply/', views.CartApplyCouponView.as_view(), name='apply_coupon'),
    path('coupon/remove/', views.CartRemoveCouponView.as_view(), name='cart_remove_coupon'),
    path('count/', views.CartCountView.as_view(), name='cart_count'),
]

# Wishlist URLs
wishlist_patterns = [
    path('', views.WishlistView.as_view(), name='wishlist'),
    path('add/', views.WishlistAddView.as_view(), name='wishlist_add'),
    path('remove/', views.WishlistRemoveView.as_view(), name='wishlist_remove'),
    path('move-to-cart/', views.WishlistMoveToCartView.as_view(), name='wishlist_move_to_cart'),
    path('share/', views.WishlistShareView.as_view(), name='wishlist_share'),
    path('shared/<str:token>/', views.SharedWishlistView.as_view(), name='shared_wishlist'),
    path('count/', views.WishlistCountView.as_view(), name='wishlist_count'),
    path('check/', views.CheckWishlistView.as_view(), name='wishlist_check'),
    path('toggle/', views.ToggleWishlistView.as_view(), name='wishlist_toggle'),
]

# Checkout URLs
checkout_patterns = [
    path('', views.CheckoutView.as_view(), name='checkout'),
    path('info/', views.CheckoutUpdateInfoView.as_view(), name='checkout_info'),
    path('shipping/', views.CheckoutSelectShippingView.as_view(), name='checkout_shipping'),
    path('payment/', views.CheckoutSelectPaymentView.as_view(), name='checkout_payment'),
    path('review/', views.CheckoutReviewView.as_view(), name='checkout_review'),
    path('complete/', views.CheckoutCompleteView.as_view(), name='checkout_complete'),
    path('confirmation/<str:order_number>/', views.OrderConfirmationView.as_view(), name='order_confirmation'),
]

urlpatterns = [
    path('cart/', include(cart_patterns)),
    path('wishlist/', include(wishlist_patterns)),
    path('checkout/', include(checkout_patterns)),
    
    # API routes
    path('api/', include('apps.commerce.api.urls')),
]
