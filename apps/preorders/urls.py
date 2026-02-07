"""
Pre-orders URLs - URL patterns for custom pre-order system
"""
from django.urls import path
from . import views

app_name = 'preorders'

urlpatterns = [
    # Landing and categories
    path('', views.PreOrderLandingView.as_view(), name='landing'),
    path('categories/', views.PreOrderCategoryListView.as_view(), name='categories'),
    path('category/<slug:slug>/', views.PreOrderCategoryDetailView.as_view(), name='category_detail'),
    
    # Pre-order wizard
    path('create/', views.PreOrderCreateWizardView.as_view(), name='wizard'),
    path('create/<int:step>/', views.PreOrderCreateWizardView.as_view(), name='wizard'),
    
    # Success page
    path('success/<str:preorder_number>/', views.PreOrderSuccessView.as_view(), name='success'),
    
    # User pre-orders
    path('my-orders/', views.MyPreOrdersView.as_view(), name='my_preorders'),
    path('order/<str:preorder_number>/', views.PreOrderDetailView.as_view(), name='detail'),
    
    # Tracking
    path('track/', views.PreOrderTrackingView.as_view(), name='tracking'),
    
    # Pre-order actions
    path('order/<str:preorder_number>/message/', views.PreOrderMessageView.as_view(), name='send_message'),
    path('order/<str:preorder_number>/revision/', views.PreOrderRevisionRequestView.as_view(), name='request_revision'),
    path('order/<str:preorder_number>/upload-design/', views.PreOrderDesignUploadView.as_view(), name='upload_design'),
    path('order/<str:preorder_number>/approve/', views.PreOrderApproveView.as_view(), name='approve'),
    path('order/<str:preorder_number>/quote/<uuid:quote_id>/respond/', views.QuoteResponseView.as_view(), name='quote_response'),
    path('order/<str:preorder_number>/mark-read/', views.MarkMessagesReadView.as_view(), name='mark_read'),
    
    # API endpoints
    path('api/category/<uuid:category_id>/options/', views.PreOrderCategoryOptionsAPIView.as_view(), name='api_category_options'),
    path('api/calculate-price/', views.PreOrderPriceCalculatorAPIView.as_view(), name='api_calculate_price'),
    path('api/order/<str:preorder_number>/status/', views.PreOrderStatusAPIView.as_view(), name='api_order_status'),
    path('api/template/<uuid:template_id>/use/', views.PreOrderTemplateUseView.as_view(), name='api_use_template'),
]
