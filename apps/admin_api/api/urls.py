from django.urls import include, path

from core.api.routers import DefaultRouter

from .views import (
    AdminAuditLogViewSet,
    AdminDashboardView,
    AdminGroupViewSet,
    AdminHealthView,
    AdminOrderViewSet,
    AdminPermissionViewSet,
    AdminSocialLoginView,
    AdminUserViewSet,
)
from .enterprise import (
    AdminAnalyticsDashboardViewSet,
    AdminAnalyticsDailyViewSet,
    AdminBannerViewSet,
    AdminBadgeViewSet,
    AdminBundleViewSet,
    AdminCannedResponseViewSet,
    AdminCategoryViewSet,
    AdminChatAgentViewSet,
    AdminChatAnalyticsViewSet,
    AdminChatSettingsViewSet,
    AdminCollectionViewSet,
    AdminContactMessageViewSet,
    AdminConversationViewSet,
    AdminCouponViewSet,
    AdminCurrencyViewSet,
    AdminCustomerPhotoViewSet,
    AdminExchangeRateViewSet,
    AdminFAQViewSet,
    AdminMessageViewSet,
    AdminNotificationDeliveryViewSet,
    AdminNotificationTemplateViewSet,
    AdminNotificationViewSet,
    AdminPageViewSet,
    AdminPaymentGatewayViewSet,
    AdminPaymentLinkViewSet,
    AdminPaymentViewSet,
    AdminPlanViewSet,
    AdminProductAnswerViewSet,
    AdminProductQuestionViewSet,
    AdminProductViewSet,
    AdminRefundViewSet,
    AdminRecurringChargeViewSet,
    AdminReviewViewSet,
    AdminSaleViewSet,
    AdminShippingCarrierViewSet,
    AdminShippingMethodViewSet,
    AdminShippingRateViewSet,
    AdminShippingZoneViewSet,
    AdminShipmentViewSet,
    AdminSpotlightViewSet,
    AdminSubscriberViewSet,
    AdminSubscriptionViewSet,
    AdminTagViewSet,
    AdminBNPLProviderViewSet,
    AdminSiteSettingsView,
    AdminShippingSettingsView,
    AdminSystemHealthDetailsView,
)

router = DefaultRouter()
router.register(r"audit-logs", AdminAuditLogViewSet, basename="admin-audit-logs")
router.register(r"users", AdminUserViewSet, basename="admin-users")
router.register(r"groups", AdminGroupViewSet, basename="admin-groups")
router.register(r"permissions", AdminPermissionViewSet, basename="admin-permissions")
router.register(r"orders", AdminOrderViewSet, basename="admin-orders")

# Catalog
router.register(r"catalog/categories", AdminCategoryViewSet, basename="admin-catalog-categories")
router.register(r"catalog/products", AdminProductViewSet, basename="admin-catalog-products")
router.register(r"catalog/collections", AdminCollectionViewSet, basename="admin-catalog-collections")
router.register(r"catalog/bundles", AdminBundleViewSet, basename="admin-catalog-bundles")
router.register(r"catalog/tags", AdminTagViewSet, basename="admin-catalog-tags")
router.register(r"catalog/badges", AdminBadgeViewSet, basename="admin-catalog-badges")
router.register(r"catalog/spotlights", AdminSpotlightViewSet, basename="admin-catalog-spotlights")
router.register(r"catalog/customer-photos", AdminCustomerPhotoViewSet, basename="admin-catalog-customer-photos")
router.register(r"catalog/questions", AdminProductQuestionViewSet, basename="admin-catalog-questions")
router.register(r"catalog/answers", AdminProductAnswerViewSet, basename="admin-catalog-answers")

# Pricing
router.register(r"pricing/currencies", AdminCurrencyViewSet, basename="admin-pricing-currencies")
router.register(r"pricing/exchange-rates", AdminExchangeRateViewSet, basename="admin-pricing-exchange-rates")

# Promotions
router.register(r"promotions/coupons", AdminCouponViewSet, basename="admin-promotions-coupons")
router.register(r"promotions/banners", AdminBannerViewSet, basename="admin-promotions-banners")
router.register(r"promotions/sales", AdminSaleViewSet, basename="admin-promotions-sales")

# CMS
router.register(r"cms/pages", AdminPageViewSet, basename="admin-cms-pages")
router.register(r"cms/faqs", AdminFAQViewSet, basename="admin-cms-faqs")
router.register(r"cms/contact-messages", AdminContactMessageViewSet, basename="admin-cms-contact-messages")
router.register(r"cms/subscribers", AdminSubscriberViewSet, basename="admin-cms-subscribers")

# Reviews
router.register(r"reviews", AdminReviewViewSet, basename="admin-reviews")

# Shipping
router.register(r"shipping/zones", AdminShippingZoneViewSet, basename="admin-shipping-zones")
router.register(r"shipping/carriers", AdminShippingCarrierViewSet, basename="admin-shipping-carriers")
router.register(r"shipping/methods", AdminShippingMethodViewSet, basename="admin-shipping-methods")
router.register(r"shipping/rates", AdminShippingRateViewSet, basename="admin-shipping-rates")
router.register(r"shipping/shipments", AdminShipmentViewSet, basename="admin-shipping-shipments")

# Payments
router.register(r"payments/gateways", AdminPaymentGatewayViewSet, basename="admin-payments-gateways")
router.register(r"payments/payments", AdminPaymentViewSet, basename="admin-payments-payments")
router.register(r"payments/links", AdminPaymentLinkViewSet, basename="admin-payments-links")
router.register(r"payments/bnpl-providers", AdminBNPLProviderViewSet, basename="admin-payments-bnpl")
router.register(r"payments/refunds", AdminRefundViewSet, basename="admin-payments-refunds")
router.register(r"payments/recurring", AdminRecurringChargeViewSet, basename="admin-payments-recurring")

# Subscriptions
router.register(r"subscriptions/plans", AdminPlanViewSet, basename="admin-subscription-plans")
router.register(r"subscriptions/subscriptions", AdminSubscriptionViewSet, basename="admin-subscriptions")

# Notifications
router.register(r"notifications", AdminNotificationViewSet, basename="admin-notifications")
router.register(r"notifications/deliveries", AdminNotificationDeliveryViewSet, basename="admin-notification-deliveries")
router.register(r"notifications/templates", AdminNotificationTemplateViewSet, basename="admin-notification-templates")

# Analytics
router.register(r"analytics/dashboard", AdminAnalyticsDashboardViewSet, basename="admin-analytics-dashboard")
router.register(r"analytics/daily", AdminAnalyticsDailyViewSet, basename="admin-analytics-daily")

# Support / chat
router.register(r"support/agents", AdminChatAgentViewSet, basename="admin-chat-agents")
router.register(r"support/conversations", AdminConversationViewSet, basename="admin-chat-conversations")
router.register(r"support/messages", AdminMessageViewSet, basename="admin-chat-messages")
router.register(r"support/canned-responses", AdminCannedResponseViewSet, basename="admin-chat-canned-responses")
router.register(r"support/settings", AdminChatSettingsViewSet, basename="admin-chat-settings")
router.register(r"support/analytics", AdminChatAnalyticsViewSet, basename="admin-chat-analytics")

urlpatterns = [
    path("auth/social/", AdminSocialLoginView.as_view(), name="admin-social-login"),
    path("dashboard/", AdminDashboardView.as_view(), name="admin-dashboard"),
    path("health/", AdminHealthView.as_view(), name="admin-health"),
    path("health/details/", AdminSystemHealthDetailsView.as_view(), name="admin-health-details"),
    path("cms/site-settings/", AdminSiteSettingsView.as_view(), name="admin-site-settings"),
    path("shipping/settings/", AdminShippingSettingsView.as_view(), name="admin-shipping-settings"),
    path("", include(router.urls)),
]
