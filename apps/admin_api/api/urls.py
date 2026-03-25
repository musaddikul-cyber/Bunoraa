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
    AdminAnalyticsCartEventViewSet,
    AdminAnalyticsCategoryStatViewSet,
    AdminAnalyticsDashboardViewSet,
    AdminAnalyticsDailyViewSet,
    AdminAnalyticsPageViewViewSet,
    AdminAnalyticsProductStatViewSet,
    AdminAnalyticsSearchQueryViewSet,
    AdminBannerViewSet,
    AdminBadgeViewSet,
    AdminBNPLAgreementViewSet,
    AdminBNPLProviderViewSet,
    AdminBundleViewSet,
    AdminCannedResponseViewSet,
    AdminCategoryFacetViewSet,
    AdminCategoryPricingProfileViewSet,
    AdminCategoryViewSet,
    AdminChatAgentViewSet,
    AdminChatAnalyticsViewSet,
    AdminChatSettingsViewSet,
    AdminCollectionViewSet,
    AdminContactMessageViewSet,
    AdminConversationViewSet,
    AdminCouponUsageViewSet,
    AdminCouponViewSet,
    AdminCurrencyViewSet,
    AdminCustomerPhotoViewSet,
    AdminEcoCertificationViewSet,
    AdminEmailLogViewSet,
    AdminExchangeRateViewSet,
    AdminFAQViewSet,
    AdminFacetViewSet,
    AdminMessageAttachmentViewSet,
    AdminMessageViewSet,
    AdminNewsletterIncentiveViewSet,
    AdminNotificationDeliveryViewSet,
    AdminNotificationPreferenceViewSet,
    AdminNotificationTemplateViewSet,
    AdminNotificationViewSet,
    AdminPageViewSet,
    AdminPaymentAuditLogViewSet,
    AdminPaymentGatewayViewSet,
    AdminPaymentLinkViewSet,
    AdminPaymentMethodViewSet,
    AdminPaymentTransactionViewSet,
    AdminPaymentViewSet,
    AdminPlanViewSet,
    AdminProductAnswerViewSet,
    AdminProductPriceViewSet,
    AdminProductQuestionViewSet,
    AdminProductVariantViewSet,
    AdminProductViewSet,
    AdminPushTokenViewSet,
    AdminRealtimeEventsView,
    AdminRefundViewSet,
    AdminRecurringChargeViewSet,
    AdminReviewImageViewSet,
    AdminReviewReportViewSet,
    AdminReviewVoteViewSet,
    AdminReviewViewSet,
    AdminSaleViewSet,
    AdminShipmentEventViewSet,
    AdminShipmentViewSet,
    AdminShippingCarrierViewSet,
    AdminShippingMaterialViewSet,
    AdminShippingMethodViewSet,
    AdminShippingRateViewSet,
    AdminShippingRestrictionViewSet,
    AdminShippingSettingsView,
    AdminShippingZoneViewSet,
    AdminSiteSettingsView,
    AdminSocialLinkViewSet,
    AdminSpotlightViewSet,
    AdminSubscriberViewSet,
    AdminSubscriptionViewSet,
    AdminSystemHealthDetailsView,
    AdminTagViewSet,
    AdminTypingIndicatorViewSet,
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
router.register(r"catalog/variants", AdminProductVariantViewSet, basename="admin-catalog-variants")
router.register(r"catalog/collections", AdminCollectionViewSet, basename="admin-catalog-collections")
router.register(r"catalog/bundles", AdminBundleViewSet, basename="admin-catalog-bundles")
router.register(r"catalog/tags", AdminTagViewSet, basename="admin-catalog-tags")
router.register(r"catalog/badges", AdminBadgeViewSet, basename="admin-catalog-badges")
router.register(r"catalog/spotlights", AdminSpotlightViewSet, basename="admin-catalog-spotlights")
router.register(r"catalog/customer-photos", AdminCustomerPhotoViewSet, basename="admin-catalog-customer-photos")
router.register(r"catalog/questions", AdminProductQuestionViewSet, basename="admin-catalog-questions")
router.register(r"catalog/answers", AdminProductAnswerViewSet, basename="admin-catalog-answers")
router.register(r"catalog/shipping-materials", AdminShippingMaterialViewSet, basename="admin-catalog-shipping-materials")
router.register(r"catalog/facets", AdminFacetViewSet, basename="admin-catalog-facets")
router.register(r"catalog/category-facets", AdminCategoryFacetViewSet, basename="admin-catalog-category-facets")
router.register(r"catalog/eco-certifications", AdminEcoCertificationViewSet, basename="admin-catalog-eco-certifications")

# Pricing
router.register(r"pricing/currencies", AdminCurrencyViewSet, basename="admin-pricing-currencies")
router.register(r"pricing/exchange-rates", AdminExchangeRateViewSet, basename="admin-pricing-exchange-rates")
router.register(r"pricing/product-prices", AdminProductPriceViewSet, basename="admin-pricing-product-prices")
router.register(r"pricing/category-profiles", AdminCategoryPricingProfileViewSet, basename="admin-pricing-category-profiles")

# Promotions
router.register(r"promotions/coupons", AdminCouponViewSet, basename="admin-promotions-coupons")
router.register(r"promotions/coupon-usage", AdminCouponUsageViewSet, basename="admin-promotions-coupon-usage")
router.register(r"promotions/banners", AdminBannerViewSet, basename="admin-promotions-banners")
router.register(r"promotions/sales", AdminSaleViewSet, basename="admin-promotions-sales")

# CMS
router.register(r"cms/pages", AdminPageViewSet, basename="admin-cms-pages")
router.register(r"cms/faqs", AdminFAQViewSet, basename="admin-cms-faqs")
router.register(r"cms/contact-messages", AdminContactMessageViewSet, basename="admin-cms-contact-messages")
router.register(r"cms/subscribers", AdminSubscriberViewSet, basename="admin-cms-subscribers")
router.register(r"cms/social-links", AdminSocialLinkViewSet, basename="admin-cms-social-links")
router.register(r"cms/newsletter-incentives", AdminNewsletterIncentiveViewSet, basename="admin-cms-newsletter-incentives")

# Reviews
router.register(r"reviews/images", AdminReviewImageViewSet, basename="admin-reviews-images")
router.register(r"reviews/votes", AdminReviewVoteViewSet, basename="admin-reviews-votes")
router.register(r"reviews/reports", AdminReviewReportViewSet, basename="admin-reviews-reports")
router.register(r"reviews", AdminReviewViewSet, basename="admin-reviews")

# Shipping
router.register(r"shipping/zones", AdminShippingZoneViewSet, basename="admin-shipping-zones")
router.register(r"shipping/carriers", AdminShippingCarrierViewSet, basename="admin-shipping-carriers")
router.register(r"shipping/methods", AdminShippingMethodViewSet, basename="admin-shipping-methods")
router.register(r"shipping/rates", AdminShippingRateViewSet, basename="admin-shipping-rates")
router.register(r"shipping/restrictions", AdminShippingRestrictionViewSet, basename="admin-shipping-restrictions")
router.register(r"shipping/shipments", AdminShipmentViewSet, basename="admin-shipping-shipments")
router.register(r"shipping/shipment-events", AdminShipmentEventViewSet, basename="admin-shipping-shipment-events")

# Payments
router.register(r"payments/gateways", AdminPaymentGatewayViewSet, basename="admin-payments-gateways")
router.register(r"payments/payments", AdminPaymentViewSet, basename="admin-payments-payments")
router.register(r"payments/methods", AdminPaymentMethodViewSet, basename="admin-payments-methods")
router.register(r"payments/transactions", AdminPaymentTransactionViewSet, basename="admin-payments-transactions")
router.register(r"payments/links", AdminPaymentLinkViewSet, basename="admin-payments-links")
router.register(r"payments/bnpl-providers", AdminBNPLProviderViewSet, basename="admin-payments-bnpl")
router.register(r"payments/bnpl-agreements", AdminBNPLAgreementViewSet, basename="admin-payments-bnpl-agreements")
router.register(r"payments/audit-logs", AdminPaymentAuditLogViewSet, basename="admin-payments-audit-logs")
router.register(r"payments/refunds", AdminRefundViewSet, basename="admin-payments-refunds")
router.register(r"payments/recurring", AdminRecurringChargeViewSet, basename="admin-payments-recurring")

# Subscriptions
router.register(r"subscriptions/plans", AdminPlanViewSet, basename="admin-subscription-plans")
router.register(r"subscriptions/subscriptions", AdminSubscriptionViewSet, basename="admin-subscriptions")

# Notifications
router.register(r"notifications/deliveries", AdminNotificationDeliveryViewSet, basename="admin-notification-deliveries")
router.register(r"notifications/templates", AdminNotificationTemplateViewSet, basename="admin-notification-templates")
router.register(r"notifications/preferences", AdminNotificationPreferenceViewSet, basename="admin-notification-preferences")
router.register(r"notifications/push-tokens", AdminPushTokenViewSet, basename="admin-notification-push-tokens")
router.register(r"notifications/email-logs", AdminEmailLogViewSet, basename="admin-notification-email-logs")
router.register(r"notifications", AdminNotificationViewSet, basename="admin-notifications")

# Analytics
router.register(r"analytics/dashboard", AdminAnalyticsDashboardViewSet, basename="admin-analytics-dashboard")
router.register(r"analytics/daily", AdminAnalyticsDailyViewSet, basename="admin-analytics-daily")
router.register(r"analytics/product-stats", AdminAnalyticsProductStatViewSet, basename="admin-analytics-product-stats")
router.register(r"analytics/category-stats", AdminAnalyticsCategoryStatViewSet, basename="admin-analytics-category-stats")
router.register(r"analytics/search-queries", AdminAnalyticsSearchQueryViewSet, basename="admin-analytics-search-queries")
router.register(r"analytics/cart-events", AdminAnalyticsCartEventViewSet, basename="admin-analytics-cart-events")
router.register(r"analytics/page-views", AdminAnalyticsPageViewViewSet, basename="admin-analytics-page-views")

# Support / chat
router.register(r"support/agents", AdminChatAgentViewSet, basename="admin-chat-agents")
router.register(r"support/conversations", AdminConversationViewSet, basename="admin-chat-conversations")
router.register(r"support/messages", AdminMessageViewSet, basename="admin-chat-messages")
router.register(r"support/message-attachments", AdminMessageAttachmentViewSet, basename="admin-chat-message-attachments")
router.register(r"support/typing", AdminTypingIndicatorViewSet, basename="admin-chat-typing")
router.register(r"support/canned-responses", AdminCannedResponseViewSet, basename="admin-chat-canned-responses")
router.register(r"support/settings", AdminChatSettingsViewSet, basename="admin-chat-settings")
router.register(r"support/analytics", AdminChatAnalyticsViewSet, basename="admin-chat-analytics")

urlpatterns = [
    path("auth/social/", AdminSocialLoginView.as_view(), name="admin-social-login"),
    path("dashboard/", AdminDashboardView.as_view(), name="admin-dashboard"),
    path("health/", AdminHealthView.as_view(), name="admin-health"),
    path("health/details/", AdminSystemHealthDetailsView.as_view(), name="admin-health-details"),
    path("realtime/events/", AdminRealtimeEventsView.as_view(), name="admin-realtime-events"),
    path("cms/site-settings/", AdminSiteSettingsView.as_view(), name="admin-site-settings"),
    path("shipping/settings/", AdminShippingSettingsView.as_view(), name="admin-shipping-settings"),
    path("", include(router.urls)),
]
