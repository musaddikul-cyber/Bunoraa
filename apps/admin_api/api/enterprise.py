from __future__ import annotations

from django.utils import timezone
from rest_framework import serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.admin_api.permissions import IsStaffWithMfa

# Catalog
from apps.catalog.models import (
    Badge,
    Bundle,
    Category,
    Collection,
    CustomerPhoto,
    Product,
    ProductAnswer,
    ProductQuestion,
    Spotlight,
    Tag,
)
from apps.catalog.api.serializers import (
    BadgeSerializer,
    BundleDetailSerializer,
    BundleListSerializer,
    CategoryListSerializer,
    CategorySerializer,
    CollectionDetailSerializer,
    CollectionListSerializer,
    CustomerPhotoSerializer,
    ProductAnswerSerializer,
    ProductDetailSerializer,
    ProductListSerializer,
    ProductQuestionSerializer,
    SpotlightSerializer,
    TagSerializer,
)

# Pricing (currency/exchange rates)
from apps.i18n.models import Currency, ExchangeRate
from apps.i18n.api.serializers import CurrencySerializer, ExchangeRateSerializer

# Promotions
from apps.promotions.models import Banner, Coupon, Sale
from apps.promotions.api.serializers import BannerSerializer, CouponSerializer, SaleSerializer

# CMS
from apps.pages.models import ContactMessage, FAQ, Page, SiteSettings, Subscriber
from apps.pages.api.serializers import (
    ContactMessageSerializer,
    FAQSerializer,
    PageDetailSerializer,
    PageListSerializer,
    SiteSettingsSerializer,
)
from apps.pages.services import ContactService

# Reviews
from apps.reviews.models import Review
from apps.reviews.api.serializers import (
    FeatureReviewSerializer,
    ModerateReviewSerializer,
    ReviewSerializer,
)
from apps.reviews.services import ReviewService

# Shipping
from apps.shipping.models import (
    Shipment,
    ShippingCarrier,
    ShippingMethod,
    ShippingRate,
    ShippingSettings,
    ShippingZone,
)
from apps.shipping.api.serializers import (
    ShipmentSerializer,
    ShippingCarrierSerializer,
    ShippingMethodSerializer,
    ShippingRateSerializer,
    ShippingSettingsSerializer,
    ShippingZoneSerializer,
)

# Payments
from apps.payments.models import BNPLProvider, Payment, PaymentGateway, PaymentLink
from apps.payments.api.serializers import (
    BNPLProviderSerializer,
    PaymentGatewaySerializer,
    PaymentLinkSerializer,
    PaymentSerializer,
    RecurringChargeSerializer,
    RefundSerializer,
)
from apps.payments.api.views import RecurringChargeAdminViewSet, RefundAdminViewSet

# Subscriptions
from apps.subscriptions.models import Plan, Subscription
from apps.subscriptions.api.serializers import PlanSerializer, SubscriptionSerializer

# Notifications
from apps.notifications.models import Notification, NotificationDelivery, NotificationTemplate
from apps.notifications.api.serializers import (
    NotificationDeliverySerializer,
    NotificationSerializer,
    NotificationTemplateSerializer,
)

# Analytics
from apps.analytics.api.views import DashboardViewSet, DailyStatViewSet

# Support / Chat
from apps.chat.api.views import (
    CannedResponseViewSet,
    ChatAgentViewSet,
    ChatAnalyticsViewSet,
    ChatSettingsViewSet,
    ConversationViewSet,
    MessageViewSet,
)


class AdminMfaPermissionMixin:
    permission_classes = [IsStaffWithMfa]

    def get_permissions(self):
        return [IsStaffWithMfa()]


# -----------------------------------------------------------------------------
# Catalog
# -----------------------------------------------------------------------------


class AdminCategoryViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Category.objects.all()

    def get_serializer_class(self):
        if self.action == "list":
            return CategoryListSerializer
        return CategorySerializer


class AdminProductViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Product.objects.all().select_related("primary_category").prefetch_related(
        "images",
        "variants",
        "categories",
        "tags",
        "attributes",
    )

    def get_serializer_class(self):
        if self.action == "list":
            return ProductListSerializer
        return ProductDetailSerializer


class AdminCollectionViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Collection.objects.all()

    def get_serializer_class(self):
        if self.action == "list":
            return CollectionListSerializer
        return CollectionDetailSerializer


class AdminBundleViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Bundle.objects.all()

    def get_serializer_class(self):
        if self.action == "list":
            return BundleListSerializer
        return BundleDetailSerializer


class AdminTagViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Tag.objects.all()
    serializer_class = TagSerializer


class AdminBadgeViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Badge.objects.all()
    serializer_class = BadgeSerializer


class AdminSpotlightViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Spotlight.objects.all()
    serializer_class = SpotlightSerializer


class AdminCustomerPhotoViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = CustomerPhoto.objects.all()
    serializer_class = CustomerPhotoSerializer


class AdminProductQuestionViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ProductQuestion.objects.all()
    serializer_class = ProductQuestionSerializer


class AdminProductAnswerViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ProductAnswer.objects.all()
    serializer_class = ProductAnswerSerializer


# -----------------------------------------------------------------------------
# Pricing (currencies, exchange rates)
# -----------------------------------------------------------------------------


class AdminCurrencyViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Currency.objects.all()
    serializer_class = CurrencySerializer


class AdminExchangeRateViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ExchangeRate.objects.all()
    serializer_class = ExchangeRateSerializer


# -----------------------------------------------------------------------------
# Promotions
# -----------------------------------------------------------------------------


class AdminCouponViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Coupon.objects.all()
    serializer_class = CouponSerializer


class AdminBannerViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Banner.objects.all()
    serializer_class = BannerSerializer


class AdminSaleViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Sale.objects.all()
    serializer_class = SaleSerializer


# -----------------------------------------------------------------------------
# CMS
# -----------------------------------------------------------------------------


class AdminPageViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Page.objects.all()

    def get_serializer_class(self):
        if self.action == "list":
            return PageListSerializer
        return PageDetailSerializer


class AdminFAQViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = FAQ.objects.all()
    serializer_class = FAQSerializer


class AdminContactMessageViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = ContactMessage.objects.all().order_by("-created_at")
    serializer_class = ContactMessageSerializer

    @action(detail=True, methods=["post"], url_path="mark-read")
    def mark_read(self, request, pk=None):
        ContactService.mark_as_read(pk)
        return Response(
            {"success": True, "message": "Message marked as read", "data": {}, "meta": {}}
        )

    @action(detail=True, methods=["post"])
    def reply(self, request, pk=None):
        reply_text = request.data.get("reply", "").strip()
        if not reply_text:
            return Response(
                {
                    "success": False,
                    "message": "Reply text is required",
                    "data": {},
                    "meta": {},
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        message = ContactService.reply_to_message(pk, reply_text, request.user)
        if not message:
            return Response(
                {
                    "success": False,
                    "message": "Message not found",
                    "data": {},
                    "meta": {},
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        return Response(
            {
                "success": True,
                "message": "Reply sent successfully",
                "data": ContactMessageSerializer(message).data,
                "meta": {},
            }
        )


class AdminSubscriberSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subscriber
        fields = ["id", "email", "name", "source", "is_active", "subscribed_at"]
        read_only_fields = ["id", "subscribed_at"]


class AdminSubscriberViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Subscriber.objects.all().order_by("-subscribed_at")
    serializer_class = AdminSubscriberSerializer


class AdminSiteSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = SiteSettings
        fields = "__all__"


class AdminSiteSettingsView(APIView):
    permission_classes = [IsStaffWithMfa]

    def get(self, request):
        settings_obj = SiteSettings.get_settings()
        data = AdminSiteSettingsSerializer(settings_obj).data
        return Response(
            {
                "success": True,
                "message": "Site settings retrieved successfully",
                "data": data,
                "meta": {},
            }
        )

    def put(self, request):
        settings_obj = SiteSettings.get_settings()
        serializer = AdminSiteSettingsSerializer(settings_obj, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(
            {
                "success": True,
                "message": "Site settings updated successfully",
                "data": serializer.data,
                "meta": {},
            }
        )

    def patch(self, request):
        return self.put(request)


# -----------------------------------------------------------------------------
# Reviews
# -----------------------------------------------------------------------------


class AdminReviewViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = Review.objects.select_related("product", "user").prefetch_related(
        "images",
        "votes",
    )
    serializer_class = ReviewSerializer

    @action(detail=True, methods=["post"], url_path="moderate")
    def moderate(self, request, pk=None):
        review = self.get_object()
        serializer = ModerateReviewSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        action_value = serializer.validated_data["action"]
        notes = serializer.validated_data.get("notes", "")
        review = ReviewService.moderate_review(
            review=review,
            moderator=request.user,
            approve=action_value == "approve",
            notes=notes,
        )
        return Response(
            {
                "success": True,
                "message": f"Review {action_value}d",
                "data": ReviewSerializer(review, context={"request": request}).data,
            }
        )

    @action(detail=True, methods=["post"], url_path="feature")
    def feature(self, request, pk=None):
        review = self.get_object()
        serializer = FeatureReviewSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        review = ReviewService.feature_review(
            review=review,
            featured=serializer.validated_data["is_featured"],
        )
        return Response(
            {
                "success": True,
                "message": "Feature state updated",
                "data": ReviewSerializer(review, context={"request": request}).data,
            }
        )


# -----------------------------------------------------------------------------
# Shipping
# -----------------------------------------------------------------------------


class AdminShippingZoneViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ShippingZone.objects.all()
    serializer_class = ShippingZoneSerializer


class AdminShippingCarrierViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ShippingCarrier.objects.all()
    serializer_class = ShippingCarrierSerializer


class AdminShippingMethodViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ShippingMethod.objects.all()
    serializer_class = ShippingMethodSerializer


class AdminShippingRateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShippingRate
        fields = "__all__"


class AdminShippingRateViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ShippingRate.objects.all()
    serializer_class = AdminShippingRateSerializer


class AdminShipmentViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Shipment.objects.all()
    serializer_class = ShipmentSerializer


class AdminShippingSettingsView(APIView):
    permission_classes = [IsStaffWithMfa]

    def get(self, request):
        settings_obj = ShippingSettings.get_settings()
        return Response(
            {
                "success": True,
                "message": "Shipping settings retrieved successfully",
                "data": ShippingSettingsSerializer(settings_obj).data,
                "meta": {},
            }
        )

    def put(self, request):
        settings_obj = ShippingSettings.get_settings()
        serializer = ShippingSettingsSerializer(settings_obj, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(
            {
                "success": True,
                "message": "Shipping settings updated successfully",
                "data": serializer.data,
                "meta": {},
            }
        )

    def patch(self, request):
        return self.put(request)


# -----------------------------------------------------------------------------
# Payments
# -----------------------------------------------------------------------------


class AdminPaymentGatewayViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = PaymentGateway.objects.all()
    serializer_class = PaymentGatewaySerializer


class AdminPaymentViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = Payment.objects.select_related("order").all().order_by("-created_at")
    serializer_class = PaymentSerializer


class AdminPaymentLinkViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = PaymentLink.objects.select_related("order", "gateway").all()
    serializer_class = PaymentLinkSerializer


class AdminBNPLProviderViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = BNPLProvider.objects.all()
    serializer_class = BNPLProviderSerializer


class AdminRefundViewSet(AdminMfaPermissionMixin, RefundAdminViewSet):
    pass


class AdminRecurringChargeViewSet(AdminMfaPermissionMixin, RecurringChargeAdminViewSet):
    serializer_class = RecurringChargeSerializer


# -----------------------------------------------------------------------------
# Subscriptions
# -----------------------------------------------------------------------------


class AdminPlanViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Plan.objects.all()
    serializer_class = PlanSerializer


class AdminSubscriptionViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Subscription.objects.all()
    serializer_class = SubscriptionSerializer


# -----------------------------------------------------------------------------
# Notifications
# -----------------------------------------------------------------------------


class AdminNotificationViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = Notification.objects.select_related("user").all().order_by("-created_at")
    serializer_class = NotificationSerializer


class AdminNotificationDeliveryViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = NotificationDelivery.objects.select_related("notification", "notification__user")
    serializer_class = NotificationDeliverySerializer


class AdminNotificationTemplateViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = NotificationTemplate.objects.all()
    serializer_class = NotificationTemplateSerializer


# -----------------------------------------------------------------------------
# Analytics
# -----------------------------------------------------------------------------


class AdminAnalyticsDashboardViewSet(AdminMfaPermissionMixin, DashboardViewSet):
    pass


class AdminAnalyticsDailyViewSet(AdminMfaPermissionMixin, DailyStatViewSet):
    pass


# -----------------------------------------------------------------------------
# Support / Chat
# -----------------------------------------------------------------------------


class AdminChatAgentViewSet(AdminMfaPermissionMixin, ChatAgentViewSet):
    pass


class AdminConversationViewSet(AdminMfaPermissionMixin, ConversationViewSet):
    pass


class AdminMessageViewSet(AdminMfaPermissionMixin, MessageViewSet):
    pass


class AdminCannedResponseViewSet(AdminMfaPermissionMixin, CannedResponseViewSet):
    pass


class AdminChatSettingsViewSet(AdminMfaPermissionMixin, ChatSettingsViewSet):
    pass


class AdminChatAnalyticsViewSet(AdminMfaPermissionMixin, ChatAnalyticsViewSet):
    pass


# -----------------------------------------------------------------------------
# System health details
# -----------------------------------------------------------------------------


class AdminSystemHealthDetailsView(APIView):
    permission_classes = [IsStaffWithMfa]

    def get(self, request):
        from core.views import check_cache, check_database, check_redis, check_storage
        from core.admin_dashboard import check_system_health

        basic_checks = {
            "database": check_database(),
            "cache": check_cache(),
            "redis": check_redis(),
            "storage": check_storage(),
        }
        system_checks = check_system_health()
        overall = "healthy" if all(
            item.get("status") in {"healthy", "ok", "skipped"}
            for item in [*basic_checks.values(), *system_checks.values()]
        ) else "degraded"

        return Response(
            {
                "success": True,
                "message": "System health details retrieved.",
                "data": {
                    "status": overall,
                    "service": "bunoraa-admin",
                    "timestamp": timezone.now().isoformat(),
                    "checks": basic_checks,
                    "services": system_checks,
                },
                "meta": None,
            }
        )
