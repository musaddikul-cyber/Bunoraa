
from __future__ import annotations

from datetime import datetime

from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.utils.dateparse import parse_datetime
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
    CategoryFacet,
    CategoryPricingProfile,
    Collection,
    CustomerPhoto,
    EcoCertification,
    Facet,
    Product,
    ProductAnswer,
    ProductPrice,
    ProductQuestion,
    ProductVariant,
    ShippingMaterial,
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
from apps.promotions.models import Banner, Coupon, CouponUsage, Sale
from apps.promotions.api.serializers import BannerSerializer, CouponSerializer, SaleSerializer

# CMS
from apps.pages.models import (
    ContactMessage,
    FAQ,
    NewsletterIncentive,
    Page,
    SiteSettings,
    SocialLink,
    Subscriber,
)
from apps.pages.api.serializers import (
    ContactMessageSerializer,
    FAQSerializer,
    PageDetailSerializer,
    PageListSerializer,
    SiteSettingsSerializer,
)
from apps.pages.services import ContactService

# Reviews
from apps.reviews.models import Review, ReviewImage, ReviewReport, ReviewVote
from apps.reviews.api.serializers import (
    FeatureReviewSerializer,
    ModerateReviewSerializer,
    ReviewSerializer,
)
from apps.reviews.services import ReviewService

# Shipping
from apps.shipping.models import (
    Shipment,
    ShipmentEvent,
    ShippingCarrier,
    ShippingMethod,
    ShippingRate,
    ShippingRestriction,
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
from apps.payments.models import (
    BNPLAgreement,
    BNPLProvider,
    AuditLog as PaymentAuditLog,
    Payment,
    PaymentGateway,
    PaymentLink,
    PaymentMethod,
    PaymentTransaction,
)
from apps.payments.api.serializers import (
    BNPLProviderSerializer,
    PaymentGatewaySerializer,
    PaymentLinkSerializer,
    PaymentMethodSerializer,
    PaymentSerializer,
    RecurringChargeSerializer,
    RefundSerializer,
)
from apps.payments.api.views import RecurringChargeAdminViewSet, RefundAdminViewSet

# Subscriptions
from apps.subscriptions.models import Plan, Subscription
from apps.subscriptions.api.serializers import PlanSerializer, SubscriptionSerializer

# Notifications
from apps.notifications.models import (
    EmailLog,
    Notification,
    NotificationDelivery,
    NotificationPreference,
    NotificationTemplate,
    PushToken,
)
from apps.notifications.api.serializers import (
    NotificationDeliverySerializer,
    NotificationPreferenceSerializer,
    NotificationSerializer,
    NotificationTemplateSerializer,
    PushTokenSerializer,
)

# Analytics
from apps.analytics.models import CartEvent, CategoryStat, DailyStat, PageView, ProductStat, SearchQuery
from apps.analytics.api.views import DashboardViewSet, DailyStatViewSet

# Support / Chat
from apps.chat.models import Conversation, MessageAttachment, TypingIndicator
from apps.chat.api.serializers import MessageAttachmentSerializer
from apps.chat.api.views import (
    CannedResponseViewSet,
    ChatAgentViewSet,
    ChatAnalyticsViewSet,
    ChatSettingsViewSet,
    ConversationViewSet,
    MessageViewSet,
)

from apps.orders.models import Order


User = get_user_model()


def _coerce_since(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    dt = parse_datetime(raw_value)
    if dt is None:
        raise ValueError("Invalid 'since' value. Use ISO-8601 datetime.")
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone=timezone.utc)
    return dt


def _status_label(value) -> str:
    return str(value or "").strip().lower()


def _is_healthy_status(value) -> bool:
    return _status_label(value) in {"healthy", "ok", "up", "skipped"}


def _module_for_notification(notification_type: str | None, reference_type: str | None = None) -> str:
    ntype = (notification_type or "").lower()
    rtype = (reference_type or "").lower()
    if ntype.startswith("order_") or rtype == "order":
        return "orders"
    if ntype.startswith("payment_") or "refund" in ntype or rtype == "payment":
        return "payments"
    if ntype.startswith("review_") or rtype == "review":
        return "reviews"
    if "stock" in ntype or "price" in ntype:
        return "catalog"
    if "promo" in ntype or "coupon" in ntype:
        return "promotions"
    if "subscription" in ntype:
        return "subscriptions"
    if "chat" in ntype or rtype in {"conversation", "message"}:
        return "support"
    return "notifications"


class AdminMfaPermissionMixin:
    permission_classes = [IsStaffWithMfa]

    def get_permissions(self):
        return [IsStaffWithMfa()]


# -----------------------------------------------------------------------------
# Admin serializers for expanded modules
# -----------------------------------------------------------------------------


class AdminProductVariantSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductVariant
        fields = "__all__"


class AdminShippingMaterialSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShippingMaterial
        fields = "__all__"


class AdminFacetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Facet
        fields = "__all__"


class AdminCategoryFacetSerializer(serializers.ModelSerializer):
    class Meta:
        model = CategoryFacet
        fields = "__all__"


class AdminEcoCertificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = EcoCertification
        fields = "__all__"


class AdminProductPriceSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductPrice
        fields = "__all__"


class AdminCategoryPricingProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = CategoryPricingProfile
        fields = "__all__"


class AdminCouponUsageSerializer(serializers.ModelSerializer):
    class Meta:
        model = CouponUsage
        fields = "__all__"


class AdminSocialLinkSerializer(serializers.ModelSerializer):
    class Meta:
        model = SocialLink
        fields = "__all__"


class AdminNewsletterIncentiveSerializer(serializers.ModelSerializer):
    class Meta:
        model = NewsletterIncentive
        fields = "__all__"


class AdminReviewImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReviewImage
        fields = "__all__"


class AdminReviewVoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReviewVote
        fields = "__all__"


class AdminReviewReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReviewReport
        fields = "__all__"


class AdminShippingRestrictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShippingRestriction
        fields = "__all__"


class AdminShipmentEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShipmentEvent
        fields = "__all__"


class AdminPaymentTransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaymentTransaction
        fields = "__all__"


class AdminBNPLAgreementSerializer(serializers.ModelSerializer):
    user_email = serializers.EmailField(source="user.email", read_only=True)
    provider_name = serializers.CharField(source="provider.name", read_only=True)

    class Meta:
        model = BNPLAgreement
        fields = "__all__"
        read_only_fields = ("created_at", "user_email", "provider_name")


class AdminPaymentAuditLogSerializer(serializers.ModelSerializer):
    user_email = serializers.EmailField(source="user.email", read_only=True)

    class Meta:
        model = PaymentAuditLog
        fields = "__all__"


class AdminPushTokenListSerializer(serializers.ModelSerializer):
    user_email = serializers.EmailField(source="user.email", read_only=True)

    class Meta:
        model = PushToken
        fields = "__all__"


class AdminEmailLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmailLog
        fields = "__all__"


class AdminProductStatSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductStat
        fields = "__all__"


class AdminCategoryStatSerializer(serializers.ModelSerializer):
    class Meta:
        model = CategoryStat
        fields = "__all__"


class AdminSearchQuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = SearchQuery
        fields = "__all__"


class AdminCartEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = CartEvent
        fields = "__all__"


class AdminPageViewSerializer(serializers.ModelSerializer):
    class Meta:
        model = PageView
        fields = "__all__"


class AdminTypingIndicatorSerializer(serializers.ModelSerializer):
    class Meta:
        model = TypingIndicator
        fields = "__all__"


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


class AdminProductVariantViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ProductVariant.objects.select_related("product").all()
    serializer_class = AdminProductVariantSerializer


class AdminShippingMaterialViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ShippingMaterial.objects.all()
    serializer_class = AdminShippingMaterialSerializer


class AdminFacetViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Facet.objects.all()
    serializer_class = AdminFacetSerializer


class AdminCategoryFacetViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = CategoryFacet.objects.select_related("category", "facet").all()
    serializer_class = AdminCategoryFacetSerializer


class AdminEcoCertificationViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = EcoCertification.objects.all()
    serializer_class = AdminEcoCertificationSerializer


# -----------------------------------------------------------------------------
# Pricing (currencies, exchange rates)
# -----------------------------------------------------------------------------


class AdminCurrencyViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Currency.objects.all()
    serializer_class = CurrencySerializer


class AdminExchangeRateViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ExchangeRate.objects.all()
    serializer_class = ExchangeRateSerializer


class AdminProductPriceViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ProductPrice.objects.select_related("product", "currency").all()
    serializer_class = AdminProductPriceSerializer


class AdminCategoryPricingProfileViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = CategoryPricingProfile.objects.select_related("category").all()
    serializer_class = AdminCategoryPricingProfileSerializer


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


class AdminCouponUsageViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = CouponUsage.objects.select_related("coupon", "user", "order").order_by("-created_at")
    serializer_class = AdminCouponUsageSerializer

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


class AdminSocialLinkViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = SocialLink.objects.all().order_by("order")
    serializer_class = AdminSocialLinkSerializer


class AdminNewsletterIncentiveViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = NewsletterIncentive.objects.all().order_by("-valid_from")
    serializer_class = AdminNewsletterIncentiveSerializer


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


class AdminReviewImageViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = ReviewImage.objects.select_related("review").order_by("-created_at")
    serializer_class = AdminReviewImageSerializer


class AdminReviewVoteViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = ReviewVote.objects.select_related("review", "user").order_by("-created_at")
    serializer_class = AdminReviewVoteSerializer


class AdminReviewReportViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = ReviewReport.objects.select_related("review", "reporter").order_by("-created_at")
    serializer_class = AdminReviewReportSerializer


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


class AdminShippingRestrictionViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = ShippingRestriction.objects.select_related("zone", "method").all()
    serializer_class = AdminShippingRestrictionSerializer


class AdminShipmentViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = Shipment.objects.all()
    serializer_class = ShipmentSerializer


class AdminShipmentEventViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = ShipmentEvent.objects.select_related("shipment").order_by("-occurred_at")
    serializer_class = AdminShipmentEventSerializer


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


class AdminPaymentMethodViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = PaymentMethod.objects.select_related("user").order_by("-created_at")
    serializer_class = PaymentMethodSerializer


class AdminPaymentTransactionViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = PaymentTransaction.objects.select_related("gateway", "payment", "user", "order").order_by("-created_at")
    serializer_class = AdminPaymentTransactionSerializer


class AdminPaymentLinkViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = PaymentLink.objects.select_related("order", "gateway").all()
    serializer_class = PaymentLinkSerializer


class AdminBNPLProviderViewSet(AdminMfaPermissionMixin, viewsets.ModelViewSet):
    queryset = BNPLProvider.objects.all()
    serializer_class = BNPLProviderSerializer


class AdminBNPLAgreementViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = BNPLAgreement.objects.select_related("user", "provider").order_by("-created_at")
    serializer_class = AdminBNPLAgreementSerializer


class AdminPaymentAuditLogViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = PaymentAuditLog.objects.select_related("user").order_by("-created_at")
    serializer_class = AdminPaymentAuditLogSerializer


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


class AdminNotificationPreferenceViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = NotificationPreference.objects.select_related("user").order_by("-updated_at")
    serializer_class = NotificationPreferenceSerializer


class AdminPushTokenViewSet(AdminMfaPermissionMixin, viewsets.ViewSet):
    def list(self, request):
        queryset = PushToken.objects.select_related("user").order_by("-last_used_at")
        scope = (request.query_params.get("scope") or "").lower()
        if not request.user.is_superuser or scope != "all":
            queryset = queryset.filter(user=request.user)
        data = AdminPushTokenListSerializer(queryset, many=True).data
        return Response(
            {
                "success": True,
                "message": "Push tokens retrieved successfully",
                "data": data,
                "meta": {"count": len(data)},
            }
        )

    def create(self, request):
        serializer = PushTokenSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated = serializer.validated_data
        token_obj, created = PushToken.objects.update_or_create(
            token=validated["token"],
            defaults={
                "user": request.user,
                "device_type": validated["device_type"],
                "device_name": validated.get("device_name"),
                "platform": validated.get("platform"),
                "app_version": validated.get("app_version"),
                "locale": validated.get("locale"),
                "timezone": validated.get("timezone"),
                "browser": validated.get("browser"),
                "user_agent": validated.get("user_agent") or request.META.get("HTTP_USER_AGENT", ""),
                "last_ip": request.META.get("REMOTE_ADDR"),
                "is_active": True,
            },
        )
        return Response(
            {
                "success": True,
                "message": "Push token registered successfully",
                "data": {"token_id": str(token_obj.id)},
                "meta": {"created": created},
            },
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )

    def destroy(self, request, pk=None):
        queryset = PushToken.objects.filter(id=pk)
        if not request.user.is_superuser:
            queryset = queryset.filter(user=request.user)
        deleted = queryset.delete()[0] > 0
        if not deleted:
            return Response(
                {
                    "success": False,
                    "message": "Push token not found",
                    "data": {},
                    "meta": {},
                },
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(
            {
                "success": True,
                "message": "Push token removed successfully",
                "data": {},
                "meta": {},
            }
        )


class AdminEmailLogViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = EmailLog.objects.select_related("recipient_user", "notification", "delivery").order_by("-created_at")
    serializer_class = AdminEmailLogSerializer


# -----------------------------------------------------------------------------
# Analytics
# -----------------------------------------------------------------------------


class AdminAnalyticsDashboardViewSet(AdminMfaPermissionMixin, DashboardViewSet):
    pass


class AdminAnalyticsDailyViewSet(AdminMfaPermissionMixin, DailyStatViewSet):
    queryset = DailyStat.objects.all()


class AdminAnalyticsProductStatViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = ProductStat.objects.select_related("product").order_by("-date")
    serializer_class = AdminProductStatSerializer


class AdminAnalyticsCategoryStatViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = CategoryStat.objects.select_related("category").order_by("-date")
    serializer_class = AdminCategoryStatSerializer


class AdminAnalyticsSearchQueryViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = SearchQuery.objects.order_by("-created_at")
    serializer_class = AdminSearchQuerySerializer


class AdminAnalyticsCartEventViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = CartEvent.objects.order_by("-created_at")
    serializer_class = AdminCartEventSerializer


class AdminAnalyticsPageViewViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = PageView.objects.order_by("-created_at")
    serializer_class = AdminPageViewSerializer


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


class AdminMessageAttachmentViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = MessageAttachment.objects.select_related("message").order_by("-created_at")
    serializer_class = MessageAttachmentSerializer


class AdminTypingIndicatorViewSet(AdminMfaPermissionMixin, viewsets.ReadOnlyModelViewSet):
    queryset = TypingIndicator.objects.select_related("conversation", "user").order_by("-started_at")
    serializer_class = AdminTypingIndicatorSerializer


# -----------------------------------------------------------------------------
# Realtime polling fallback
# -----------------------------------------------------------------------------


class AdminRealtimeEventsView(APIView):
    permission_classes = [IsStaffWithMfa]

    def get(self, request):
        try:
            limit_raw = request.query_params.get("limit", "50")
            limit = min(max(int(limit_raw), 1), 200)
        except (TypeError, ValueError):
            return Response(
                {
                    "success": False,
                    "message": "Invalid 'limit' value. Must be an integer.",
                    "data": {},
                    "meta": {},
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            since = _coerce_since(request.query_params.get("since"))
        except ValueError as exc:
            return Response(
                {
                    "success": False,
                    "message": str(exc),
                    "data": {},
                    "meta": {},
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        events: list[dict] = []

        notifications_qs = Notification.objects.select_related("user").order_by("-created_at")
        if since:
            notifications_qs = notifications_qs.filter(created_at__gt=since)
        for notif in notifications_qs[:limit]:
            events.append(
                {
                    "type": "notification",
                    "module": _module_for_notification(notif.type, notif.reference_type),
                    "entity_type": notif.reference_type or "notification",
                    "entity_id": notif.reference_id or str(notif.id),
                    "timestamp": notif.created_at.isoformat(),
                    "payload": {
                        "notification_id": str(notif.id),
                        "notification_type": notif.type,
                        "title": notif.title,
                        "message": notif.message,
                        "url": notif.url or "",
                        "user_id": str(notif.user_id),
                        "priority": notif.priority,
                    },
                }
            )

        orders_qs = Order.objects.order_by("-updated_at")
        if since:
            orders_qs = orders_qs.filter(updated_at__gt=since)
        for order in orders_qs[:limit]:
            events.append(
                {
                    "type": "order_update",
                    "module": "orders",
                    "entity_type": "order",
                    "entity_id": str(order.id),
                    "timestamp": order.updated_at.isoformat(),
                    "payload": {
                        "order_id": str(order.id),
                        "order_number": order.order_number,
                        "status": order.status,
                        "payment_status": order.payment_status,
                    },
                }
            )

        payments_qs = Payment.objects.order_by("-updated_at")
        if since:
            payments_qs = payments_qs.filter(updated_at__gt=since)
        for payment in payments_qs[:limit]:
            events.append(
                {
                    "type": "payment_update",
                    "module": "payments",
                    "entity_type": "payment",
                    "entity_id": str(payment.id),
                    "timestamp": payment.updated_at.isoformat(),
                    "payload": {
                        "payment_id": str(payment.id),
                        "status": payment.status,
                        "amount": str(payment.amount),
                        "currency": payment.currency,
                        "order_id": str(payment.order_id) if payment.order_id else "",
                    },
                }
            )

        conversations_qs = Conversation.objects.order_by("-updated_at")
        if since:
            conversations_qs = conversations_qs.filter(updated_at__gt=since)
        for conversation in conversations_qs[:limit]:
            events.append(
                {
                    "type": "chat_update",
                    "module": "support",
                    "entity_type": "conversation",
                    "entity_id": str(conversation.id),
                    "timestamp": conversation.updated_at.isoformat(),
                    "payload": {
                        "conversation_id": str(conversation.id),
                        "status": conversation.status,
                        "priority": conversation.priority,
                        "subject": conversation.subject or "",
                        "last_message_at": (
                            conversation.last_message_at.isoformat()
                            if conversation.last_message_at
                            else ""
                        ),
                    },
                }
            )

        events.sort(key=lambda item: item.get("timestamp", ""))
        events = events[-limit:]
        next_since = events[-1]["timestamp"] if events else timezone.now().isoformat()

        return Response(
            {
                "success": True,
                "message": "Realtime events retrieved successfully",
                "data": {
                    "events": events,
                    "next_since": next_since,
                    "server_time": timezone.now().isoformat(),
                },
                "meta": {"count": len(events), "limit": limit},
            }
        )


# -----------------------------------------------------------------------------
# System health details
# -----------------------------------------------------------------------------


class AdminSystemHealthDetailsView(APIView):
    permission_classes = [IsStaffWithMfa]

    def _check_workers(self):
        broker_url = getattr(settings, "CELERY_BROKER_URL", "")
        if not broker_url:
            return {"status": "skipped", "reason": "CELERY_BROKER_URL is not configured"}
        try:
            from celery import current_app

            inspector = current_app.control.inspect(timeout=1.0)
            ping = inspector.ping() or {}
            if ping:
                workers = sorted(ping.keys())
                return {"status": "ok", "workers": workers, "worker_count": len(workers)}
            return {"status": "degraded", "workers": [], "worker_count": 0}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def _module_counts(self):
        return {
            "catalog": {
                "products": Product.objects.filter(is_deleted=False).count(),
                "categories": Category.objects.filter(is_deleted=False).count(),
                "collections": Collection.objects.count(),
            },
            "orders": {
                "total": Order.objects.filter(is_deleted=False).count(),
                "pending": Order.objects.filter(is_deleted=False, status=Order.STATUS_PENDING).count(),
            },
            "payments": {
                "payments": Payment.objects.count(),
                "audit_logs": PaymentAuditLog.objects.count(),
            },
            "subscriptions": {
                "plans": Plan.objects.count(),
                "subscriptions": Subscription.objects.count(),
            },
            "notifications": {
                "notifications": Notification.objects.count(),
                "push_tokens": PushToken.objects.count(),
            },
            "support": {
                "conversations": Conversation.objects.count(),
                "attachments": MessageAttachment.objects.count(),
            },
            "analytics": {
                "daily_stats": DailyStat.objects.count(),
                "page_views": PageView.objects.count(),
            },
            "users": {
                "total": User.objects.count(),
                "staff": User.objects.filter(is_staff=True).count(),
            },
        }

    def get(self, request):
        from core.views import check_cache, check_database, check_redis, check_storage
        from core.admin_dashboard import check_system_health
        from core.websocket.monitoring import WebSocketHealthCheck

        basic_checks = {
            "database": check_database(),
            "cache": check_cache(),
            "redis": check_redis(),
            "storage": check_storage(),
        }
        service_checks = check_system_health()
        websocket = WebSocketHealthCheck.check_health()
        workers = self._check_workers()

        all_status_values = []
        all_status_values.extend([check.get("status") for check in basic_checks.values()])
        all_status_values.extend([check.get("status") for check in service_checks.values()])
        all_status_values.append(websocket.get("status"))
        all_status_values.append(workers.get("status"))

        overall = "healthy" if all(_is_healthy_status(value) for value in all_status_values) else "degraded"

        return Response(
            {
                "success": True,
                "message": "System health details retrieved.",
                "data": {
                    "status": overall,
                    "service": "bunoraa-admin",
                    "timestamp": timezone.now().isoformat(),
                    "checks": basic_checks,
                    "services": service_checks,
                    "websocket": websocket,
                    "workers": workers,
                    "module_counts": self._module_counts(),
                },
                "meta": None,
            }
        )
