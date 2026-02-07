from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from django.conf import settings
from django.utils import timezone

import stripe

from ..models import Plan, Subscription
from .serializers import PlanSerializer, SubscriptionSerializer, ChangePlanSerializer, PreviewInvoiceSerializer
from ..services import SubscriptionService


class PlanViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Plan.objects.filter(active=True)
    serializer_class = PlanSerializer
    permission_classes = [AllowAny]

    def list(self, request, *args, **kwargs):
        qs = self.get_queryset()
        serializer = self.get_serializer(qs, many=True)
        return Response({"success": True, "message": "Plans retrieved", "data": serializer.data, "meta": {"count": len(serializer.data)}})


class SubscriptionViewSet(viewsets.ModelViewSet):
    serializer_class = SubscriptionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Subscription.objects.filter(user=self.request.user, is_deleted=False)

    def list(self, request, *args, **kwargs):
        qs = self.get_queryset()
        serializer = self.get_serializer(qs, many=True)
        return Response({"success": True, "message": "Subscriptions retrieved", "data": serializer.data, "meta": {"count": len(serializer.data)}})

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        sub = serializer.save()
        return Response({"success": True, "message": "Subscription created", "data": self.get_serializer(sub).data}, status=status.HTTP_201_CREATED)

    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        sub = serializer.save()
        return Response({"success": True, "message": "Subscription updated", "data": self.get_serializer(sub).data})

    @action(detail=True, methods=["post"])
    def change_plan(self, request, pk=None):
        instance = self.get_object()
        serializer = ChangePlanSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        plan_id = serializer.validated_data["plan_id"]
        proration = serializer.validated_data.get("proration_behavior", "none")

        plan = get_object_or_404(Plan, id=plan_id)
        SubscriptionService.change_plan(instance, plan, proration_behavior=proration)
        instance.refresh_from_db()
        return Response({"success": True, "message": "Plan changed", "data": self.get_serializer(instance).data})

    @action(detail=True, methods=["post"])
    def resume(self, request, pk=None):
        instance = self.get_object()
        SubscriptionService.resume(instance)
        instance.refresh_from_db()
        return Response({"success": True, "message": "Subscription resumed", "data": self.get_serializer(instance).data})

    @action(detail=True, methods=["post"])
    def cancel(self, request, pk=None):
        instance = self.get_object()
        at_period_end = bool(request.data.get("at_period_end", False))
        SubscriptionService.cancel(instance, cancel_at_period_end=at_period_end, at=timezone.now())
        instance.refresh_from_db()
        return Response({"success": True, "message": "Subscription canceled", "data": self.get_serializer(instance).data})

    @action(detail=True, methods=["get"])
    def preview_invoice(self, request, pk=None):
        instance = self.get_object()
        try:
            invoice = SubscriptionService.preview_invoice(instance)
            return Response({"success": True, "message": "Preview invoice", "data": invoice})
        except stripe.error.StripeError as e:
            return Response({"success": False, "message": str(e), "data": {}}, status=status.HTTP_400_BAD_REQUEST)


class StripeWebhookView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        payload = request.body
        sig_header = request.META.get("HTTP_STRIPE_SIGNATURE", "")
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, settings.STRIPE_WEBHOOK_SECRET)
        except Exception as exc:
            return Response({"success": False, "message": "Invalid webhook signature", "error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        # Handle events
        typ = event["type"]
        data = event["data"]["object"]

        # Invoice paid
        if typ == "invoice.payment_succeeded":
            SubscriptionService.handle_invoice_payment_succeeded(data)

        # Invoice failed
        if typ == "invoice.payment_failed":
            SubscriptionService.handle_invoice_payment_failed(data)

        if typ == "customer.subscription.deleted":
            SubscriptionService.handle_subscription_deleted(data)

        return Response({"success": True, "message": "Webhook received"})