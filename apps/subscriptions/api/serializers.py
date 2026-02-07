from rest_framework import serializers
from ..models import Plan, Subscription
from apps.payments.models import RecurringCharge


class PlanSerializer(serializers.ModelSerializer):
    class Meta:
        model = Plan
        fields = (
            "id",
            "name",
            "description",
            "interval",
            "price_amount",
            "currency",
            "stripe_price_id",
            "trial_period_days",
            "active",
            "metadata",
        )


class RecurringChargeInlineSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecurringCharge
        fields = ("id", "amount", "currency", "status", "attempt_at", "processed_at", "stripe_subscription_id")


class SubscriptionSerializer(serializers.ModelSerializer):
    plan = PlanSerializer(read_only=True)
    plan_id = serializers.UUIDField(write_only=True, required=True)
    recurring_charges = RecurringChargeInlineSerializer(many=True, read_only=True)

    class Meta:
        model = Subscription
        fields = (
            "id",
            "plan",
            "plan_id",
            "status",
            "stripe_subscription_id",
            "quantity",
            "metadata",
            "trial_ends",
            "current_period_start",
            "current_period_end",
            "next_billing_at",
            "canceled_at",
            "ended_at",
            "created_at",
            "updated_at",
            "recurring_charges",
        )
        read_only_fields = ("created_at", "updated_at", "current_period_start", "current_period_end", "next_billing_at", "canceled_at", "ended_at")

    def create(self, validated_data):
        user = self.context["request"].user
        plan_id = validated_data.pop("plan_id")
        quantity = validated_data.get("quantity", 1)
        metadata = validated_data.get("metadata", {})

        from ..services import SubscriptionService
        plan = Plan.objects.get(id=plan_id)
        sub = SubscriptionService.create(user=user, plan=plan, quantity=quantity, metadata=metadata)
        return sub

    def update(self, instance, validated_data):
        # Allow metadata and quantity updates
        instance.quantity = validated_data.get("quantity", instance.quantity)
        instance.metadata = validated_data.get("metadata", instance.metadata)
        instance.save(update_fields=["quantity", "metadata"])
        return instance


class ChangePlanSerializer(serializers.Serializer):
    plan_id = serializers.UUIDField(required=True)
    proration_behavior = serializers.ChoiceField(choices=("none", "create_prorations"), default="none")


class PreviewInvoiceSerializer(serializers.Serializer):
    subscription_id = serializers.UUIDField(required=True)
