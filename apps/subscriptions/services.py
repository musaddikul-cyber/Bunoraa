"""Business logic for subscriptions and Stripe orchestration."""
from __future__ import annotations

from django.conf import settings
from django.utils import timezone
from django.db import transaction
from typing import Optional

import stripe

from .models import Subscription, Plan
from apps.payments.services import StripeService
from apps.payments.models import RecurringCharge

stripe.api_key = getattr(settings, "STRIPE_SECRET_KEY", "")


class SubscriptionService:
    @staticmethod
    def create(user, plan: Plan, quantity: int = 1, metadata: dict = None, payment_method_id: Optional[str] = None):
        """Create a subscription both in Stripe and locally.

        Ensures the user has a Stripe customer id (saved to user.stripe_customer_id when possible).
        """
        metadata = metadata or {}

        # Ensure Stripe customer exists
        customer_id = getattr(user, "stripe_customer_id", None)
        if not customer_id:
            customer = StripeService.create_customer(email=user.email, name=user.get_full_name(), metadata={"user_id": str(user.id)})
            customer_id = customer.id
            setattr(user, "stripe_customer_id", customer_id)
            user.save(update_fields=["stripe_customer_id"])

        stripe_params = {
            "customer": customer_id,
            "items": [{"price": plan.stripe_price_id, "quantity": quantity}],
            "metadata": metadata,
            "expand": ["latest_invoice.payment_intent"],
        }

        if plan.trial_period_days:
            stripe_params["trial_period_days"] = int(plan.trial_period_days)

        # If a payment method is provided, attach it and set as default
        if payment_method_id:
            try:
                StripeService.attach_payment_method(payment_method_id, customer_id)
                stripe.Customer.modify(customer_id, invoice_settings={"default_payment_method": payment_method_id})
            except Exception:
                pass

        sub = stripe.Subscription.create(**stripe_params)

        # Persist local model
        with transaction.atomic():
            s, _ = Subscription.objects.update_or_create(
                stripe_subscription_id=sub.id,
                defaults={
                    "user": user,
                    "plan": plan,
                    "status": sub.status,
                    "current_period_start": timezone.datetime.fromtimestamp(sub.current_period_start, tz=timezone.utc) if getattr(sub, "current_period_start", None) else None,
                    "current_period_end": timezone.datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc) if getattr(sub, "current_period_end", None) else None,
                    "quantity": quantity,
                    "metadata": metadata or {},
                    "next_billing_at": timezone.datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc) if getattr(sub, "current_period_end", None) else None,
                },
            )

        return s

    @staticmethod
    def cancel(subscription: Subscription, cancel_at_period_end: bool = True, at=None):
        at = at or timezone.now()
        if not subscription.stripe_subscription_id:
            subscription.mark_canceled(at=at)
            return subscription

        if cancel_at_period_end:
            stripe.Subscription.modify(subscription.stripe_subscription_id, cancel_at_period_end=True)
            subscription.canceled_at = at
            subscription.status = Subscription.STATUS_CANCELED
            subscription.save(update_fields=["canceled_at", "status"])
        else:
            stripe.Subscription.delete(subscription.stripe_subscription_id)
            subscription.canceled_at = at
            subscription.ended_at = at
            subscription.status = Subscription.STATUS_CANCELED
            subscription.save(update_fields=["canceled_at", "ended_at", "status"])
        return subscription

    @staticmethod
    def resume(subscription: Subscription):
        if not subscription.stripe_subscription_id:
            subscription.status = Subscription.STATUS_ACTIVE
            subscription.save(update_fields=["status"])
            return subscription

        # Unset cancel at period end if present
        try:
            stripe.Subscription.modify(subscription.stripe_subscription_id, cancel_at_period_end=False)
            subscription.status = Subscription.STATUS_ACTIVE
            subscription.canceled_at = None
            subscription.save(update_fields=["status", "canceled_at"])
        except stripe.error.StripeError:
            # If subscription was deleted upstream, attempt to reactivate by creating a new one
            subscription.status = Subscription.STATUS_CANCELED
            subscription.save(update_fields=["status"])
        return subscription

    @staticmethod
    def change_plan(subscription: Subscription, new_plan: Plan, proration_behavior: str = "none"):
        if not subscription.stripe_subscription_id:
            # Local change only
            subscription.plan = new_plan
            subscription.save(update_fields=["plan"])
            return subscription

        # Find item id for subscription
        stripe_sub = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
        items = stripe_sub.get("items", {}).get("data", [])
        if not items:
            raise ValueError("Unable to locate subscription items in Stripe subscription")

        item_id = items[0].id
        stripe.Subscription.modify(
            subscription.stripe_subscription_id,
            items=[{"id": item_id, "price": new_plan.stripe_price_id, "quantity": subscription.quantity}],
            proration_behavior=proration_behavior,
        )

        # Update local
        subscription.plan = new_plan
        subscription.save(update_fields=["plan"])
        return subscription

    @staticmethod
    def preview_invoice(subscription: Subscription):
        if not subscription.stripe_subscription_id:
            raise ValueError("No external subscription attached")
        customer_id = getattr(subscription.user, "stripe_customer_id", None)
        invoice = stripe.Invoice.upcoming(customer=customer_id, subscription=subscription.stripe_subscription_id)
        # Minimal normalized response
        return {
            "amount_due": getattr(invoice, "amount_due", None),
            "currency": getattr(invoice, "currency", None),
            "lines": [l.to_dict() for l in getattr(invoice, "lines", {}).get("data", [])],
            "period_start": getattr(invoice, "period_start", None),
            "period_end": getattr(invoice, "period_end", None),
        }

    @staticmethod
    def sync_with_stripe(subscription: Subscription):
        if not subscription.stripe_subscription_id:
            return subscription
        try:
            stripe_sub = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
        except stripe.error.InvalidRequestError:
            # Subscription not found in Stripe
            subscription.status = Subscription.STATUS_CANCELED
            subscription.ended_at = timezone.now()
            subscription.save(update_fields=["status", "ended_at"])
            return subscription

        subscription.status = getattr(stripe_sub, "status", subscription.status)
        subscription.current_period_start = timezone.datetime.fromtimestamp(getattr(stripe_sub, "current_period_start", 0), tz=timezone.utc) if getattr(stripe_sub, "current_period_start", None) else None
        subscription.current_period_end = timezone.datetime.fromtimestamp(getattr(stripe_sub, "current_period_end", 0), tz=timezone.utc) if getattr(stripe_sub, "current_period_end", None) else None
        subscription.next_billing_at = subscription.current_period_end
        subscription.save()
        return subscription

    # Webhook helpers
    @staticmethod
    def handle_invoice_payment_succeeded(invoice_data: dict):
        sub_id = invoice_data.get("subscription")
        if not sub_id:
            return
        try:
            subscription = Subscription.objects.filter(stripe_subscription_id=sub_id).first()
            if not subscription:
                return

            amount = (invoice_data.get("total") or invoice_data.get("amount_due") or 0) / 100.0
            currency = invoice_data.get("currency")
            attempt_at = timezone.datetime.fromtimestamp(invoice_data.get("created"), tz=timezone.utc) if invoice_data.get("created") else timezone.now()

            RecurringCharge.objects.create(
                subscription=subscription,
                amount=amount,
                currency=currency.upper() if currency else subscription.plan.currency,
                status="succeeded",
                attempt_at=attempt_at,
                processed_at=timezone.now(),
                stripe_subscription_id=sub_id,
                metadata={"stripe_invoice_id": invoice_data.get("id")},
            )

            # Ensure subscription marked active
            subscription.status = Subscription.STATUS_ACTIVE
            if invoice_data.get("period_end"):
                subscription.current_period_end = timezone.datetime.fromtimestamp(invoice_data.get("period_end"), tz=timezone.utc)
            subscription.save()
        except Exception:
            # swallow - webhook processing should be idempotent and resilient
            pass

    @staticmethod
    def handle_invoice_payment_failed(invoice_data: dict):
        sub_id = invoice_data.get("subscription")
        if not sub_id:
            return
        subscription = Subscription.objects.filter(stripe_subscription_id=sub_id).first()
        if not subscription:
            return
        # Mark as past due
        subscription.status = Subscription.STATUS_PAST_DUE
        subscription.save(update_fields=["status"])

    @staticmethod
    def handle_subscription_deleted(data: dict):
        sub_id = data.get("id")
        subscription = Subscription.objects.filter(stripe_subscription_id=sub_id).first()
        if not subscription:
            return
        subscription.status = Subscription.STATUS_CANCELED
        subscription.ended_at = timezone.now()
        subscription.save(update_fields=["status", "ended_at"])