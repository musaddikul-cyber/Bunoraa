from __future__ import annotations

from core.seed.base import JSONSeedSpec
from core.seed.registry import register_seed
from apps.payments.models import PaymentGateway, BNPLProvider


register_seed(
    JSONSeedSpec(
        name="payments.payment_gateways",
        app_label="payments",
        model=PaymentGateway,
        data_path="apps/payments/data/payment_gateways.json",
        key_fields=["code"],
        update_fields=[
            "code",
            "name",
            "description",
            "icon_class",
            "color",
            "fee_type",
            "fee_amount",
            "fee_text",
            "is_active",
            "currencies",
            "countries",
            "min_amount",
            "max_amount",
            "api_key",
            "api_secret",
            "merchant_id",
            "webhook_secret",
            "is_sandbox",
            "ssl_store_id",
            "ssl_store_passwd",
            "bkash_mode",
            "bkash_app_key",
            "bkash_app_secret",
            "bkash_username",
            "bkash_password",
            "nagad_merchant_id",
            "nagad_public_key",
            "nagad_private_key",
            "supports_partial",
            "supports_recurring",
            "supports_bnpl",
            "instructions",
            "bank_name",
            "bank_account_name",
            "bank_account_number",
            "bank_routing_number",
            "bank_branch",
            "sort_order",
        ],
    )
)

register_seed(
    JSONSeedSpec(
        name="payments.bnpl_providers",
        app_label="payments",
        model=BNPLProvider,
        data_path="apps/payments/data/bnpl_providers.json",
        key_fields=["code"],
        update_fields=["code", "name", "is_active", "config"],
    )
)
