"use client";

import * as React from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { formatAddressLine } from "@/lib/address";
import { formatMoney } from "@/lib/checkout";
import type { CartSummary, CheckoutSession, CheckoutValidation } from "@/lib/types";

const schema = z.object({
  terms_accepted: z
    .boolean()
    .refine((value) => value, "You must accept the terms to continue."),
  order_notes: z.string().optional(),
});

type ReviewFormValues = z.infer<typeof schema>;

type CheckoutReviewStepProps = {
  checkoutSession?: CheckoutSession | null;
  shippingCountryName?: string | null;
  billingCountryName?: string | null;
  cartSummary?: CartSummary | null;
  validation?: CheckoutValidation | null;
  isValidating?: boolean;
  onSubmit: (values: ReviewFormValues) => Promise<void>;
  onBack: () => void;
  isSubmitting?: boolean;
};

export function CheckoutReviewStep({
  checkoutSession,
  shippingCountryName,
  billingCountryName,
  cartSummary,
  validation,
  isValidating,
  onSubmit,
  onBack,
  isSubmitting,
}: CheckoutReviewStepProps) {
  const form = useForm<ReviewFormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      terms_accepted: false,
      order_notes: checkoutSession?.order_notes || "",
    },
  });

  React.useEffect(() => {
    form.reset({
      terms_accepted: false,
      order_notes: checkoutSession?.order_notes || "",
    });
  }, [checkoutSession?.order_notes, form]);

  const issues = validation?.issues || [];
  const warnings = validation?.warnings || [];
  const hasBlockingIssues = Boolean(validation && !validation.is_valid && issues.length);
  const displayCountry = shippingCountryName || checkoutSession?.shipping_country;
  const billingCountryDisplay =
    billingCountryName || checkoutSession?.billing_country || displayCountry;
  const shippingName = `${checkoutSession?.shipping_first_name || ""} ${
    checkoutSession?.shipping_last_name || ""
  }`.trim();
  const billingName = `${checkoutSession?.billing_first_name || ""} ${
    checkoutSession?.billing_last_name || ""
  }`.trim();
  const hasBillingAddress =
    Boolean(checkoutSession?.billing_address_line_1) ||
    Boolean(checkoutSession?.billing_city) ||
    Boolean(checkoutSession?.billing_postal_code);
  const billingIsShipping =
    checkoutSession?.billing_same_as_shipping === true || !hasBillingAddress;

  const billingAddressLine = formatAddressLine(
    {
      address_line_1: billingIsShipping
        ? checkoutSession?.shipping_address_line_1
        : checkoutSession?.billing_address_line_1,
      address_line_2: billingIsShipping
        ? checkoutSession?.shipping_address_line_2
        : checkoutSession?.billing_address_line_2,
      city: billingIsShipping
        ? checkoutSession?.shipping_city
        : checkoutSession?.billing_city,
      state: billingIsShipping
        ? checkoutSession?.shipping_state
        : checkoutSession?.billing_state,
      postal_code: billingIsShipping
        ? checkoutSession?.shipping_postal_code
        : checkoutSession?.billing_postal_code,
      country: billingIsShipping ? displayCountry : billingCountryDisplay,
    },
    { countryName: billingIsShipping ? displayCountry : billingCountryDisplay }
  );

  const formattedShippingAddress = formatAddressLine(
    {
      address_line_1: checkoutSession?.shipping_address_line_1,
      address_line_2: checkoutSession?.shipping_address_line_2,
      city: checkoutSession?.shipping_city,
      state: checkoutSession?.shipping_state,
      postal_code: checkoutSession?.shipping_postal_code,
      country: displayCountry || checkoutSession?.shipping_country,
    },
    { countryName: displayCountry }
  );

  const feeCurrency =
    cartSummary?.currency_code || cartSummary?.currency || undefined;
  const formattedFee =
    cartSummary?.formatted_payment_fee ||
    formatMoney(
      cartSummary?.payment_fee_amount || checkoutSession?.payment_fee_amount,
      feeCurrency
    );
  return (
    <Card variant="bordered" className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-foreground/60">
          Step 4
        </p>
        <h2 className="text-xl font-semibold">Review & place order</h2>
        <p className="text-sm text-foreground/60">
          Confirm details before submitting your order.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-border bg-card p-4 text-sm">
          <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
            Shipping
          </p>
          <p className="mt-2 font-semibold">{shippingName || "Recipient"}</p>
          {formattedShippingAddress ? (
            <p className="text-foreground/70">{formattedShippingAddress}</p>
          ) : null}
        </div>
        <div className="rounded-xl border border-border bg-card p-4 text-sm">
          <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
            Billing
          </p>
          <p className="mt-2 font-semibold">
            {billingIsShipping ? shippingName || "Recipient" : billingName || "Billing contact"}
            {billingIsShipping ? (
              <span className="ml-2 text-xs text-foreground/60">
                (Same as shipping)
              </span>
            ) : null}
          </p>
          {billingAddressLine ? (
            <p className="text-foreground/70">{billingAddressLine}</p>
          ) : null}
        </div>
        <div className="rounded-xl border border-border bg-card p-4 text-sm">
          <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
            Payment
          </p>
          <p className="mt-2 font-semibold">
            {checkoutSession?.payment_method
              ? checkoutSession.payment_method.toUpperCase()
              : "Not selected"}
          </p>
          {checkoutSession?.payment_method ? (
            <p className="text-foreground/70">
              Fee: {formattedFee || "0.00"}
            </p>
          ) : null}
        </div>
      </div>

      <div className="space-y-2">
        <p className="text-sm font-semibold">Cart validation</p>
        {isValidating ? (
          <p className="text-sm text-foreground/60">Validating cart...</p>
        ) : validation ? (
          <>
            {issues.length ? (
              <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-100">
                <p className="font-semibold">Issues to fix</p>
                <ul className="mt-2 space-y-1 text-xs">
                  {issues.map((issue, index) => (
                    <li key={`${issue.type}-${index}`}>
                      {issue.message || "Resolve this issue to continue."}
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <p className="text-sm text-emerald-500">
                Cart is ready for checkout.
              </p>
            )}
            {warnings.length ? (
              <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-100">
                <p className="font-semibold">Warnings</p>
                <ul className="mt-2 space-y-1">
                  {warnings.map((warning, index) => (
                    <li key={`${warning.type}-${index}`}>
                      {warning.message || "Review this warning."}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </>
        ) : (
          <p className="text-sm text-foreground/60">
            Validation will run before placing your order.
          </p>
        )}
      </div>

      <form className="space-y-4" onSubmit={form.handleSubmit(onSubmit)}>
        <label className="block text-sm">
          Order notes (optional)
          <textarea
            rows={3}
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
            {...form.register("order_notes")}
          />
        </label>

        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" {...form.register("terms_accepted")} />
          I agree to the terms and conditions.
        </label>
        {form.formState.errors.terms_accepted ? (
          <p className="text-xs text-rose-500">
            {form.formState.errors.terms_accepted.message}
          </p>
        ) : null}

        <div className="flex flex-col gap-3 sm:flex-row sm:justify-between">
          <Button type="button" variant="secondary" onClick={onBack}>
            Back
          </Button>
          <Button
            type="submit"
            disabled={isSubmitting || hasBlockingIssues || Boolean(isValidating)}
          >
            {isSubmitting ? "Placing order..." : "Place order"}
          </Button>
        </div>
        {hasBlockingIssues ? (
          <p className="text-xs text-rose-500">
            Fix the issues above before placing your order.
          </p>
        ) : null}
      </form>
    </Card>
  );
}
