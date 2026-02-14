"use client";

import * as React from "react";
import { useForm, useWatch } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";
import { formatMoney } from "@/lib/checkout";
import type { Country, PaymentGateway, SavedPaymentMethod } from "@/lib/types";

const schema = z
  .object({
    payment_method: z.string().min(1, "Select a payment method"),
    billing_same_as_shipping: z.boolean().optional(),
    billing_first_name: z.string().optional(),
    billing_last_name: z.string().optional(),
    billing_address_line_1: z.string().optional(),
    billing_address_line_2: z.string().optional(),
    billing_city: z.string().optional(),
    billing_state: z.string().optional(),
    billing_postal_code: z.string().optional(),
    billing_country: z.string().optional(),
  })
  .superRefine((data, ctx) => {
    if (data.billing_same_as_shipping !== false) return;
    const requiredFields: Array<keyof typeof data> = [
      "billing_first_name",
      "billing_last_name",
      "billing_address_line_1",
      "billing_city",
      "billing_postal_code",
      "billing_country",
    ];
    requiredFields.forEach((field) => {
      const value = data[field];
      if (!value || (typeof value === "string" && !value.trim())) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "This field is required",
          path: [field],
        });
      }
    });
  });

export type CheckoutPaymentFormValues = z.infer<typeof schema>;

type CheckoutPaymentStepProps = {
  gateways: PaymentGateway[];
  savedMethods: SavedPaymentMethod[];
  countries?: Country[];
  defaultValues: Partial<CheckoutPaymentFormValues>;
  shippingDefaults?: Partial<CheckoutPaymentFormValues>;
  currencyCode?: string;
  onSubmit: (values: CheckoutPaymentFormValues) => Promise<void>;
  onSelectionChange?: (paymentMethod: string) => void;
  onBack: () => void;
  isSubmitting?: boolean;
  isLoadingGateways?: boolean;
  isAutoSaving?: boolean;
};

export function CheckoutPaymentStep({
  gateways,
  savedMethods,
  countries,
  defaultValues,
  shippingDefaults,
  currencyCode,
  onSubmit,
  onSelectionChange,
  onBack,
  isSubmitting,
  isLoadingGateways,
  isAutoSaving,
}: CheckoutPaymentStepProps) {
  const form = useForm<CheckoutPaymentFormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      billing_same_as_shipping: true,
      ...defaultValues,
    },
  });
  const [selectedPayment, setSelectedPayment] = React.useState(
    defaultValues.payment_method || ""
  );

  React.useEffect(() => {
    form.reset({
      billing_same_as_shipping: true,
      ...defaultValues,
    });
    setSelectedPayment(defaultValues.payment_method || "");
  }, [defaultValues, form]);

  React.useEffect(() => {
    if (!gateways.length) return;
    const current = form.getValues("payment_method");
    const hasCurrent =
      Boolean(current) && gateways.some((gateway) => gateway.code === current);
    if (hasCurrent) {
      setSelectedPayment(current);
      return;
    }
    form.setValue("payment_method", gateways[0].code, { shouldValidate: true });
    setSelectedPayment(gateways[0].code);
  }, [gateways, form]);

  React.useEffect(() => {
    if (!onSelectionChange) return;
    if (!selectedPayment) return;
    onSelectionChange(selectedPayment);
  }, [selectedPayment, onSelectionChange]);

  const billingSame = useWatch({
    control: form.control,
    name: "billing_same_as_shipping",
  });
  const gatewaysAvailable = gateways.length > 0;
  const sortedCountries = React.useMemo(() => {
    const list = countries ? [...countries] : [];
    return list.sort((a, b) => a.name.localeCompare(b.name));
  }, [countries]);

  const handleResetBilling = React.useCallback(() => {
    const current = form.getValues();
    const resetValues: CheckoutPaymentFormValues = {
      ...current,
      billing_first_name: defaultValues.billing_first_name || "",
      billing_last_name: defaultValues.billing_last_name || "",
      billing_address_line_1: defaultValues.billing_address_line_1 || "",
      billing_address_line_2: defaultValues.billing_address_line_2 || "",
      billing_city: defaultValues.billing_city || "",
      billing_state: defaultValues.billing_state || "",
      billing_postal_code: defaultValues.billing_postal_code || "",
      billing_country: defaultValues.billing_country || "",
    };
    form.reset(resetValues);
  }, [form, defaultValues]);

  React.useEffect(() => {
    if (!billingSame) return;
    if (!shippingDefaults) return;
    const fields: Array<keyof CheckoutPaymentFormValues> = [
      "billing_first_name",
      "billing_last_name",
      "billing_address_line_1",
      "billing_address_line_2",
      "billing_city",
      "billing_state",
      "billing_postal_code",
      "billing_country",
    ];
    fields.forEach((field) => {
      const value = shippingDefaults[field];
      if (value === undefined) return;
      form.setValue(field, (value as string) || "", {
        shouldValidate: false,
        shouldDirty: false,
      });
    });
  }, [billingSame, shippingDefaults, form]);

  const renderError = (name: keyof CheckoutPaymentFormValues) => {
    const error = form.formState.errors[name];
    if (!error) return null;
    return (
      <p className="mt-1 text-xs text-rose-500" id={`${name}-error`}>
        {error.message}
      </p>
    );
  };

  return (
    <Card variant="bordered" className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-foreground/60">
          Step 3
        </p>
        <h2 className="text-xl font-semibold">Payment method</h2>
        <p className="text-sm text-foreground/60">
          Select how you want to pay. We will not collect card details here.
        </p>
      </div>

      <form className="space-y-6" onSubmit={form.handleSubmit(onSubmit)}>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <p className="font-semibold">Available gateways</p>
            {isLoadingGateways ? (
              <span className="text-xs text-foreground/60">Loading...</span>
            ) : isAutoSaving ? (
              <span className="text-xs text-foreground/60">Saving...</span>
            ) : null}
          </div>
          <p className="text-xs text-foreground/60">
            Showing gateways based on your selected currency.
          </p>
          {gateways.length ? (
            <div className="space-y-2">
              {gateways.map((gateway) => {
                const paymentField = form.register("payment_method");
                const isSelected = selectedPayment === gateway.code;
                return (
                  <label
                    key={gateway.code}
                    className={cn(
                      "flex cursor-pointer flex-col gap-3 rounded-xl border px-4 py-3 text-sm sm:flex-row sm:items-start sm:justify-between",
                      isSelected
                        ? "border-primary bg-primary/10"
                        : "border-border bg-card hover:bg-muted"
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <input
                        type="radio"
                        className="mt-1 h-4 w-4"
                        value={gateway.code}
                        {...paymentField}
                        onChange={(event) => {
                          paymentField.onChange(event);
                          setSelectedPayment(event.target.value);
                        }}
                      />
                      <div>
                        <p className="font-semibold">{gateway.name}</p>
                        {gateway.description ? (
                          <p className="text-xs text-foreground/60">
                            {gateway.description}
                          </p>
                        ) : null}
                        {gateway.instructions ? (
                          <p className="text-xs text-foreground/60">
                            {gateway.instructions}
                          </p>
                        ) : null}
                      </div>
                    </div>
                    <div className="self-end text-right text-xs text-foreground/60 sm:self-auto">
                      {gateway.fee_amount_converted !== null &&
                      gateway.fee_amount_converted !== undefined ? (
                        <p>
                          Fee{" "}
                          {formatMoney(
                            gateway.fee_amount_converted,
                            currencyCode
                          )}
                        </p>
                      ) : gateway.fee_text ? (
                        <p>{gateway.fee_text}</p>
                      ) : null}
                    </div>
                  </label>
                );
              })}
            </div>
          ) : (
            <p className="text-sm text-foreground/60">
              No payment gateways available for this order.
            </p>
          )}
          {renderError("payment_method")}
        </div>

        {savedMethods.length ? (
          <div className="space-y-2 rounded-xl border border-border bg-muted/40 p-4 text-sm">
            <p className="font-semibold">Saved payment methods</p>
            <div className="space-y-1 text-xs text-foreground/70">
              {savedMethods.map((method) => (
                <div key={method.id} className="flex items-center justify-between">
                  <span>
                    {method.display_name ||
                      method.type_display ||
                      method.type ||
                      "Payment method"}
                    {method.card_last_four ? ` **** ${method.card_last_four}` : ""}
                  </span>
                  {method.is_default ? (
                    <span className="text-primary">Default</span>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div className="border-t border-border pt-6">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                {...form.register("billing_same_as_shipping")}
              />
              Billing address is same as shipping
            </label>
            {!billingSame ? (
              <Button type="button" variant="secondary" onClick={handleResetBilling}>
                Reset billing
              </Button>
            ) : null}
          </div>
        </div>

        {!billingSame ? (
          <div className="grid gap-4 md:grid-cols-2">
            <label className="block text-sm">
              First name
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                autoComplete="billing given-name"
                {...form.register("billing_first_name")}
                aria-invalid={Boolean(form.formState.errors.billing_first_name)}
                aria-describedby={
                  form.formState.errors.billing_first_name
                    ? "billing_first_name-error"
                    : undefined
                }
              />
              {renderError("billing_first_name")}
            </label>
            <label className="block text-sm">
              Last name
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                autoComplete="billing family-name"
                {...form.register("billing_last_name")}
                aria-invalid={Boolean(form.formState.errors.billing_last_name)}
                aria-describedby={
                  form.formState.errors.billing_last_name
                    ? "billing_last_name-error"
                    : undefined
                }
              />
              {renderError("billing_last_name")}
            </label>
            <label className="block text-sm md:col-span-2">
              Address line 1
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                autoComplete="billing address-line1"
                {...form.register("billing_address_line_1")}
                aria-invalid={Boolean(form.formState.errors.billing_address_line_1)}
                aria-describedby={
                  form.formState.errors.billing_address_line_1
                    ? "billing_address_line_1-error"
                    : undefined
                }
              />
              {renderError("billing_address_line_1")}
            </label>
            <label className="block text-sm md:col-span-2">
              Address line 2 (optional)
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                autoComplete="billing address-line2"
                {...form.register("billing_address_line_2")}
              />
            </label>
            <label className="block text-sm">
              City
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                autoComplete="billing address-level2"
                {...form.register("billing_city")}
                aria-invalid={Boolean(form.formState.errors.billing_city)}
                aria-describedby={
                  form.formState.errors.billing_city
                    ? "billing_city-error"
                    : undefined
                }
              />
              {renderError("billing_city")}
            </label>
            <label className="block text-sm">
              State / Province
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                autoComplete="billing address-level1"
                {...form.register("billing_state")}
              />
            </label>
            <label className="block text-sm">
              Postal code
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                autoComplete="billing postal-code"
                inputMode="text"
                {...form.register("billing_postal_code")}
                aria-invalid={Boolean(form.formState.errors.billing_postal_code)}
                aria-describedby={
                  form.formState.errors.billing_postal_code
                    ? "billing_postal_code-error"
                    : undefined
                }
              />
              {renderError("billing_postal_code")}
            </label>
            <label className="block text-sm">
              Country
              <select
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                autoComplete="billing country-name"
                {...form.register("billing_country")}
                aria-invalid={Boolean(form.formState.errors.billing_country)}
                aria-describedby={
                  form.formState.errors.billing_country
                    ? "billing_country-error"
                    : undefined
                }
              >
                <option value="">Select country</option>
                {sortedCountries.map((country) => (
                  <option key={country.code} value={country.name}>
                    {country.flag_emoji ? `${country.flag_emoji} ` : ""}
                    {country.name}
                  </option>
                ))}
              </select>
              {renderError("billing_country")}
            </label>
          </div>
        ) : null}

        <div className="flex flex-col gap-3 sm:flex-row sm:justify-between">
          <Button type="button" variant="secondary" className="w-full sm:w-auto" onClick={onBack}>
            Back
          </Button>
          <Button
            type="submit"
            className="w-full sm:w-auto sm:min-w-[220px]"
            disabled={isSubmitting || !gatewaysAvailable}
          >
            {isSubmitting ? "Saving..." : "Continue to review"}
          </Button>
        </div>
      </form>
    </Card>
  );
}
