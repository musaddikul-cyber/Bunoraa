"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { AuthGate } from "@/components/auth/AuthGate";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { useAddresses } from "@/components/account/useAddresses";
import { CheckoutSteps } from "@/components/checkout/CheckoutSteps";
import { CheckoutInfoStep, CheckoutInfoFormValues } from "@/components/checkout/CheckoutInfoStep";
import { CheckoutShippingStep } from "@/components/checkout/CheckoutShippingStep";
import { CheckoutPaymentStep, CheckoutPaymentFormValues } from "@/components/checkout/CheckoutPaymentStep";
import { CheckoutReviewStep } from "@/components/checkout/CheckoutReviewStep";
import { CheckoutSummary } from "@/components/checkout/CheckoutSummary";
import { useCheckoutData } from "@/components/checkout/useCheckoutData";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useToast } from "@/components/ui/ToastProvider";
import { ApiError } from "@/lib/api";
import type { CheckoutValidation, ShippingMethodOption } from "@/lib/types";

const stepOrder = ["information", "shipping", "payment", "review"] as const;
type Step = (typeof stepOrder)[number];

const parseStep = (value: string | null): Step | null => {
  if (!value) return null;
  return stepOrder.includes(value as Step) ? (value as Step) : null;
};

export function CheckoutPage() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const { push } = useToast();
  const auth = useAuthContext();
  const { profileQuery, hasToken } = auth;
  const { addressesQuery } = useAddresses({ enabled: hasToken });

  const {
    checkoutQuery,
    cartQuery,
    cartSummaryQuery,
    countriesQuery,
    pickupLocationsQuery,
    paymentGatewaysQuery,
    savedPaymentMethodsQuery,
    updateShippingInfo,
    selectShippingMethod,
    selectPaymentMethod,
    completeCheckout,
    calculateShipping,
    validateCart,
    updateGiftOptions,
    applyCoupon,
    removeCoupon,
  } = useCheckoutData({ enabled: hasToken });

  const checkoutSession = checkoutQuery.data;
  const cart = cartQuery.data;
  const cartSummary = cartSummaryQuery.data;
  const profile = profileQuery.data;
  const countries = React.useMemo(() => countriesQuery.data ?? [], [countriesQuery.data]);
  const pickupLocations = React.useMemo(
    () => pickupLocationsQuery.data ?? [],
    [pickupLocationsQuery.data]
  );

  const resolveCountryName = React.useCallback(
    (value?: string | null) => {
      if (!value) return "";
      const trimmed = value.trim();
      if (!trimmed) return "";
      const byCode = countries.find(
        (country) => country.code.toLowerCase() === trimmed.toLowerCase()
      );
      if (byCode) return byCode.name;
      const byName = countries.find(
        (country) => country.name.toLowerCase() === trimmed.toLowerCase()
      );
      return byName?.name || trimmed;
    },
    [countries]
  );

  const [localMaxStep, setLocalMaxStep] = React.useState(0);
  const [maxStepInitialized, setMaxStepInitialized] = React.useState(false);
  const [validation, setValidation] = React.useState<CheckoutValidation | null>(null);
  const [shippingRates, setShippingRates] = React.useState<ShippingMethodOption[]>([]);
  const [shippingRatesError, setShippingRatesError] = React.useState<string | null>(null);
  const lastRatesKey = React.useRef<string>("");
  const lastValidationKey = React.useRef<string>("");
  const lastAutoShippingKey = React.useRef<string>("");
  const lastAutoPaymentKey = React.useRef<string>("");
  const [autoSavingShipping, setAutoSavingShipping] = React.useState(false);
  const [autoSavingPayment, setAutoSavingPayment] = React.useState(false);

  const cartEmpty = !cart || cart.item_count === 0;
  const isLoading =
    checkoutQuery.isLoading || cartQuery.isLoading || cartSummaryQuery.isLoading;

  const infoComplete = Boolean(
    checkoutSession?.email &&
      checkoutSession?.shipping_first_name &&
      checkoutSession?.shipping_last_name &&
      checkoutSession?.shipping_address_line_1 &&
      checkoutSession?.shipping_city &&
      checkoutSession?.shipping_postal_code &&
      checkoutSession?.shipping_country
  );
  const shippingComplete = Boolean(checkoutSession?.shipping_method);
  const paymentComplete = Boolean(checkoutSession?.payment_method);

  let maxStepIndex = 0;
  if (infoComplete) maxStepIndex = 1;
  if (shippingComplete) maxStepIndex = 2;
  if (paymentComplete) maxStepIndex = 3;

  React.useEffect(() => {
    if (maxStepInitialized) return;
    if (!checkoutSession) return;
    setLocalMaxStep(maxStepIndex);
    setMaxStepInitialized(true);
  }, [checkoutSession, maxStepIndex, maxStepInitialized]);

  const stepParam = parseStep(searchParams.get("step"));
  const allowedIndex = maxStepInitialized ? localMaxStep : maxStepIndex;
  const targetIndex = stepParam ? stepOrder.indexOf(stepParam) : allowedIndex;
  const clampedIndex = Math.min(targetIndex, allowedIndex);
  const currentStep = stepOrder[clampedIndex] || "information";

  const validationKey = React.useMemo(() => {
    if (currentStep !== "review") return "";
    return JSON.stringify({
      cartId: cart?.id || null,
      cartUpdatedAt: cart?.updated_at || null,
      summaryTotal: cartSummary?.total || null,
      summaryShipping: cartSummary?.shipping_cost || null,
      shippingMethod: checkoutSession?.shipping_method || null,
      paymentMethod: checkoutSession?.payment_method || null,
      giftWrap: checkoutSession?.gift_wrap || false,
      coupon:
        checkoutSession?.coupon_code ||
        cartSummary?.coupon_code ||
        cart?.coupon_code ||
        null,
    });
  }, [
    currentStep,
    cart?.id,
    cart?.updated_at,
    cartSummary?.total,
    cartSummary?.shipping_cost,
    cartSummary?.coupon_code,
    checkoutSession?.shipping_method,
    checkoutSession?.payment_method,
    checkoutSession?.gift_wrap,
    checkoutSession?.coupon_code,
    cart?.coupon_code,
  ]);

  const goToStep = React.useCallback(
    (step: Step) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set("step", step);
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [pathname, router, searchParams]
  );

  React.useEffect(() => {
    const params = new URLSearchParams(searchParams.toString());
    if (params.get("step") !== currentStep) {
      params.set("step", currentStep);
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    }
  }, [currentStep, pathname, router, searchParams]);

  const shippingPayload = React.useMemo(() => {
    if (!infoComplete || !cart || !cartSummary || !checkoutSession) return null;
    const countryName =
      resolveCountryName(checkoutSession.shipping_country) ||
      checkoutSession.shipping_country ||
      "";
    return {
      country: countryName,
      state: checkoutSession.shipping_state || "",
      city: checkoutSession.shipping_city || "",
      postal_code: checkoutSession.shipping_postal_code || "",
      subtotal: cartSummary.subtotal || cart.subtotal || "0",
      item_count: cart.item_count || 1,
      product_ids: cart.items?.map((item) => item.product_id) || [],
    };
  }, [infoComplete, cart, cartSummary, checkoutSession, resolveCountryName]);

  React.useEffect(() => {
    if (!shippingPayload) return;
    if (!["shipping", "payment", "review"].includes(currentStep)) return;
    const key = JSON.stringify(shippingPayload);
    if (key === lastRatesKey.current) return;
    lastRatesKey.current = key;
    calculateShipping.mutate(shippingPayload, {
      onSuccess: (data) => {
        setShippingRates(data.methods || []);
        setShippingRatesError(null);
      },
      onError: (error) => {
        setShippingRates([]);
        setShippingRatesError(
          error instanceof Error ? error.message : "Failed to load shipping rates."
        );
      },
    });
  }, [shippingPayload, currentStep, calculateShipping]);

  React.useEffect(() => {
    if (currentStep !== "review") return;
    if (!validationKey) return;
    if (validateCart.isPending) return;
    if (validationKey === lastValidationKey.current) return;
    lastValidationKey.current = validationKey;
    validateCart.mutate(undefined, {
      onSuccess: (data) => setValidation(data),
      onError: () => setValidation(null),
    });
  }, [currentStep, validationKey, validateCart]);

  const handleAutoSaveShipping = React.useCallback(
    (payload: {
      shipping_type: "delivery" | "pickup";
      shipping_rate_id?: string;
      pickup_location_id?: string;
    }) => {
      const key = JSON.stringify(payload);
      if (key === lastAutoShippingKey.current) return;
      lastAutoShippingKey.current = key;
      setAutoSavingShipping(true);
      selectShippingMethod.mutate(payload, {
        onSettled: () => setAutoSavingShipping(false),
      });
    },
    [selectShippingMethod]
  );

  const handleAutoSavePayment = React.useCallback(
    (paymentMethod: string) => {
      if (!paymentMethod) return;
      if (paymentMethod === lastAutoPaymentKey.current) return;
      lastAutoPaymentKey.current = paymentMethod;
      setAutoSavingPayment(true);
      selectPaymentMethod.mutate(
        { payment_method: paymentMethod },
        { onSettled: () => setAutoSavingPayment(false) }
      );
    },
    [selectPaymentMethod]
  );

  const fallbackCountry = React.useMemo(() => {
    const resolved = resolveCountryName(checkoutSession?.shipping_country);
    if (resolved) return resolved;
    const defaultCountry =
      countries.find((country) => country.code === "BD") ||
      countries.find(
        (country) => country.name.toLowerCase() === "bangladesh"
      );
    return defaultCountry?.name || countries[0]?.name || "";
  }, [checkoutSession?.shipping_country, countries, resolveCountryName]);

  const shippingCountryName =
    resolveCountryName(checkoutSession?.shipping_country) || fallbackCountry;
  const billingCountryName =
    resolveCountryName(checkoutSession?.billing_country) || fallbackCountry;

  const infoDefaults = React.useMemo<Partial<CheckoutInfoFormValues>>(
    () => ({
      email: checkoutSession?.email || profile?.email || "",
      shipping_first_name:
        checkoutSession?.shipping_first_name || profile?.first_name || "",
      shipping_last_name:
        checkoutSession?.shipping_last_name || profile?.last_name || "",
      shipping_phone: checkoutSession?.shipping_phone || profile?.phone || "",
      shipping_address_line_1: checkoutSession?.shipping_address_line_1 || "",
      shipping_address_line_2: checkoutSession?.shipping_address_line_2 || "",
      shipping_city: checkoutSession?.shipping_city || "",
      shipping_state: checkoutSession?.shipping_state || "",
      shipping_postal_code: checkoutSession?.shipping_postal_code || "",
      shipping_country:
        resolveCountryName(checkoutSession?.shipping_country) || fallbackCountry,
      save_address: false,
    }),
    [
      checkoutSession?.email,
      checkoutSession?.shipping_first_name,
      checkoutSession?.shipping_last_name,
      checkoutSession?.shipping_phone,
      checkoutSession?.shipping_address_line_1,
      checkoutSession?.shipping_address_line_2,
      checkoutSession?.shipping_city,
      checkoutSession?.shipping_state,
      checkoutSession?.shipping_postal_code,
      checkoutSession?.shipping_country,
      profile?.email,
      profile?.first_name,
      profile?.last_name,
      profile?.phone,
      fallbackCountry,
      resolveCountryName,
    ]
  );

  const paymentDefaults = React.useMemo<Partial<CheckoutPaymentFormValues>>(
    () => ({
      payment_method: checkoutSession?.payment_method || "",
      billing_same_as_shipping:
        checkoutSession?.billing_same_as_shipping ?? true,
      billing_first_name: checkoutSession?.billing_first_name || "",
      billing_last_name: checkoutSession?.billing_last_name || "",
      billing_address_line_1: checkoutSession?.billing_address_line_1 || "",
      billing_address_line_2: checkoutSession?.billing_address_line_2 || "",
      billing_city: checkoutSession?.billing_city || "",
      billing_state: checkoutSession?.billing_state || "",
      billing_postal_code: checkoutSession?.billing_postal_code || "",
      billing_country:
        resolveCountryName(checkoutSession?.billing_country) || fallbackCountry,
    }),
    [
      checkoutSession?.payment_method,
      checkoutSession?.billing_same_as_shipping,
      checkoutSession?.billing_first_name,
      checkoutSession?.billing_last_name,
      checkoutSession?.billing_address_line_1,
      checkoutSession?.billing_address_line_2,
      checkoutSession?.billing_city,
      checkoutSession?.billing_state,
      checkoutSession?.billing_postal_code,
      checkoutSession?.billing_country,
      fallbackCountry,
      resolveCountryName,
    ]
  );

  const shippingToBillingDefaults = React.useMemo<Partial<CheckoutPaymentFormValues>>(
    () => ({
      billing_first_name: checkoutSession?.shipping_first_name || "",
      billing_last_name: checkoutSession?.shipping_last_name || "",
      billing_address_line_1: checkoutSession?.shipping_address_line_1 || "",
      billing_address_line_2: checkoutSession?.shipping_address_line_2 || "",
      billing_city: checkoutSession?.shipping_city || "",
      billing_state: checkoutSession?.shipping_state || "",
      billing_postal_code: checkoutSession?.shipping_postal_code || "",
      billing_country:
        resolveCountryName(checkoutSession?.shipping_country) || fallbackCountry,
    }),
    [
      checkoutSession?.shipping_first_name,
      checkoutSession?.shipping_last_name,
      checkoutSession?.shipping_address_line_1,
      checkoutSession?.shipping_address_line_2,
      checkoutSession?.shipping_city,
      checkoutSession?.shipping_state,
      checkoutSession?.shipping_postal_code,
      checkoutSession?.shipping_country,
      fallbackCountry,
      resolveCountryName,
    ]
  );

  const handleInfoSubmit = async (values: CheckoutInfoFormValues) => {
    try {
      const result = await updateShippingInfo.mutateAsync(values);
      const payload =
        result && typeof result === "object" && "data" in result
          ? (result as { data?: Record<string, unknown> }).data
          : null;
      const addressSaved = payload?.address_saved;
      const addressError = payload?.address_save_error;
      if (addressSaved === false && typeof addressError === "string" && addressError.trim()) {
        push(addressError, "error");
      }
      setLocalMaxStep((prev) => Math.max(prev, 1));
      goToStep("shipping");
    } catch (error) {
      push(
        error instanceof Error ? error.message : "Could not save shipping info.",
        "error"
      );
    }
  };

  const handleShippingSubmit = async (payload: {
    shipping_type: "delivery" | "pickup";
    shipping_rate_id?: string;
    pickup_location_id?: string;
    delivery_instructions?: string;
  }) => {
    try {
      await selectShippingMethod.mutateAsync(payload);
      setLocalMaxStep((prev) => Math.max(prev, 2));
      goToStep("payment");
    } catch (error) {
      push(
        error instanceof Error ? error.message : "Could not save shipping method.",
        "error"
      );
    }
  };

  const handlePaymentSubmit = async (values: CheckoutPaymentFormValues) => {
    try {
      await selectPaymentMethod.mutateAsync(values);
      setLocalMaxStep((prev) => Math.max(prev, 3));
      goToStep("review");
    } catch (error) {
      push(
        error instanceof Error ? error.message : "Could not save payment method.",
        "error"
      );
    }
  };

  const handleReviewSubmit = async (values: { terms_accepted: boolean; order_notes?: string }) => {
    try {
      let validationResult = validation;
      if (!validationResult) {
        validationResult = await validateCart.mutateAsync();
        setValidation(validationResult);
      }
      if (validationResult && !validationResult.is_valid && validationResult.issues.length) {
        push("Resolve the checkout issues before placing the order.", "error");
        return;
      }
      const result = await completeCheckout.mutateAsync(values);
      const payload = result && typeof result === "object" && "data" in result
        ? (result as { data?: Record<string, unknown> }).data
        : null;
      const redirectUrl = payload?.payment_redirect_url || payload?.redirect_url;
      if (typeof redirectUrl === "string" && redirectUrl.trim()) {
        window.location.href = redirectUrl;
        return;
      }
      const orderId = payload?.order_id as string | undefined;
      const orderNumber = payload?.order_number as string | undefined;
      if (orderId || orderNumber) {
        const params = new URLSearchParams();
        if (orderId) params.set("order_id", orderId);
        if (orderNumber) params.set("order_number", orderNumber);
        router.push(`/checkout/success?${params.toString()}`);
      } else {
        router.push("/orders/");
      }
    } catch (error) {
      push(
        error instanceof Error ? error.message : "Could not place the order.",
        "error"
      );
    }
  };

  if (isLoading) {
    return (
      <AuthGate nextHref="/checkout">
        <div className="mx-auto w-full max-w-6xl px-4 sm:px-6 py-16">
          <Card variant="bordered" className="space-y-4">
            <div className="h-6 w-48 rounded bg-muted animate-pulse" />
            <div className="h-4 w-full rounded bg-muted animate-pulse" />
            <div className="h-10 w-full rounded bg-muted animate-pulse" />
          </Card>
        </div>
      </AuthGate>
    );
  }

  if (checkoutQuery.isError) {
    const error = checkoutQuery.error;
    if (error instanceof ApiError && error.status === 400) {
      return (
        <AuthGate nextHref="/checkout">
          <div className="mx-auto w-full max-w-5xl px-4 sm:px-6 py-16">
            <Card variant="bordered" className="space-y-3 text-center">
              <h1 className="text-2xl font-semibold">Your cart is empty</h1>
              <p className="text-sm text-foreground/60">
                Add items to your cart before checking out.
              </p>
              <Button asChild>
                <Link href="/cart/">Go to cart</Link>
              </Button>
            </Card>
          </div>
        </AuthGate>
      );
    }
    return (
      <AuthGate nextHref="/checkout">
        <div className="mx-auto w-full max-w-5xl px-4 sm:px-6 py-16">
          <Card variant="bordered" className="space-y-3 text-center">
            <h1 className="text-2xl font-semibold">Checkout unavailable</h1>
            <p className="text-sm text-foreground/60">
              We couldn&apos;t load checkout right now. Please try again.
            </p>
            <Button asChild>
              <Link href="/cart/">Back to cart</Link>
            </Button>
          </Card>
        </div>
      </AuthGate>
    );
  }

  if (cartEmpty) {
    return (
      <AuthGate nextHref="/checkout">
        <div className="mx-auto w-full max-w-5xl px-4 sm:px-6 py-16">
          <Card variant="bordered" className="space-y-3 text-center">
            <h1 className="text-2xl font-semibold">Your cart is empty</h1>
            <p className="text-sm text-foreground/60">
              Add items to your cart before checking out.
            </p>
            <Button asChild>
              <Link href="/cart/">Go to cart</Link>
            </Button>
          </Card>
        </div>
      </AuthGate>
    );
  }

  return (
    <AuthGate
      nextHref="/checkout"
      title="Sign in to checkout"
      description="Please sign in to continue with checkout."
    >
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6 sm:py-12">
          <div className="mb-8">
            <p className="text-xs uppercase tracking-[0.3em] text-foreground/60">
              Checkout
            </p>
            <h1 className="text-3xl font-semibold">Secure checkout</h1>
            <p className="mt-2 max-w-2xl text-sm text-foreground/60">
              Complete your order in a few quick steps. Your information is
              protected with encrypted connections.
            </p>
          </div>

          <CheckoutSteps current={currentStep} onStepClick={goToStep} />

          <div className="grid gap-6 lg:gap-8 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="space-y-6">
              {currentStep === "information" ? (
                <CheckoutInfoStep
                  defaultValues={infoDefaults}
                  countries={countries}
                  savedAddresses={addressesQuery.data || []}
                  allowSaveAddress={Boolean(hasToken)}
                  onSubmit={handleInfoSubmit}
                  isSubmitting={updateShippingInfo.isPending}
                />
              ) : null}

              {currentStep === "shipping" ? (
                <CheckoutShippingStep
                  shippingRates={shippingRates}
                  shippingRatesLoading={calculateShipping.isPending}
                  shippingRatesError={shippingRatesError}
                  pickupLocations={pickupLocations}
                  defaultShippingType={
                    checkoutSession?.shipping_method === "pickup" ? "pickup" : "delivery"
                  }
                  defaultRateId={cartSummary?.shipping_rate_id || ""}
                  defaultMethodCode={checkoutSession?.shipping_method || ""}
                  defaultPickupId={
                    cartSummary?.pickup_location_id ||
                    checkoutSession?.pickup_location?.id ||
                    ""
                  }
                  defaultInstructions={checkoutSession?.delivery_instructions || ""}
                  currencyCode={
                    cartSummary?.currency_code || cartSummary?.currency || ""
                  }
                  onSubmit={handleShippingSubmit}
                  onSelectionChange={handleAutoSaveShipping}
                  onBack={() => goToStep("information")}
                  isSubmitting={selectShippingMethod.isPending}
                  isAutoSaving={autoSavingShipping}
                />
              ) : null}

              {currentStep === "payment" ? (
              <CheckoutPaymentStep
                gateways={paymentGatewaysQuery.data || []}
                savedMethods={savedPaymentMethodsQuery.data || []}
                countries={countries}
                defaultValues={paymentDefaults}
                shippingDefaults={shippingToBillingDefaults}
                currencyCode={
                  cartSummary?.currency_code || cartSummary?.currency || ""
                }
                onSubmit={handlePaymentSubmit}
                onSelectionChange={handleAutoSavePayment}
                onBack={() => goToStep("shipping")}
                isSubmitting={selectPaymentMethod.isPending}
                isLoadingGateways={paymentGatewaysQuery.isLoading}
                isAutoSaving={autoSavingPayment}
                />
              ) : null}

              {currentStep === "review" ? (
                <CheckoutReviewStep
                  checkoutSession={checkoutSession}
                  shippingCountryName={shippingCountryName}
                  billingCountryName={billingCountryName}
                  cartSummary={cartSummary}
                  validation={validation}
                  isValidating={validateCart.isPending}
                  onSubmit={handleReviewSubmit}
                  onBack={() => goToStep("payment")}
                  isSubmitting={completeCheckout.isPending}
                />
              ) : null}
            </div>

            <div className="lg:sticky lg:top-24 lg:self-start">
              <CheckoutSummary
                cart={cart}
                cartSummary={cartSummary}
                checkoutSession={checkoutSession}
                onApplyCoupon={(code) => applyCoupon.mutateAsync(code)}
                onRemoveCoupon={() => removeCoupon.mutateAsync()}
                onUpdateGift={(payload) => updateGiftOptions.mutateAsync(payload)}
                isUpdatingGift={updateGiftOptions.isPending}
                isApplyingCoupon={applyCoupon.isPending}
                isRemovingCoupon={removeCoupon.isPending}
              />
            </div>
          </div>
        </div>
      </div>
    </AuthGate>
  );
}
