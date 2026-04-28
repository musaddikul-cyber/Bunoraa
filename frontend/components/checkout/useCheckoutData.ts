import * as React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type {
  Cart,
  CartSummary,
  CheckoutSession,
  Country,
  ShippingRateResponse,
  PaymentGateway,
  SavedPaymentMethod,
  CheckoutValidation,
  GiftOptionsResponse,
  StoreLocation,
} from "@/lib/types";

type GatewayParams = {
  currency?: string | null;
  country?: string | null;
  amount?: string | number | null;
};

function extractCartFromResponse(response: unknown): Cart | null {
  if (!response || typeof response !== "object") return null;

  const root = response as Record<string, unknown>;
  if (root.cart && typeof root.cart === "object") {
    return root.cart as Cart;
  }

  const payload = root.data;
  if (payload && typeof payload === "object") {
    const payloadObj = payload as Record<string, unknown>;
    if (payloadObj.cart && typeof payloadObj.cart === "object") {
      return payloadObj.cart as Cart;
    }
  }

  return null;
}

function mergeSummaryWithCart(previous: CartSummary | undefined, cart: Cart): CartSummary {
  const base: CartSummary = previous ?? {
    id: cart.id,
    item_count: cart.item_count,
    subtotal: cart.subtotal,
    discount_amount: cart.discount_amount,
    total: cart.total,
    coupon_code: cart.coupon_code ?? null,
    currency: cart.currency,
    currency_code: cart.currency,
  };

  return {
    ...base,
    id: base.id || cart.id,
    item_count: cart.item_count,
    subtotal: cart.subtotal,
    discount_amount: cart.discount_amount,
    total: cart.total,
    coupon_code: cart.coupon_code ?? null,
    currency: base.currency || cart.currency,
    currency_code: base.currency_code || cart.currency,
  };
}

async function fetchCheckoutSession() {
  const response = await apiFetch<CheckoutSession>("/commerce/checkout/", {
    allowGuest: true,
    suppressErrorStatus: [400],
  });
  return response.data;
}

async function fetchCart() {
  const response = await apiFetch<Cart>("/commerce/cart/", { allowGuest: true });
  return response.data;
}

async function fetchCartSummary() {
  const response = await apiFetch<CartSummary>("/commerce/cart/summary/", {
    allowGuest: true,
  });
  return response.data;
}

async function fetchCountries() {
  const response = await apiFetch<Country[]>("/i18n/countries/");
  return response.data;
}

async function fetchPickupLocations() {
  const response = await apiFetch<StoreLocation[]>(
    "/contacts/locations/pickup/"
  );
  return response.data;
}

async function fetchPaymentGateways(params?: GatewayParams) {
  const response = await apiFetch<PaymentGateway[]>(
    "/payments/gateways/available/",
    {
      params: {
        currency: params?.currency || undefined,
        country: params?.country || undefined,
        amount: params?.amount || undefined,
      },
    }
  );
  return response.data;
}

async function fetchSavedPaymentMethods() {
  const response = await apiFetch<SavedPaymentMethod[]>("/payments/methods/");
  return response.data;
}

export function useCheckoutData(options?: {
  gatewayParams?: GatewayParams;
  enablePaymentMethods?: boolean;
  enabled?: boolean;
}) {
  const queryClient = useQueryClient();
  const enabled = options?.enabled ?? true;

  const checkoutQuery = useQuery({
    queryKey: ["checkout", "session"],
    queryFn: fetchCheckoutSession,
    retry: false,
    enabled,
  });

  const cartQuery = useQuery({
    queryKey: ["cart"],
    queryFn: fetchCart,
    enabled,
  });

  const cartSummaryQuery = useQuery({
    queryKey: ["cart", "summary"],
    queryFn: fetchCartSummary,
    enabled,
  });

  const derivedGatewayParams = React.useMemo(() => {
    return (
      options?.gatewayParams || {
        currency:
          cartSummaryQuery.data?.currency_code ||
          cartSummaryQuery.data?.currency ||
          cartQuery.data?.currency ||
          undefined,
        amount:
          cartSummaryQuery.data?.total ||
          cartQuery.data?.total ||
          undefined,
      }
    );
  }, [
    options?.gatewayParams,
    cartSummaryQuery.data?.currency_code,
    cartSummaryQuery.data?.currency,
    cartSummaryQuery.data?.total,
    cartQuery.data?.currency,
    cartQuery.data?.total,
  ]);

  const countriesQuery = useQuery({
    queryKey: ["i18n", "countries"],
    queryFn: fetchCountries,
    enabled,
  });

  const pickupLocationsQuery = useQuery({
    queryKey: ["contacts", "pickup"],
    queryFn: fetchPickupLocations,
    enabled,
  });

  const paymentGatewaysQuery = useQuery({
    queryKey: ["payments", "gateways", derivedGatewayParams],
    queryFn: () => fetchPaymentGateways(derivedGatewayParams),
    enabled: Boolean(
      enabled &&
        (derivedGatewayParams?.currency ||
          derivedGatewayParams?.country ||
          derivedGatewayParams?.amount)
    ),
  });

  const savedPaymentMethodsQuery = useQuery({
    queryKey: ["payments", "methods"],
    queryFn: fetchSavedPaymentMethods,
    enabled: enabled && (options?.enablePaymentMethods ?? true),
  });

  const updateShippingInfo = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      return apiFetch("/commerce/checkout/shipping_info/", {
        method: "POST",
        body: payload,
        allowGuest: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["checkout", "session"] });
      queryClient.invalidateQueries({ queryKey: ["cart", "summary"] });
      queryClient.invalidateQueries({ queryKey: ["addresses"] });
    },
  });

  const selectShippingMethod = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      return apiFetch("/commerce/checkout/shipping_method/", {
        method: "POST",
        body: payload,
        allowGuest: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["checkout", "session"] });
      queryClient.invalidateQueries({ queryKey: ["cart", "summary"] });
    },
  });

  const selectPaymentMethod = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      return apiFetch("/commerce/checkout/payment_method/", {
        method: "POST",
        body: payload,
        allowGuest: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["checkout", "session"] });
      queryClient.invalidateQueries({ queryKey: ["cart", "summary"] });
    },
  });

  const completeCheckout = useMutation({
    mutationFn: async (payload?: Record<string, unknown>) => {
      return apiFetch("/commerce/checkout/complete/", {
        method: "POST",
        body: payload,
        allowGuest: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["cart"] });
      queryClient.invalidateQueries({ queryKey: ["cart", "summary"] });
    },
  });

  const calculateShipping = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      const response = await apiFetch<ShippingRateResponse>("/shipping/calculate/", {
        method: "POST",
        body: payload,
      });
      return response.data;
    },
  });

  const validateCart = useMutation({
    mutationFn: async () => {
      const response = await apiFetch<CheckoutValidation>("/commerce/cart/validate/", {
        method: "POST",
        allowGuest: true,
        suppressErrorStatus: [400],
      });
      return response.data;
    },
  });

  const updateGiftOptions = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      const response = await apiFetch<GiftOptionsResponse>("/commerce/cart/gift/", {
        method: "POST",
        body: payload,
        allowGuest: true,
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["cart", "summary"] });
      queryClient.invalidateQueries({ queryKey: ["checkout", "session"] });
    },
  });

  const applyCoupon = useMutation({
    mutationFn: async (code: string) => {
      return apiFetch("/promotions/coupons/apply/", {
        method: "POST",
        body: { code },
        allowGuest: true,
        suppressErrorStatus: [400],
      });
    },
    onSuccess: (response) => {
      const nextCart = extractCartFromResponse(response);
      if (nextCart) {
        queryClient.setQueryData<Cart>(["cart"], nextCart);
        queryClient.setQueryData<CartSummary>(["cart", "summary"], (previous) =>
          mergeSummaryWithCart(previous, nextCart)
        );
      }
      queryClient.invalidateQueries({ queryKey: ["cart"] });
      queryClient.invalidateQueries({ queryKey: ["cart", "summary"] });
      queryClient.invalidateQueries({ queryKey: ["checkout", "session"] });
    },
  });

  const removeCoupon = useMutation({
    mutationFn: async () => {
      return apiFetch("/commerce/cart/remove_coupon/", {
        method: "POST",
        allowGuest: true,
      });
    },
    onSuccess: (response) => {
      const nextCart = extractCartFromResponse(response);
      if (nextCart) {
        queryClient.setQueryData<Cart>(["cart"], nextCart);
        queryClient.setQueryData<CartSummary>(["cart", "summary"], (previous) =>
          mergeSummaryWithCart(previous, nextCart)
        );
      }
      queryClient.invalidateQueries({ queryKey: ["cart"] });
      queryClient.invalidateQueries({ queryKey: ["cart", "summary"] });
      queryClient.invalidateQueries({ queryKey: ["checkout", "session"] });
    },
  });

  return {
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
  };
}
