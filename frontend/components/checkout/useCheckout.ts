import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { CheckoutSession } from "@/lib/types";

const checkoutKey = ["checkout", "session"] as const;
const cartSummaryKey = ["cart", "summary"] as const;

async function fetchCheckout() {
  const response = await apiFetch<CheckoutSession>("/commerce/checkout/");
  return response.data;
}

export function useCheckout() {
  const queryClient = useQueryClient();

  const sessionQuery = useQuery({
    queryKey: checkoutKey,
    queryFn: fetchCheckout,
  });

  const updateShippingInfo = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      return apiFetch("/commerce/checkout/shipping_info/", {
        method: "POST",
        body: payload,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: checkoutKey });
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
  });

  const selectShippingMethod = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      return apiFetch("/commerce/checkout/shipping_method/", {
        method: "POST",
        body: payload,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: checkoutKey });
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
  });

  const selectPaymentMethod = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      return apiFetch("/commerce/checkout/payment_method/", {
        method: "POST",
        body: payload,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: checkoutKey });
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
  });

  const completeCheckout = useMutation({
    mutationFn: async (payload?: Record<string, unknown>) => {
      return apiFetch("/commerce/checkout/complete/", {
        method: "POST",
        body: payload,
      });
    },
  });

  return {
    sessionQuery,
    updateShippingInfo,
    selectShippingMethod,
    selectPaymentMethod,
    completeCheckout,
  };
}
