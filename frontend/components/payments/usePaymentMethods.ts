import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { SavedPaymentMethod } from "@/lib/types";

const methodsKey = ["payments", "methods"] as const;

async function fetchMethods() {
  const response = await apiFetch<SavedPaymentMethod[]>("/payments/methods/");
  return response.data;
}

export function usePaymentMethods() {
  const queryClient = useQueryClient();

  const methodsQuery = useQuery({
    queryKey: methodsKey,
    queryFn: fetchMethods,
  });

  const removeMethod = useMutation({
    mutationFn: async (id: string) => {
      return apiFetch(`/payments/methods/${id}/`, { method: "DELETE" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: methodsKey });
    },
  });

  const setDefault = useMutation({
    mutationFn: async (id: string) => {
      return apiFetch(`/payments/methods/${id}/set-default/`, {
        method: "POST",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: methodsKey });
    },
  });

  const setupIntent = useMutation({
    mutationFn: async () => {
      const response = await apiFetch<{ client_secret: string }>(
        "/payments/methods/setup-intent/"
      );
      return response.data;
    },
  });

  const saveMethod = useMutation({
    mutationFn: async (paymentMethodId: string) => {
      return apiFetch("/payments/methods/", {
        method: "POST",
        body: { payment_method_id: paymentMethodId },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: methodsKey });
    },
  });

  return {
    methodsQuery,
    removeMethod,
    setDefault,
    setupIntent,
    saveMethod,
  };
}
