import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { Address } from "@/lib/types";

const addressKey = ["addresses"] as const;

async function fetchAddresses() {
  const response = await apiFetch<Address[]>("/accounts/addresses/");
  return response.data;
}

export function useAddresses(options?: { enabled?: boolean }) {
  const queryClient = useQueryClient();
  const enabled = options?.enabled ?? true;

  const addressesQuery = useQuery({
    queryKey: addressKey,
    queryFn: fetchAddresses,
    enabled,
  });

  const createAddress = useMutation({
    mutationFn: async (payload: Partial<Address>) => {
      return apiFetch<Address>("/accounts/addresses/", {
        method: "POST",
        body: payload,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: addressKey });
    },
  });

  const updateAddress = useMutation({
    mutationFn: async ({ id, payload }: { id: string; payload: Partial<Address> }) => {
      return apiFetch<Address>(`/accounts/addresses/${id}/`, {
        method: "PATCH",
        body: payload,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: addressKey });
    },
  });

  const deleteAddress = useMutation({
    mutationFn: async (id: string) => {
      return apiFetch(`/accounts/addresses/${id}/`, { method: "DELETE" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: addressKey });
    },
  });

  return {
    addressesQuery,
    createAddress,
    updateAddress,
    deleteAddress,
  };
}
