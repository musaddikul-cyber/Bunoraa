import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { OrderListItem } from "@/lib/types";

type UseOrdersOptions = {
  status?: string;
  date_from?: string;
  date_to?: string;
  q?: string;
  ordering?: string;
  enabled?: boolean;
};

async function fetchOrders(params?: Omit<UseOrdersOptions, "enabled">) {
  const response = await apiFetch<OrderListItem[]>("/orders/", {
    params: {
      status: params?.status || undefined,
      date_from: params?.date_from || undefined,
      date_to: params?.date_to || undefined,
      q: params?.q || undefined,
      ordering: params?.ordering || undefined,
    },
  });
  return response;
}

export function useOrders(options?: UseOrdersOptions) {
  const enabled = options?.enabled ?? true;
  return useQuery({
    queryKey: [
      "orders",
      options?.status || "",
      options?.date_from || "",
      options?.date_to || "",
      options?.q || "",
      options?.ordering || "",
    ],
    queryFn: () =>
      fetchOrders({
        status: options?.status,
        date_from: options?.date_from,
        date_to: options?.date_to,
        q: options?.q,
        ordering: options?.ordering,
      }),
    enabled,
  });
}
