import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { OrderListItem } from "@/lib/types";

async function fetchOrders() {
  const response = await apiFetch<OrderListItem[]>("/orders/");
  return response;
}

export function useOrders() {
  return useQuery({
    queryKey: ["orders"],
    queryFn: fetchOrders,
  });
}
