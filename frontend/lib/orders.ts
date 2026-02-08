import type { OrderListItem } from "@/lib/types";

const UUID_REGEX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function isUuid(value: string) {
  return UUID_REGEX.test(value);
}

export function resolveOrderId(
  input: string,
  orders: OrderListItem[]
): string | null {
  const match = orders.find((order) => order.order_number === input);
  return match ? match.id : null;
}
