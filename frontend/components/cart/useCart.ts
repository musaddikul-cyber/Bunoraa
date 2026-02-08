import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch, ApiError } from "@/lib/api";
import type { Cart, CartSummary } from "@/lib/types";

type AddItemInput = {
  productId: string;
  quantity?: number;
  variantId?: string | null;
};

type UpdateItemInput = {
  itemId: string;
  quantity: number;
};

type ApplyCouponInput = {
  code: string;
};

type ShareCartInput = {
  name?: string;
  permission?: string;
  expires_days?: number;
  password?: string;
};

type GiftOptionsInput = {
  is_gift?: boolean;
  gift_message?: string;
  gift_wrap?: boolean;
};

const cartKey = ["cart"] as const;
const cartSummaryKey = ["cart", "summary"] as const;

function parseMoney(value: string | number | null | undefined) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (/[^0-9.,-]/.test(trimmed)) return null;
  const normalized = trimmed.replace(/,/g, "");
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatMoney(value: number) {
  return value.toFixed(2);
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

export function useCart() {
  const queryClient = useQueryClient();

  const cartQuery = useQuery({
    queryKey: cartKey,
    queryFn: fetchCart,
    staleTime: 2 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchOnMount: false,
    retry: (failureCount, error) => {
      if (error instanceof ApiError && error.status === 429) return false;
      return failureCount < 2;
    },
  });

  const cartSummaryQuery = useQuery({
    queryKey: cartSummaryKey,
    queryFn: fetchCartSummary,
    staleTime: 2 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchOnMount: false,
    retry: (failureCount, error) => {
      if (error instanceof ApiError && error.status === 429) return false;
      return failureCount < 2;
    },
  });

  const addItem = useMutation({
    mutationFn: async ({ productId, quantity = 1, variantId }: AddItemInput) => {
      return apiFetch("/commerce/cart/add/", {
        method: "POST",
        body: { product_id: productId, quantity, variant_id: variantId },
        allowGuest: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: cartKey });
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
  });

  const updateItem = useMutation({
    mutationFn: async ({ itemId, quantity }: UpdateItemInput) => {
      return apiFetch(`/commerce/cart/update/${itemId}/`, {
        method: "POST",
        body: { quantity },
        allowGuest: true,
        suppressError: true,
        suppressErrorStatus: [400],
      });
    },
    onMutate: async ({ itemId, quantity }) => {
      await queryClient.cancelQueries({ queryKey: cartKey });
      const previous = queryClient.getQueryData<Cart>(cartKey);
      if (!previous) return { previous };

      const nextItems = previous.items.map((item) => {
        if (item.id !== itemId) return item;
        const unitPrice = parseMoney(item.unit_price);
        const nextTotal =
          unitPrice !== null ? formatMoney(unitPrice * quantity) : item.total;
        return {
          ...item,
          quantity,
          total: nextTotal,
        };
      });

      const nextItemCount = nextItems.reduce(
        (sum, item) => sum + (Number.isFinite(item.quantity) ? item.quantity : 0),
        0
      );

      const lineTotals = nextItems.map((item) => parseMoney(item.total));
      const canSum = lineTotals.every((value) => value !== null);
      const subtotal = canSum
        ? formatMoney(
            lineTotals.reduce((sum, value) => sum + (value ?? 0), 0)
          )
        : previous.subtotal;
      const discount = parseMoney(previous.discount_amount);
      const total =
        canSum && discount !== null
          ? formatMoney(Math.max(0, parseMoney(subtotal) ?? 0) - discount)
          : previous.total;

      queryClient.setQueryData<Cart>(cartKey, {
        ...previous,
        items: nextItems,
        item_count: nextItemCount,
        subtotal,
        total,
      });

      return { previous };
    },
    onError: (_error, _vars, context) => {
      if (context?.previous) {
        queryClient.setQueryData<Cart>(cartKey, context.previous);
      }
    },
    onSuccess: (response) => {
      if (response && typeof response === "object" && "cart" in response) {
        const nextCart = (response as { cart?: Cart }).cart;
        if (nextCart) {
          queryClient.setQueryData(cartKey, nextCart);
          queryClient.invalidateQueries({ queryKey: cartSummaryKey });
          return;
        }
      }
      queryClient.invalidateQueries({ queryKey: cartKey });
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
  });

  const removeItem = useMutation({
    mutationFn: async (itemId: string) => {
      return apiFetch(`/commerce/cart/remove/${itemId}/`, {
        method: "POST",
        allowGuest: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: cartKey });
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
  });

  const clearCart = useMutation({
    mutationFn: async () => {
      return apiFetch(`/commerce/cart/clear/`, {
        method: "POST",
        allowGuest: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: cartKey });
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
  });

  const applyCoupon = useMutation({
    mutationFn: async ({ code }: ApplyCouponInput) => {
      return apiFetch(`/promotions/coupons/apply/`, {
        method: "POST",
        body: { code },
        allowGuest: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: cartKey });
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
  });

  const validateCoupon = useMutation({
    mutationFn: async ({ code, subtotal }: { code: string; subtotal?: string }) => {
      return apiFetch(`/promotions/coupons/validate/`, {
        method: "POST",
        body: subtotal ? { code, subtotal } : { code },
        allowGuest: true,
      });
    },
  });

  const removeCoupon = useMutation({
    mutationFn: async () => {
      return apiFetch(`/commerce/cart/remove_coupon/`, {
        method: "POST",
        allowGuest: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: cartKey });
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
  });

  const updateGiftOptions = useMutation({
    mutationFn: async (payload: GiftOptionsInput) => {
      return apiFetch(`/commerce/cart/gift/`, {
        method: "POST",
        body: payload,
        allowGuest: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
  });

  const validateCart = useMutation({
    mutationFn: async () => {
      return apiFetch(`/commerce/cart/validate/`, {
        method: "POST",
        allowGuest: true,
      });
    },
  });

  const lockPrices = useMutation({
    mutationFn: async (durationHours: number = 24) => {
      return apiFetch(`/commerce/cart/lock-prices/`, {
        method: "POST",
        body: { duration_hours: durationHours },
        allowGuest: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: cartSummaryKey });
    },
  });

  const shareCart = useMutation({
    mutationFn: async (payload: ShareCartInput) => {
      return apiFetch(`/commerce/cart/share/`, {
        method: "POST",
        body: payload,
        allowGuest: true,
      });
    },
  });

  return {
    cartQuery,
    cartSummaryQuery,
    addItem,
    updateItem,
    removeItem,
    clearCart,
    applyCoupon,
    validateCoupon,
    removeCoupon,
    updateGiftOptions,
    validateCart,
    lockPrices,
    shareCart,
  };
}
