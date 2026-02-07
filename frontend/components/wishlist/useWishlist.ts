import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { WishlistItem } from "@/lib/types";

const wishlistKey = ["wishlist"] as const;

async function fetchWishlist() {
  const response = await apiFetch<WishlistItem[]>("/commerce/wishlist/");
  return response;
}

export function useWishlist(options?: { enabled?: boolean }) {
  const queryClient = useQueryClient();
  const enabled = options?.enabled ?? true;

  const wishlistQuery = useQuery({
    queryKey: wishlistKey,
    queryFn: fetchWishlist,
    enabled,
  });

  const removeItem = useMutation({
    mutationFn: async (itemId: string) => {
      return apiFetch(`/commerce/wishlist/remove/${itemId}/`, { method: "POST" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: wishlistKey });
    },
  });

  const moveToCart = useMutation({
    mutationFn: async (itemId: string) => {
      return apiFetch(`/commerce/wishlist/move-to-cart/${itemId}/`, {
        method: "POST",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: wishlistKey });
      queryClient.invalidateQueries({ queryKey: ["cart"] });
    },
  });

  const addItem = useMutation({
    mutationFn: async ({
      productId,
      variantId,
    }: {
      productId: string;
      variantId?: string | null;
    }) => {
      return apiFetch(`/commerce/wishlist/`, {
        method: "POST",
        body: {
          product_id: productId,
          variant_id: variantId || undefined,
        },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: wishlistKey });
    },
  });

  return {
    wishlistQuery,
    addItem,
    removeItem,
    moveToCart,
  };
}
