"use client";

import * as React from "react";
import { Button, ButtonProps } from "@/components/ui/Button";
import { useCart } from "@/components/cart/useCart";
import { ApiError } from "@/lib/api";
import { useToast } from "@/components/ui/ToastProvider";

type AddToCartButtonProps = {
  productId: string;
  variantId?: string | null;
  quantity?: number;
  label?: string;
} & Omit<ButtonProps, "onClick">;

export function AddToCartButton({
  productId,
  variantId,
  quantity = 1,
  label = "Add to cart",
  ...props
}: AddToCartButtonProps) {
  const { addItem } = useCart();
  const { push } = useToast();

  const resolveMessage = React.useCallback((response: unknown, fallback: string) => {
    if (response && typeof response === "object" && "message" in response) {
      const message = String((response as { message?: string }).message || "").trim();
      if (message && message.toLowerCase() !== "ok") return message;
    }
    return fallback;
  }, []);

  const handleClick = React.useCallback(async () => {
    try {
      const response = await addItem.mutateAsync({ productId, quantity, variantId });
      push(resolveMessage(response, "Added to cart."), "success");
    } catch (error) {
      if (error instanceof ApiError) {
        if (typeof error.data === "object" && error.data && "message" in error.data) {
          const message = String((error.data as { message?: string }).message || "").trim();
          push(message || "Could not add to cart.", "error");
          return;
        }
        push(error.message || "Could not add to cart.", "error");
        return;
      }
      push("Could not add to cart.", "error");
    }
  }, [addItem, productId, quantity, resolveMessage, push, variantId]);

  return (
    <Button onClick={handleClick} disabled={addItem.isPending} {...props}>
      {addItem.isPending ? "Adding..." : label}
    </Button>
  );
}
