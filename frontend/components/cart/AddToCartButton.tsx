"use client";

import * as React from "react";
import { Button, ButtonProps } from "@/components/ui/Button";
import { useCart } from "@/components/cart/useCart";

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

  const handleClick = React.useCallback(() => {
    addItem.mutate({ productId, quantity, variantId });
  }, [addItem, productId, quantity, variantId]);

  return (
    <Button onClick={handleClick} disabled={addItem.isPending} {...props}>
      {addItem.isPending ? "Adding..." : label}
    </Button>
  );
}
