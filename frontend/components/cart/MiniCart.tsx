"use client";

import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useCart } from "@/components/cart/useCart";

export function MiniCart() {
  const { cartQuery, removeItem, updateItem } = useCart();

  if (cartQuery.isLoading) {
    return <div className="text-sm text-foreground/60">Loading cart...</div>;
  }

  if (cartQuery.isError || !cartQuery.data) {
    return <div className="text-sm text-foreground/60">Cart unavailable.</div>;
  }

  const cart = cartQuery.data;

  return (
    <Card variant="bordered" className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Mini cart</h3>
        <span className="text-sm text-foreground/60">
          {cart.item_count} item{cart.item_count === 1 ? "" : "s"}
        </span>
      </div>

      <div className="space-y-3">
        {cart.items.length === 0 ? (
          <p className="text-sm text-foreground/60">Your cart is empty.</p>
        ) : (
          cart.items.map((item) => (
            <div key={item.id} className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-medium">{item.product_name}</p>
                <p className="text-xs text-foreground/60">
                  {item.unit_price} {cart.currency}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    updateItem.mutate({
                      itemId: item.id,
                      quantity: Math.max(1, item.quantity - 1),
                    })
                  }
                >
                  -
                </Button>
                <span className="text-sm">{item.quantity}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    updateItem.mutate({
                      itemId: item.id,
                      quantity: item.quantity + 1,
                    })
                  }
                >
                  +
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeItem.mutate(item.id)}
                >
                  Remove
                </Button>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="flex items-center justify-between border-t border-border pt-3 text-sm">
        <span className="text-foreground/70">Total</span>
        <span className="font-semibold">
          {cart.total} {cart.currency}
        </span>
      </div>

      <Button asChild variant="primary-gradient">
        <Link href="/cart/">View cart</Link>
      </Button>
    </Card>
  );
}
