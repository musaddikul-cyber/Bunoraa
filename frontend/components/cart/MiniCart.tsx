"use client";

import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useCart } from "@/components/cart/useCart";

function formatMoney(amount: string | number, currency: string) {
  if (typeof amount === "string") {
    const trimmed = amount.trim();
    if (!trimmed) return "";
    if (/[^0-9.,-]/.test(trimmed)) {
      return trimmed;
    }
    const normalized = trimmed.replace(/,/g, "");
    const parsed = Number(normalized);
    if (Number.isFinite(parsed)) {
      amount = parsed;
    }
  }
  const numeric = Number(amount);
  if (!Number.isFinite(numeric)) {
    return String(amount);
  }
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency,
      maximumFractionDigits: 2,
    }).format(numeric);
  } catch {
    return `${numeric.toFixed(2)} ${currency}`;
  }
}

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

export function MiniCart({
  title = "Mini cart",
  onClose,
}: {
  title?: string;
  onClose?: () => void;
}) {
  const { cartQuery, cartSummaryQuery, removeItem, updateItem } = useCart();

  if (cartQuery.isLoading) {
    return <div className="text-sm text-foreground/60">Loading cart...</div>;
  }

  if (cartQuery.isError || !cartQuery.data) {
    return <div className="text-sm text-foreground/60">Cart unavailable.</div>;
  }

  const cart = cartQuery.data;
  const summary = cartSummaryQuery.data;
  const currency = summary?.currency_code || cart.currency || "";
  const derivedSubtotal = cart.items.reduce((sum, item) => {
    const lineTotal = parseMoney(item.total);
    if (lineTotal !== null) return sum + lineTotal;
    const unit = parseMoney(item.unit_price) ?? 0;
    const qty = Number.isFinite(item.quantity) ? item.quantity : 0;
    return sum + unit * qty;
  }, 0);
  const apiSubtotal = parseMoney(summary?.subtotal ?? cart.subtotal);
  const preferDerivedSubtotal = derivedSubtotal > 0 && (apiSubtotal === null || apiSubtotal === 0);
  const subtotalLabel =
    summary?.formatted_subtotal && !preferDerivedSubtotal
      ? summary.formatted_subtotal
      : formatMoney(preferDerivedSubtotal ? derivedSubtotal : apiSubtotal ?? 0, currency);
  const totalLabel =
    summary?.formatted_total ||
    (summary?.shipping_cost || summary?.tax_amount
      ? formatMoney(cart.total, currency)
      : "");
  const showEstimatedTotal =
    Boolean(totalLabel) &&
    totalLabel !== subtotalLabel &&
    (summary?.shipping_cost !== undefined || summary?.tax_amount !== undefined);

  return (
    <Card variant="bordered" className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">{title}</h3>
          <span className="text-sm text-foreground/60">
            {cart.item_count} item{cart.item_count === 1 ? "" : "s"}
          </span>
        </div>
        {onClose ? (
          <button
            type="button"
            className="text-sm text-foreground/60 hover:text-foreground"
            onClick={onClose}
          >
            Close
          </button>
        ) : null}
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
                  {formatMoney(item.unit_price, currency)}
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

      <div className="space-y-2 border-t border-border pt-3 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-foreground/70">Subtotal</span>
          <span className="font-semibold">{subtotalLabel}</span>
        </div>
        {showEstimatedTotal ? (
          <div className="flex items-center justify-between">
            <span className="text-foreground/70">Estimated total</span>
            <span className="font-semibold">{totalLabel}</span>
          </div>
        ) : null}
      </div>

      <Button asChild variant="primary-gradient">
        <Link href="/cart/">View cart</Link>
      </Button>
    </Card>
  );
}
