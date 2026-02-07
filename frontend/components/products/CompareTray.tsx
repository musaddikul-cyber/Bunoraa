"use client";

import * as React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { getCompareItems, removeCompareItem, clearCompareItems } from "@/lib/compare";
import { formatMoney } from "@/lib/money";

export function CompareTray() {
  const [items, setItems] = React.useState(getCompareItems());

  React.useEffect(() => {
    const handler = () => setItems(getCompareItems());
    handler();
    window.addEventListener("compare-updated", handler);
    return () => window.removeEventListener("compare-updated", handler);
  }, []);

  if (!items.length) return null;

  return (
    <div className="fixed bottom-4 left-1/2 z-50 w-[95%] max-w-4xl -translate-x-1/2 rounded-2xl border border-border bg-card p-4 shadow-soft">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          {items.map((item) => (
            <div key={item.id} className="flex items-center gap-2 rounded-full border border-border bg-background px-3 py-1 text-xs">
              <span className="max-w-[160px] truncate">{item.name}</span>
              <button
                type="button"
                className="flex h-6 w-6 items-center justify-center rounded-full text-foreground/60 hover:bg-muted"
                onClick={() => {
                  removeCompareItem(item.id);
                  setItems(getCompareItems());
                }}
                aria-label="Remove from compare"
              >
                x
              </button>
            </div>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => clearCompareItems()}>
            Clear
          </Button>
          <Button asChild size="sm" variant="primary-gradient">
            <Link href="/compare/">Compare ({items.length})</Link>
          </Button>
          <div className="hidden text-xs text-foreground/60 sm:block">
            {items[0]?.current_price
              ? `Starting at ${formatMoney(items[0]?.current_price, items[0]?.currency || "USD")}`
              : ""}
          </div>
        </div>
      </div>
    </div>
  );
}
