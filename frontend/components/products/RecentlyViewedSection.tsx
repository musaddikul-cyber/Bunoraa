"use client";

import * as React from "react";
import Link from "next/link";
import { Card } from "@/components/ui/Card";
import { formatMoney } from "@/lib/money";
import { getRecentlyViewed } from "@/lib/recentlyViewed";

export function RecentlyViewedSection() {
  const [items, setItems] = React.useState(getRecentlyViewed());

  React.useEffect(() => {
    const handler = () => setItems(getRecentlyViewed());
    handler();
    window.addEventListener("recently-viewed-updated", handler);
    return () => window.removeEventListener("recently-viewed-updated", handler);
  }, []);

  if (!items.length) return null;

  return (
    <div className="space-y-4">
      <h3 className="text-xl font-semibold">Recently viewed</h3>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {items.map((item) => (
          <Card key={item.id} variant="bordered" className="flex flex-col gap-3">
            <div className="aspect-[4/5] overflow-hidden rounded-xl bg-muted">
              {item.primary_image ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={item.primary_image} alt={item.name} className="h-full w-full object-cover" />
              ) : null}
            </div>
            <div>
              <Link href={`/products/${item.slug}/`} className="font-semibold">
                {item.name}
              </Link>
              <p className="text-sm text-foreground/70">
                {formatMoney(item.current_price, item.currency || "USD")}
              </p>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}