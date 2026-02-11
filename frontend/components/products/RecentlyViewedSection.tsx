"use client";

import * as React from "react";
import Link from "next/link";
import { Card } from "@/components/ui/Card";
import { formatMoney } from "@/lib/money";
import { apiFetch } from "@/lib/api";
import { getRecentlyViewed, setRecentlyViewed } from "@/lib/recentlyViewed";
import { useLocale } from "@/components/providers/LocaleProvider";
import type { ProductDetail } from "@/lib/types";

export function RecentlyViewedSection() {
  const [items, setItems] = React.useState<ReturnType<typeof getRecentlyViewed>>([]);
  const { locale } = useLocale();
  const lastCurrencyRef = React.useRef<string | undefined>(undefined);

  React.useEffect(() => {
    const handler = () => setItems(getRecentlyViewed());
    handler();
    window.addEventListener("recently-viewed-updated", handler);
    return () => window.removeEventListener("recently-viewed-updated", handler);
  }, []);

  React.useEffect(() => {
    const currency = locale.currency;
    if (!currency) return;
    if (lastCurrencyRef.current === currency) return;
    lastCurrencyRef.current = currency;

    const current = getRecentlyViewed();
    if (!current.length) return;

    const needsRefresh = current.some((item) => item.currency && item.currency !== currency);
    if (!needsRefresh) {
      setItems(current);
      return;
    }

    let cancelled = false;
    const refresh = async () => {
      const updated = await Promise.all(
        current.map(async (item) => {
          if (!item.currency || item.currency === currency) return item;
          try {
            const response = await apiFetch<ProductDetail>(`/catalog/products/${item.slug}/`, {
              method: "GET",
            });
            const product = response.data;
            const primaryImage =
              typeof product.primary_image === "string"
                ? product.primary_image
                : (product.primary_image as { image?: string | null } | null)?.image || null;
            const fallbackImage = product.images?.[0]?.image || null;
            return {
              ...item,
              name: product.name,
              slug: product.slug,
              current_price: product.current_price,
              currency: product.currency,
              primary_image: primaryImage || fallbackImage,
              average_rating: product.average_rating ?? item.average_rating,
            };
          } catch {
            return item;
          }
        })
      );
      if (cancelled) return;
      setRecentlyViewed(updated);
      setItems(updated);
    };

    refresh();
    return () => {
      cancelled = true;
    };
  }, [locale.currency]);

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
