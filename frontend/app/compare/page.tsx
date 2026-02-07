"use client";

import * as React from "react";
import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { ProductDetail } from "@/lib/types";
import { getCompareItems, clearCompareItems } from "@/lib/compare";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { ProductPrice } from "@/components/products/ProductPrice";
import { RatingStars } from "@/components/products/RatingStars";

export default function ComparePage() {
  const [items, setItems] = React.useState(getCompareItems());
  const [details, setDetails] = React.useState<Record<string, ProductDetail>>({});

  React.useEffect(() => {
    const handler = () => setItems(getCompareItems());
    handler();
    window.addEventListener("compare-updated", handler);
    return () => window.removeEventListener("compare-updated", handler);
  }, []);

  React.useEffect(() => {
    const load = async () => {
      const next: Record<string, ProductDetail> = {};
      await Promise.all(
        items.map(async (item) => {
          try {
            const response = await apiFetch<ProductDetail>(`/catalog/products/${item.slug}/`);
            next[item.id] = response.data;
          } catch {
            return;
          }
        })
      );
      setDetails(next);
    };
    if (items.length) {
      load();
    }
  }, [items]);

  if (!items.length) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-4xl px-6 py-12">
          <Card variant="bordered" className="space-y-4 text-center">
            <h1 className="text-2xl font-semibold">Compare products</h1>
            <p className="text-sm text-foreground/70">
              Add products to compare from the catalog.
            </p>
            <Button asChild variant="primary-gradient">
              <Link href="/products/">Browse products</Link>
            </Button>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-6xl px-6 py-12">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Compare
            </p>
            <h1 className="text-3xl font-semibold">Side-by-side details</h1>
          </div>
          <Button variant="ghost" onClick={() => { clearCompareItems(); setItems([]); }}>
            Clear compare list
          </Button>
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-[220px_1fr]">
          <div className="space-y-4 text-sm text-foreground/60">
            <div className="font-semibold text-foreground">Overview</div>
            <div>Price</div>
            <div>Rating</div>
            <div>Stock</div>
            <div>Category</div>
            <div>Description</div>
            <div>Attributes</div>
          </div>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {items.map((item) => {
              const detail = details[item.id];
              return (
                <Card key={item.id} variant="bordered" className="space-y-3 p-4">
                  <div className="aspect-[4/5] overflow-hidden rounded-xl bg-muted">
                    {item.primary_image ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={item.primary_image} alt={item.name} className="h-full w-full object-cover" />
                    ) : null}
                  </div>
                  <div>
                    <Link href={`/products/${item.slug}/`} className="text-lg font-semibold">
                      {item.name}
                    </Link>
                    <ProductPrice
                      price={detail?.price || item.current_price}
                      salePrice={detail?.sale_price}
                      currentPrice={detail?.current_price || item.current_price}
                      currency={detail?.currency || item.currency || "USD"}
                    />
                  </div>
                  <RatingStars
                    rating={detail?.average_rating || item.average_rating || 0}
                    count={detail?.reviews_count || item.reviews_count}
                  />
                  <p className="text-sm">
                    {detail?.is_in_stock ? "In stock" : "Out of stock"}
                  </p>
                  <p className="text-sm">
                    {detail?.primary_category?.name || item.primary_category_name || "-"}
                  </p>
                  <p className="text-sm text-foreground/70 max-h-16 overflow-hidden">
                    {detail?.short_description || detail?.description || "-"}
                  </p>
                  <div className="text-xs text-foreground/60">
                    {detail?.attributes?.length
                      ? detail.attributes.slice(0, 4).map((attr) => (
                          <div key={attr.id}>{`${attr.attribute.name}: ${attr.value}`}</div>
                        ))
                      : "-"}
                  </div>
                </Card>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
