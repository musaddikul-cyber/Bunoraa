"use client";

import * as React from "react";
import type { ProductListItem } from "@/lib/types";
import { ProductCard } from "@/components/products/ProductCard";
import { ProductCardSkeleton } from "@/components/products/ProductCardSkeleton";
import { QuickViewModal } from "@/components/products/QuickViewModal";
import { cn } from "@/lib/utils";

export function ProductGrid({
  products,
  view = "grid",
  isLoading = false,
  emptyMessage = "No products found.",
}: {
  products: ProductListItem[];
  view?: "grid" | "list";
  isLoading?: boolean;
  emptyMessage?: string;
}) {
  const [quickViewSlug, setQuickViewSlug] = React.useState<string | null>(null);

  if (isLoading) {
    return (
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, index) => (
          <ProductCardSkeleton key={index} />
        ))}
      </div>
    );
  }

  if (!products.length) {
    return <p className="text-sm text-foreground/60">{emptyMessage}</p>;
  }

  return (
    <>
      <div
        className={cn(
          "grid gap-6",
          view === "list"
            ? "grid-cols-1"
            : "sm:grid-cols-2 lg:grid-cols-3"
        )}
      >
        {products.map((product) => (
          <ProductCard
            key={product.id}
            product={product}
            variant={view === "list" ? "list" : "grid"}
            showQuickView
            onQuickView={setQuickViewSlug}
          />
        ))}
      </div>
      <QuickViewModal
        slug={quickViewSlug}
        isOpen={Boolean(quickViewSlug)}
        onClose={() => setQuickViewSlug(null)}
      />
    </>
  );
}