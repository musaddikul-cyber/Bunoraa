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
  emptyMessage = "We could not find any products matching your current filters.",
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
    return (
      <div className="rounded-2xl border border-dashed border-border bg-card/40 px-6 py-10 text-center">
        <h3 className="text-lg font-semibold text-foreground">No products found</h3>
        <p className="mt-2 text-sm text-foreground/70">{emptyMessage}</p>
        <p className="mt-3 text-xs text-foreground/50">
          Try adjusting your filters or search terms.
        </p>
      </div>
    );
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
