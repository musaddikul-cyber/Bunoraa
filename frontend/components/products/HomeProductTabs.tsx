"use client";

import * as React from "react";
import type { ProductListItem } from "@/lib/types";
import { ProductGrid } from "@/components/products/ProductGrid";
import { cn } from "@/lib/utils";

export function HomeProductTabs({
  newDrops,
  trending,
  allowQuickView = true,
  showWishlist = true,
}: {
  newDrops: ProductListItem[];
  trending: ProductListItem[];
  allowQuickView?: boolean;
  showWishlist?: boolean;
}) {
  const [active, setActive] = React.useState<"new" | "trending">("new");
  const hasNew = newDrops.length > 0;
  const hasTrending = trending.length > 0;

  if (!hasNew && !hasTrending) {
    return null;
  }

  const showNew = active === "new" && hasNew;
  const showTrending = active === "trending" && hasTrending;

  return (
    <div className="space-y-6">
      <div
        className={cn(
          "grid gap-2 sm:flex sm:flex-wrap sm:items-center sm:justify-center sm:gap-3",
          hasNew && hasTrending ? "grid-cols-2" : "grid-cols-1"
        )}
      >
        {hasNew ? (
          <button
            type="button"
            onClick={() => setActive("new")}
            className={cn(
              "min-h-11 rounded-full border px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.08em] transition sm:min-h-0 sm:w-auto sm:px-5 sm:text-sm sm:tracking-[0.12em]",
              active === "new"
                ? "border-foreground bg-foreground text-background"
                : "border-border bg-transparent text-foreground hover:border-foreground/60"
            )}
          >
            New Drops
          </button>
        ) : null}
        {hasTrending ? (
          <button
            type="button"
            onClick={() => setActive("trending")}
            className={cn(
              "min-h-11 rounded-full border px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.08em] transition sm:min-h-0 sm:w-auto sm:px-5 sm:text-sm sm:tracking-[0.12em]",
              active === "trending"
                ? "border-foreground bg-foreground text-background"
                : "border-border bg-transparent text-foreground hover:border-foreground/60"
            )}
          >
            Most Trending
          </button>
        ) : null}
      </div>
      {showNew ? (
        <ProductGrid
          products={newDrops}
          cardStyle="minimal"
          allowQuickView={allowQuickView}
          showWishlist={showWishlist}
        />
      ) : null}
      {showTrending ? (
        <ProductGrid
          products={trending}
          cardStyle="minimal"
          allowQuickView={allowQuickView}
          showWishlist={showWishlist}
        />
      ) : null}
    </div>
  );
}
