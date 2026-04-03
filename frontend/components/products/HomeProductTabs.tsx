"use client";

import * as React from "react";
import type { ProductListItem } from "@/lib/types";
import { ProductGrid } from "@/components/products/ProductGrid";
import { cn } from "@/lib/utils";

export function HomeProductTabs({
  newDrops,
  trending,
}: {
  newDrops: ProductListItem[];
  trending: ProductListItem[];
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
    <div className="space-y-4">
      <div className="flex items-center gap-6 border-b border-border/70 pb-2 text-xs font-semibold uppercase tracking-[0.2em] text-foreground/60">
        {hasNew ? (
          <button
            type="button"
            onClick={() => setActive("new")}
            className={cn(
              "pb-1 transition",
              active === "new" ? "text-foreground" : "hover:text-foreground"
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
              "pb-1 transition",
              active === "trending" ? "text-foreground" : "hover:text-foreground"
            )}
          >
            Most Trending
          </button>
        ) : null}
      </div>
      {showNew ? <ProductGrid products={newDrops} cardStyle="minimal" /> : null}
      {showTrending ? <ProductGrid products={trending} cardStyle="minimal" /> : null}
    </div>
  );
}
