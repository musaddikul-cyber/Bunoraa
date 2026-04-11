"use client";

import * as React from "react";
import Link from "next/link";
import type { ProductListItem } from "@/lib/types";
import { ProductGrid } from "@/components/products/ProductGrid";
import { cn } from "@/lib/utils";

export type HomeCategoryTabBand = {
  id: string;
  name: string;
  href: string;
  products: ProductListItem[];
};

export function HomeCategoryTabs({ bands }: { bands: HomeCategoryTabBand[] }) {
  const [activeId, setActiveId] = React.useState<string>(bands[0]?.id || "");

  React.useEffect(() => {
    if (!bands.length) {
      setActiveId("");
      return;
    }
    if (!bands.some((band) => band.id === activeId)) {
      setActiveId(bands[0].id);
    }
  }, [activeId, bands]);

  if (!bands.length) {
    return null;
  }

  const activeBand = bands.find((band) => band.id === activeId) || bands[0];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3">
        {bands.map((band) => (
          <button
            key={band.id}
            type="button"
            onClick={() => setActiveId(band.id)}
            className={cn(
              "rounded-full border px-4 py-2 text-sm font-medium transition",
              band.id === activeBand.id
                ? "border-foreground bg-foreground text-background"
                : "border-border bg-transparent text-foreground hover:border-foreground/60"
            )}
          >
            {band.name}
          </button>
        ))}
      </div>

      <ProductGrid products={activeBand.products} cardStyle="minimal" />

      <div className="flex justify-center">
        <Link
          href={activeBand.href}
          className="inline-flex items-center justify-center border border-border px-6 py-2.5 text-xs font-semibold uppercase tracking-[0.24em] text-foreground transition hover:border-foreground hover:bg-muted"
        >
          View All
        </Link>
      </div>
    </div>
  );
}
