"use client";

import * as React from "react";
import type { ProductFilterResponse } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import {
  FilterPanel,
  CategoryFacet,
  CategoryFilterItem,
} from "@/components/products/FilterPanel";
import { cn } from "@/lib/utils";

export function FilterDrawer({
  filters,
  facets,
  categories,
  productCount,
  className,
  filterParams,
}: {
  filters: ProductFilterResponse | null;
  facets?: CategoryFacet[];
  categories?: CategoryFilterItem[];
  productCount?: number;
  className?: string;
  filterParams?: Record<string, string>;
}) {
  const [open, setOpen] = React.useState(false);
  const shouldHideFilters = typeof productCount === "number" && productCount <= 1;
  const closeButtonRef = React.useRef<HTMLButtonElement | null>(null);

  React.useEffect(() => {
    if (!open || shouldHideFilters) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, shouldHideFilters]);

  React.useEffect(() => {
    if (!open || shouldHideFilters) return;
    const originalOverflow = document.body.style.overflow;
    const originalOverscroll = document.body.style.overscrollBehavior;
    document.body.style.overflow = "hidden";
    document.body.style.overscrollBehavior = "contain";
    return () => {
      document.body.style.overflow = originalOverflow;
      document.body.style.overscrollBehavior = originalOverscroll;
    };
  }, [open, shouldHideFilters]);

  React.useEffect(() => {
    if (!open || shouldHideFilters) return;
    closeButtonRef.current?.focus();
  }, [open, shouldHideFilters]);

  if (shouldHideFilters) {
    return null;
  }

  return (
    <div className={cn("relative", className)}>
      <Button
        variant="secondary"
        className="w-full sm:w-auto"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls="mobile-filter-drawer"
      >
        Filters
        {typeof productCount === "number" ? ` (${productCount})` : ""}
      </Button>

      {open ? (
        <div
          className="fixed inset-0 z-50"
          role="dialog"
          aria-modal="true"
          aria-labelledby="mobile-filter-title"
        >
          <div className="absolute inset-0 bg-black/45 backdrop-blur-[1px]" onClick={() => setOpen(false)} />
          <div
            id="mobile-filter-drawer"
            className="absolute inset-x-0 bottom-0 max-h-[88dvh] overflow-hidden rounded-t-2xl border border-border bg-background shadow-2xl sm:inset-y-0 sm:left-0 sm:right-auto sm:h-[100dvh] sm:max-h-none sm:w-full sm:max-w-md sm:rounded-none sm:border-r"
          >
            <div className="flex h-full flex-col">
              <div className="sticky top-0 z-10 border-b border-border bg-background/95 px-4 pb-3 pt-[max(0.75rem,env(safe-area-inset-top))] backdrop-blur supports-[backdrop-filter]:bg-background/90 sm:px-5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h2 id="mobile-filter-title" className="text-base font-semibold sm:text-lg">
                      Filters
                    </h2>
                    {typeof productCount === "number" ? (
                      <p className="text-xs text-foreground/60">{productCount} products</p>
                    ) : null}
                  </div>
                  <Button
                    ref={closeButtonRef}
                    variant="ghost"
                    size="sm"
                    onClick={() => setOpen(false)}
                  >
                    Close
                  </Button>
                </div>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-[max(1rem,env(safe-area-inset-bottom))] pt-4 sm:px-5">
                <FilterPanel
                  filters={filters}
                  facets={facets}
                  categories={categories}
                  productCount={productCount}
                  filterParams={filterParams}
                />
              </div>
              <div className="border-t border-border bg-background px-4 pb-[max(0.9rem,env(safe-area-inset-bottom))] pt-3 sm:hidden">
                <Button variant="secondary" className="w-full" onClick={() => setOpen(false)}>
                  Show products
                </Button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
