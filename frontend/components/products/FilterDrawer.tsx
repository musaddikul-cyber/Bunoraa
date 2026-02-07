"use client";

import * as React from "react";
import type { ProductFilterResponse } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { FilterPanel, CategoryFacet } from "@/components/products/FilterPanel";
import { cn } from "@/lib/utils";

export function FilterDrawer({
  filters,
  facets,
  className,
  filterParams,
}: {
  filters: ProductFilterResponse | null;
  facets?: CategoryFacet[];
  className?: string;
  filterParams?: Record<string, string>;
}) {
  const [open, setOpen] = React.useState(false);

  return (
    <div className={cn("relative", className)}>
      <Button variant="secondary" onClick={() => setOpen(true)}>
        Filters
      </Button>

      {open ? (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <div className="absolute left-0 top-0 h-full w-full max-w-md overflow-y-auto bg-background p-6 shadow-lg">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Filters</h2>
              <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
                Close
              </Button>
            </div>
            <div className="mt-6">
              <FilterPanel filters={filters} facets={facets} filterParams={filterParams} />
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
