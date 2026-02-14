"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { getAppliedFilters, parseFilters, removeAppliedFilter, clearAllFilters } from "@/lib/productFilters";

export function AppliedFilters() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const filters = parseFilters(searchParams);
  const applied = getAppliedFilters(filters);

  if (!applied.length) return null;

  return (
    <div className="rounded-xl border border-border/70 bg-card/40 p-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-medium uppercase tracking-[0.14em] text-foreground/60">
          Applied filters
        </p>
        <Button
          variant="ghost"
          size="sm"
          className="min-h-8 px-2.5 text-xs sm:text-sm"
          onClick={() => {
            const params = clearAllFilters(searchParams);
            router.push(`?${params.toString()}`);
          }}
        >
          Clear all
        </Button>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        {applied.map((filter) => (
          <button
            key={`${filter.key}-${filter.value || filter.label}`}
            type="button"
            onClick={() => {
              const params = removeAppliedFilter(searchParams, filter);
              router.push(`?${params.toString()}`);
            }}
            className="inline-flex min-h-9 items-center rounded-full border border-border bg-card px-3.5 py-1.5 text-sm text-foreground/80"
          >
            {filter.label}
          </button>
        ))}
      </div>
    </div>
  );
}
