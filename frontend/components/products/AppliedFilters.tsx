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
    <div className="flex flex-wrap items-center gap-2">
      {applied.map((filter) => (
        <button
          key={`${filter.key}-${filter.value || filter.label}`}
          type="button"
          onClick={() => {
            const params = removeAppliedFilter(searchParams, filter);
            router.push(`?${params.toString()}`);
          }}
          className="rounded-full border border-border bg-card px-3 py-1 text-xs text-foreground/70"
        >
          {filter.label}
        </button>
      ))}
      <Button
        variant="ghost"
        size="sm"
        onClick={() => {
          const params = clearAllFilters(searchParams);
          router.push(`?${params.toString()}`);
        }}
      >
        Clear all
      </Button>
    </div>
  );
}