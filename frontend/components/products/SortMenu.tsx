"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { updateParamValue } from "@/lib/productFilters";
import { cn } from "@/lib/utils";

const orderingOptions = [
  { value: "-created_at", label: "Newest" },
  { value: "price", label: "Price: Low to High" },
  { value: "-price", label: "Price: High to Low" },
  { value: "name", label: "Name: A-Z" },
  { value: "-name", label: "Name: Z-A" },
  { value: "-sales_count", label: "Bestsellers" },
  { value: "-average_rating", label: "Top rated" },
];

export function SortMenu({ className }: { className?: string } = {}) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const currentOrdering = searchParams.get("ordering") || "-created_at";

  return (
    <label className={cn("flex w-full items-center gap-2 text-sm text-foreground/70 sm:w-auto", className)}>
      <span className="whitespace-nowrap text-xs font-medium uppercase tracking-[0.12em] text-foreground/60 sm:text-sm sm:normal-case sm:tracking-normal">
        Sort
      </span>
      <select
        value={currentOrdering}
        onChange={(event) => {
          const params = updateParamValue(searchParams, "ordering", event.target.value);
          router.push(`?${params.toString()}`);
        }}
        className="h-10 w-full rounded-xl border border-border bg-card px-3 text-sm text-foreground sm:h-9 sm:w-[12.5rem]"
      >
        {orderingOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
