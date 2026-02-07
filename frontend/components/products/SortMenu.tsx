"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { updateParamValue } from "@/lib/productFilters";

const orderingOptions = [
  { value: "-created_at", label: "Newest" },
  { value: "price", label: "Price: Low to High" },
  { value: "-price", label: "Price: High to Low" },
  { value: "name", label: "Name: A-Z" },
  { value: "-name", label: "Name: Z-A" },
  { value: "-sales_count", label: "Bestsellers" },
  { value: "-average_rating", label: "Top rated" },
];

export function SortMenu() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const currentOrdering = searchParams.get("ordering") || "-created_at";

  return (
    <label className="flex items-center gap-2 text-xs text-foreground/60">
      <span className="hidden sm:inline">Sort</span>
      <select
        value={currentOrdering}
        onChange={(event) => {
          const params = updateParamValue(searchParams, "ordering", event.target.value);
          router.push(`?${params.toString()}`);
        }}
        className="h-9 rounded-lg border border-border bg-card px-3 text-sm"
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