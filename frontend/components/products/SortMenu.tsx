"use client";

import * as React from "react";
import { useRouter, useSearchParams, usePathname } from "next/navigation";
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

const minimalOrderingOptions = [
  { value: "", label: "Default Sorting" },
  { value: "-created_at", label: "Latest" },
  { value: "price", label: "Sort by price: low to high" },
  { value: "-price", label: "Sort by price: high to low" },
];

export function SortMenu({
  className,
  variant = "default",
}: {
  className?: string;
  variant?: "default" | "minimal";
} = {}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const options = variant === "minimal" ? minimalOrderingOptions : orderingOptions;
  const fallbackOrdering = variant === "minimal" ? "" : "-created_at";
  const currentOrdering = searchParams.get("ordering") ?? fallbackOrdering;
  const selectClass =
    variant === "minimal"
      ? "h-9 w-full border border-border bg-transparent px-2 text-xs uppercase tracking-[0.18em] text-foreground sm:w-[13rem]"
      : "h-10 min-h-10 w-full rounded-xl border border-border bg-card px-3 text-sm text-foreground sm:w-[12.5rem]";

  return (
    <label className={cn("flex w-full items-center gap-2 text-sm text-foreground/70 sm:w-auto", className)}>
      <span className="whitespace-nowrap text-xs font-medium uppercase tracking-[0.12em] text-foreground/60 sm:text-sm sm:normal-case sm:tracking-normal">
        Sort
      </span>
      <select
        value={currentOrdering}
        onChange={(event) => {
          const params = updateParamValue(searchParams, "ordering", event.target.value);
          router.push(`${pathname}?${params.toString()}`);
        }}
        className={selectClass}
      >
        {options.map((option) => (
          <option key={option.value || "default"} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
