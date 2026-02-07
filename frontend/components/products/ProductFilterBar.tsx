"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";

const orderingOptions = [
  { value: "", label: "Newest" },
  { value: "price", label: "Price: Low to High" },
  { value: "-price", label: "Price: High to Low" },
  { value: "name", label: "Name: A-Z" },
  { value: "-name", label: "Name: Z-A" },
];

export function ProductFilterBar() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isPending, startTransition] = React.useTransition();

  const currentOrdering = searchParams.get("ordering") || "";
  const currentQuery = searchParams.get("q") || "";

  const updateParam = (key: string, value: string) => {
    const params = new URLSearchParams(searchParams.toString());
    if (value) {
      params.set(key, value);
    } else {
      params.delete(key);
    }
    params.delete("page");
    startTransition(() => {
      router.push(`/products/?${params.toString()}`);
    });
  };

  return (
    <div className="flex flex-wrap items-center gap-3">
      <input
        className="w-full max-w-xs rounded-lg border border-border bg-card px-3 py-2 text-sm"
        placeholder="Search products"
        defaultValue={currentQuery}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            updateParam("q", (event.target as HTMLInputElement).value);
          }
        }}
      />
      <select
        className="rounded-lg border border-border bg-card px-3 py-2 text-sm"
        value={currentOrdering}
        onChange={(event) => updateParam("ordering", event.target.value)}
      >
        {orderingOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {isPending ? (
        <span className="text-xs text-foreground/50">Updating...</span>
      ) : null}
    </div>
  );
}
