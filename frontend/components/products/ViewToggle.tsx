"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { updateParamValue } from "@/lib/productFilters";
import { Button } from "@/components/ui/Button";

export function ViewToggle() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const view = (searchParams.get("view") as "grid" | "list") || "grid";

  return (
    <div className="flex items-center gap-1 rounded-full border border-border bg-card p-1">
      <Button
        size="sm"
        variant={view === "grid" ? "secondary" : "ghost"}
        onClick={() => {
          const params = updateParamValue(searchParams, "view", "grid");
          router.push(`?${params.toString()}`);
        }}
      >
        Grid
      </Button>
      <Button
        size="sm"
        variant={view === "list" ? "secondary" : "ghost"}
        onClick={() => {
          const params = updateParamValue(searchParams, "view", "list");
          router.push(`?${params.toString()}`);
        }}
      >
        List
      </Button>
    </div>
  );
}