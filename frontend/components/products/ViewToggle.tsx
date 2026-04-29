"use client";

import * as React from "react";
import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { updateParamValue } from "@/lib/productFilters";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";

export function ViewToggle({ className }: { className?: string } = {}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const view = (searchParams.get("view") as "grid" | "list") || "grid";

  return (
    <div
      className={cn(
        "grid w-full grid-cols-2 gap-1 rounded-xl border border-border bg-card p-1 sm:inline-flex sm:w-auto sm:items-center",
        className
      )}
    >
      <Button
        size="sm"
        variant={view === "grid" ? "secondary" : "ghost"}
        className="w-full"
        onClick={() => {
          const params = updateParamValue(searchParams, "view", "grid");
          router.push(`${pathname}?${params.toString()}`);
        }}
      >
        Grid
      </Button>
      <Button
        size="sm"
        variant={view === "list" ? "secondary" : "ghost"}
        className="w-full"
        onClick={() => {
          const params = updateParamValue(searchParams, "view", "list");
          router.push(`${pathname}?${params.toString()}`);
        }}
      >
        List
      </Button>
    </div>
  );
}
