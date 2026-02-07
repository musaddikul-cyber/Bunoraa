"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { ProductListItem } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";
import { AddToCartButton } from "@/components/cart/AddToCartButton";
import Link from "next/link";

type QuickViewData = ProductListItem & {
  badges?: Array<{ id: string; name: string; slug: string }>;
};

async function fetchQuickView(slug: string) {
  const response = await apiFetch<QuickViewData>(
    `/catalog/products/${slug}/quick-view/`
  );
  return response.data;
}

export function QuickViewModal({
  slug,
  isOpen,
  onClose,
}: {
  slug: string | null;
  isOpen: boolean;
  onClose: () => void;
}) {
  const { data, isLoading } = useQuery({
    queryKey: ["quick-view", slug],
    queryFn: () => fetchQuickView(slug as string),
    enabled: isOpen && !!slug,
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="absolute left-1/2 top-1/2 w-full max-w-3xl -translate-x-1/2 -translate-y-1/2 px-4">
        <Card variant="bordered" className={cn("bg-background", "p-6")}
        >
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Quick view</h2>
            <Button variant="ghost" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>

          {isLoading || !data ? (
            <div className="h-32 animate-pulse rounded-xl bg-muted" />
          ) : (
            <div className="grid gap-6 md:grid-cols-[1fr_1.2fr]">
              <div className="aspect-[4/5] overflow-hidden rounded-xl bg-muted">
                {data.primary_image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={data.primary_image}
                    alt={data.name}
                    className="h-full w-full object-cover"
                  />
                ) : null}
              </div>
              <div className="flex flex-col gap-3">
                <div>
                  <p className="text-sm text-foreground/60">
                    {data.primary_category_name || "Featured"}
                  </p>
                  <h3 className="text-2xl font-semibold">{data.name}</h3>
                </div>
                <p className="text-sm text-foreground/70">
                  {data.short_description}
                </p>
                <div className="text-lg font-semibold">
                  {data.current_price} {data.currency}
                </div>
                <div className="flex flex-wrap gap-2">
                  <AddToCartButton
                    productId={data.id}
                    variant="primary-gradient"
                    size="sm"
                  />
                  <Button asChild variant="secondary" size="sm">
                    <Link href={`/products/${data.slug}/`}>View details</Link>
                  </Button>
                </div>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
