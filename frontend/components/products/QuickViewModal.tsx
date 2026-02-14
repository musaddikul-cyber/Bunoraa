"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { ProductBadge, ProductListItem } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";
import { AddToCartButton } from "@/components/cart/AddToCartButton";
import Link from "next/link";
import { RatingStars } from "@/components/products/RatingStars";
import { ProductBadges } from "@/components/products/ProductBadges";
import { ProductPrice } from "@/components/products/ProductPrice";

type QuickViewData = ProductListItem & {
  badges?: ProductBadge[];
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

  React.useEffect(() => {
    if (!isOpen) return;

    const previousOverflow = document.body.style.overflow;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center md:items-center"
      role="dialog"
      aria-modal="true"
      aria-label="Quick view"
    >
      <button
        type="button"
        className="absolute inset-0 bg-black/50"
        aria-label="Close quick view"
        onClick={onClose}
      />
      <div className="relative z-10 w-full max-w-3xl px-2 pb-2 sm:px-4 sm:pb-4 md:pb-0">
        <Card
          variant="bordered"
          className={cn(
            "max-h-[92dvh] overflow-y-auto bg-background p-4 sm:p-6",
            "rounded-2xl md:rounded-2xl"
          )}
        >
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Quick view</h2>
            <Button variant="ghost" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>

          {isLoading || !data ? (
            <div className="h-40 animate-pulse rounded-xl bg-muted" />
          ) : (
            <div className="grid gap-4 sm:gap-6 md:grid-cols-[1fr_1.2fr]">
              <div className="aspect-[4/5] max-h-[50dvh] overflow-hidden rounded-xl bg-muted md:max-h-none">
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
                  <h3 className="text-xl font-semibold sm:text-2xl">{data.name}</h3>
                </div>
                <ProductBadges product={data} badges={data.badges} />
                <p className="text-sm text-foreground/70">
                  {data.short_description || "No description available."}
                </p>
                <RatingStars rating={data.average_rating || 0} count={data.reviews_count} />
                <ProductPrice
                  price={data.price}
                  salePrice={data.sale_price}
                  currentPrice={data.current_price}
                  currency={data.currency}
                />
                <div className="flex flex-col gap-2 sm:flex-row">
                  <AddToCartButton
                    productId={data.id}
                    variant="primary-gradient"
                    size="sm"
                    className="h-11 w-full px-4 sm:w-auto"
                  />
                  <Button
                    asChild
                    variant="secondary"
                    size="sm"
                    className="h-11 w-full px-4 sm:w-auto"
                  >
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
