"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
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
import { buildProductPath } from "@/lib/productPaths";

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
  const { data, isLoading, isFetching } = useQuery({
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
            <Button
              variant="secondary"
              size="sm"
              onClick={onClose}
              className="h-9 min-h-9 rounded-full border-border/70 bg-background/80 px-3 text-xs backdrop-blur supports-[backdrop-filter]:bg-background/70"
            >
              Close
            </Button>
          </div>

          {isLoading || isFetching || !data ? (
            <div className="flex h-48 items-center justify-center rounded-xl border border-border bg-muted/20">
              <div className="flex items-center gap-2 text-sm text-foreground/70">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading quick view...
              </div>
            </div>
          ) : (
            <div className="grid gap-4 sm:gap-6 md:grid-cols-[1fr_1.2fr]">
              <div className="aspect-[4/5] max-h-[50dvh] overflow-hidden rounded-xl bg-muted md:max-h-none">
                {data.primary_image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={data.primary_image}
                    alt={data.name}
                    loading="lazy"
                    decoding="async"
                    className="h-full w-full object-cover"
                  />
                ) : null}
              </div>
              <div className="flex flex-col gap-3">
                <div>
                  {data.primary_category_name ? (
                    <p className="inline-flex items-center rounded-full border border-border/70 bg-muted/60 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-foreground/75">
                      {data.primary_category_name}
                    </p>
                  ) : null}
                  <h3 className="text-xl font-semibold sm:text-2xl">{data.name}</h3>
                </div>
                <ProductBadges product={data} badges={data.badges} omitOnSale />
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
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  <AddToCartButton
                    productId={data.id}
                    variant="primary-gradient"
                    size="sm"
                    className="h-11 w-full px-4"
                  />
                  <Button
                    asChild
                    variant="secondary"
                    size="sm"
                    className="h-11 w-full px-4"
                  >
                    <Link
                      href={buildProductPath(data)}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      View details
                    </Link>
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
