"use client";

import * as React from "react";
import Link from "next/link";
import type { ProductListItem } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { WishlistIconButton } from "@/components/wishlist/WishlistIconButton";
import { AddToCartButton } from "@/components/cart/AddToCartButton";
import { RatingStars } from "@/components/products/RatingStars";
import { ProductBadges } from "@/components/products/ProductBadges";
import { ProductPrice } from "@/components/products/ProductPrice";
import { cn } from "@/lib/utils";
import { compareItemFromProduct, useCompareToggle } from "@/components/products/compareHelpers";

export function ProductCard({
  product,
  variant = "grid",
  showQuickView,
  onQuickView,
}: {
  product: ProductListItem;
  variant?: "grid" | "list";
  showQuickView?: boolean;
  onQuickView?: (slug: string) => void;
}) {
  const { isInCompare, toggleCompare } = useCompareToggle(product);
  const image =
    typeof product.primary_image === "string"
      ? product.primary_image
      : (product.primary_image as unknown as { image?: string | null })?.image || null;

  return (
    <Card
      variant="bordered"
      className={cn(
        "group flex flex-col gap-3 p-4 sm:gap-4 sm:p-5",
        variant === "list" ? "sm:flex-row sm:items-center" : ""
      )}
    >
      <div
        className={cn(
          "relative overflow-hidden rounded-xl bg-muted",
          variant === "list" ? "h-40 w-full sm:h-40 sm:w-56" : "aspect-[4/5]"
        )}
      >
        <Link
          href={`/products/${product.slug}/`}
          className="absolute inset-0 z-0"
          aria-label={`View ${product.name}`}
        />
        <WishlistIconButton
          productId={product.id}
          variant="ghost"
          size="lg"
          color="fixed-black"
          className="absolute right-2 top-2 z-20 opacity-100 transition sm:pointer-events-none sm:opacity-0 sm:group-hover:pointer-events-auto sm:group-hover:opacity-100"
        />
        {image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={image}
            alt={product.name}
            loading="lazy"
            decoding="async"
            className="h-full w-full object-cover"
          />
        ) : null}
        <div className="absolute left-2 top-2 z-10 sm:left-3 sm:top-3">
          <ProductBadges product={product} omitOnSale />
        </div>
        {showQuickView ? (
          <div className="absolute bottom-2 left-2 right-2 z-20 opacity-100 transition sm:bottom-3 sm:left-3 sm:right-auto sm:pointer-events-none sm:opacity-0 sm:group-hover:pointer-events-auto sm:group-hover:opacity-100">
            <Button
              size="sm"
              variant="secondary"
              className="w-full bg-background/90 backdrop-blur sm:w-auto"
              onClick={() => onQuickView?.(product.slug)}
            >
              Quick view
            </Button>
          </div>
        ) : null}
      </div>

      <div className="flex flex-1 flex-col gap-2">
        <div>
          <p className="text-[11px] uppercase tracking-[0.16em] text-foreground/60">
            {product.primary_category_name || "Featured"}
          </p>
          <Link
            href={`/products/${product.slug}/`}
            className="block text-base font-semibold leading-snug sm:text-lg"
          >
            {product.name}
          </Link>
        </div>
        <RatingStars rating={product.average_rating || 0} count={product.reviews_count} />
        <ProductPrice
          price={product.price}
          salePrice={product.sale_price}
          currentPrice={product.current_price}
          currency={product.currency}
        />
        <div className="mt-auto grid grid-cols-2 gap-2 sm:flex sm:flex-nowrap sm:items-center">
          <AddToCartButton
            productId={product.id}
            size="sm"
            variant="secondary"
            className="w-full justify-center sm:flex-1"
            label={product.is_in_stock ? "Add to cart" : "Out of stock"}
            disabled={!product.is_in_stock}
          />
          <Button
            size="sm"
            variant={isInCompare ? "primary" : "secondary"}
            className="w-full justify-center sm:w-auto sm:min-w-[110px]"
            onClick={() => toggleCompare(compareItemFromProduct(product))}
          >
            {isInCompare ? "Compared" : "Compare"}
          </Button>
        </div>
      </div>
    </Card>
  );
}
