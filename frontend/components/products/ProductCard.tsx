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
        "group flex flex-col gap-4",
        variant === "list" ? "sm:flex-row sm:items-center" : ""
      )}
    >
      <div
        className={cn(
          "relative overflow-hidden rounded-xl bg-muted",
          variant === "list" ? "h-40 w-full sm:h-40 sm:w-56" : "aspect-[4/5]"
        )}
      >
        <WishlistIconButton
          productId={product.id}
          variant="ghost"
          size="lg"
          color="fixed-black"
          className="absolute right-2 top-2 pointer-events-none opacity-0 transition group-hover:pointer-events-auto group-hover:opacity-100"
        />
        {image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={image}
            alt={product.name}
            className="h-full w-full object-cover"
          />
        ) : null}
        <div className="absolute left-3 top-3">
          <ProductBadges product={product} omitOnSale />
        </div>
        {showQuickView ? (
          <div className="absolute bottom-3 left-3 pointer-events-none opacity-0 transition group-hover:pointer-events-auto group-hover:opacity-100">
            <Button
              size="sm"
              variant="secondary"
              className="bg-background/80 backdrop-blur"
              onClick={() => onQuickView?.(product.slug)}
            >
              Quick view
            </Button>
          </div>
        ) : null}
      </div>

      <div className="flex flex-1 flex-col gap-2">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
            {product.primary_category_name || "Featured"}
          </p>
          <Link href={`/products/${product.slug}/`} className="text-lg font-semibold">
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
        <div className="mt-auto flex flex-wrap gap-2">
          <AddToCartButton
            productId={product.id}
            size="sm"
            variant="secondary"
            label={product.is_in_stock ? "Add to cart" : "Out of stock"}
            disabled={!product.is_in_stock}
          />
          <Button
            size="sm"
            variant={isInCompare ? "primary" : "secondary"}
            onClick={() => toggleCompare(compareItemFromProduct(product))}
          >
            {isInCompare ? "Compare" : "Add to compare"}
          </Button>
        </div>
      </div>
    </Card>
  );
}
