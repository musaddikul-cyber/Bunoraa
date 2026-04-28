"use client";

import * as React from "react";
import Link from "next/link";
import Image from "next/image";
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
import { buildProductPath } from "@/lib/productPaths";

const DEFAULT_CARD_ASPECT_RATIO = 4 / 5;

function parseAspectRatio(value?: string | null) {
  if (!value) return DEFAULT_CARD_ASPECT_RATIO;
  const normalized = String(value).trim();
  if (!normalized) return DEFAULT_CARD_ASPECT_RATIO;

  const parts = normalized.split(/[/:]/).map((part) => Number(part.trim()));
  if (parts.length !== 2) return DEFAULT_CARD_ASPECT_RATIO;
  const [width, height] = parts;
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return DEFAULT_CARD_ASPECT_RATIO;
  }
  return width / height;
}

function MinimalProductCard({
  product,
  showWishlist = true,
  showQuickView,
  onQuickView,
}: {
  product: ProductListItem;
  showWishlist?: boolean;
  showQuickView?: boolean;
  onQuickView?: (slug: string) => void;
}) {
  const image =
    typeof product.primary_image === "string"
      ? product.primary_image
      : (product.primary_image as unknown as { image?: string | null })?.image || null;
  const productHref = buildProductPath(product);

  const canQuickView = typeof onQuickView === "function";
  const aspectRatioValue = parseAspectRatio(product.aspect_ratio);
  const gridImageSizes =
    "(max-width: 640px) 50vw, (max-width: 1024px) 33vw, 25vw";
  return (
    <div className="group">
      <div
        className="relative overflow-hidden bg-muted"
        style={{ aspectRatio: aspectRatioValue }}
      >
        {showWishlist ? (
          <WishlistIconButton
            productId={product.id}
            variant="ghost"
            size="lg"
            color="fixed-black"
            className="absolute right-0 top-0 z-40 opacity-100 scale-75 transition sm:scale-100 sm:right-2 sm:top-2"
          />
        ) : null}
        {canQuickView ? (
          <button
            type="button"
            className="absolute inset-0 z-10"
            onClick={() => onQuickView?.(product.slug)}
            aria-label={`Quick view ${product.name}`}
          >
            <span className="sr-only">Quick view</span>
          </button>
        ) : (
          <Link
            href={productHref}
            prefetch={false}
            className="absolute inset-0 z-10"
            aria-label={`View ${product.name}`}
          />
        )}
        {image ? (
          <Image
            src={image}
            alt={product.name}
            fill
            sizes={gridImageSizes}
            quality={72}
            loading="lazy"
            decoding="async"
            className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
          />
        ) : null}
      </div>
      <div className="mt-3 space-y-1">
        {!product.is_in_stock ? (
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-foreground/70">
            Sold Out
          </p>
        ) : null}
        <Link
          href={productHref}
          prefetch={false}
          className="block text-sm font-normal leading-snug text-foreground"
        >
          {product.name}
        </Link>
        <ProductPrice
          price={product.price}
          salePrice={product.sale_price}
          currentPrice={product.current_price}
          currency={product.currency}
          className="text-foreground"
          priceClassName="text-[14px] font-medium sm:text-[16px]"
        />
      </div>
    </div>
  );
}

function InteractiveProductCard({
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
  const productHref = buildProductPath(product);

  const canQuickView = typeof onQuickView === "function";
  const aspectRatioValue = parseAspectRatio(product.aspect_ratio);
  const gridImageSizes =
    "(max-width: 640px) 50vw, (max-width: 1024px) 33vw, 25vw";
  const listImageSizes = "(max-width: 640px) 100vw, 224px";

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
          variant === "list" ? "h-40 w-full sm:h-40 sm:w-56" : ""
        )}
        style={variant !== "list" ? { aspectRatio: aspectRatioValue } : undefined}
      >
        {canQuickView ? (
          <button
            type="button"
            className="absolute inset-0 z-0"
            onClick={() => onQuickView?.(product.slug)}
            aria-label={`Quick view ${product.name}`}
          >
            <span className="sr-only">Quick view</span>
          </button>
        ) : (
          <Link
            href={productHref}
            prefetch={false}
            className="absolute inset-0 z-0"
            aria-label={`View ${product.name}`}
            target="_blank"
            rel="noopener noreferrer"
          />
        )}
        <WishlistIconButton
          productId={product.id}
          variant="ghost"
          size="lg"
          color="fixed-black"
          className="absolute right-0 top-0 z-40 opacity-100 scale-75 transition sm:scale-100 sm:right-2 sm:top-2 sm:pointer-events-none sm:opacity-0 sm:group-hover:pointer-events-auto sm:group-hover:opacity-100"
        />
        {image ? (
          <Image
            src={image}
            alt={product.name}
            fill
            sizes={variant === "list" ? listImageSizes : gridImageSizes}
            quality={72}
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
            href={productHref}
            prefetch={false}
            className="block text-base font-normal leading-snug sm:text-lg"
            target="_blank"
            rel="noopener noreferrer"
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
          priceClassName="text-sm font-medium sm:text-base"
        />
        <div className="mt-auto grid grid-cols-2 gap-2 sm:flex sm:flex-nowrap sm:items-center">
          <AddToCartButton
            productId={product.id}
            size="sm"
            variant="secondary"
            className="w-full justify-center sm:flex-1"
            label={product.is_in_stock ? "Add to bag" : "Out of stock"}
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

export function ProductCard({
  product,
  variant = "grid",
  showWishlist = true,
  showQuickView,
  onQuickView,
}: {
  product: ProductListItem;
  variant?: "grid" | "list" | "minimal";
  showWishlist?: boolean;
  showQuickView?: boolean;
  onQuickView?: (slug: string) => void;
}) {
  if (variant === "minimal") {
    return (
      <MinimalProductCard
        product={product}
        showWishlist={showWishlist}
        showQuickView={showQuickView}
        onQuickView={onQuickView}
      />
    );
  }

  return (
    <InteractiveProductCard
      product={product}
      variant={variant}
      showQuickView={showQuickView}
      onQuickView={onQuickView}
    />
  );
}
