"use client";

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
import { buildProductPath } from "@/lib/productPaths";

export type ProductCardVariantName =
  | "standard"
  | "compact"
  | "horizontal"
  | "overlay"
  | "deal"
  | "quick-add"
  | "minimal"
  | "editorial"
  | "rating-focus"
  | "compare-focus"
  | "inventory-focus"
  | "dense-row";

export const PRODUCT_CARD_VARIANTS: Array<{
  id: ProductCardVariantName;
  name: string;
  description: string;
  bestFor: string;
}> = [
  {
    id: "standard",
    name: "Standard Grid",
    description: "Balanced card with media, badges, price, rating, and full actions.",
    bestFor: "Category and search grids",
  },
  {
    id: "compact",
    name: "Compact Tile",
    description: "Small footprint card that keeps core details visible.",
    bestFor: "Sidebars and mobile carousels",
  },
  {
    id: "horizontal",
    name: "Horizontal List",
    description: "Wide row card with quick scan layout and direct actions.",
    bestFor: "List view and recommendation rails",
  },
  {
    id: "overlay",
    name: "Overlay Hero",
    description: "Image-led card with gradient overlay and concise CTA strip.",
    bestFor: "Editorial modules and homepage highlights",
  },
  {
    id: "deal",
    name: "Deal Spotlight",
    description: "Discount-first card with savings emphasis.",
    bestFor: "Promotions and sale collections",
  },
  {
    id: "quick-add",
    name: "Quick Add",
    description: "Action-heavy card optimized for fast bag additions.",
    bestFor: "Repeat purchase and checkout upsell zones",
  },
  {
    id: "minimal",
    name: "Minimal",
    description: "Quiet card with essential data and low visual noise.",
    bestFor: "Dense product rows and mixed-content pages",
  },
  {
    id: "editorial",
    name: "Editorial",
    description: "Story-like presentation with strong typography and media.",
    bestFor: "Landing pages and artisan storytelling",
  },
  {
    id: "rating-focus",
    name: "Rating Focus",
    description: "Trust-forward card prioritizing social proof.",
    bestFor: "Review-heavy products and marketplace ranking pages",
  },
  {
    id: "compare-focus",
    name: "Compare Focus",
    description: "Specification summary card with compare-first action.",
    bestFor: "Decision-support and shortlist pages",
  },
  {
    id: "inventory-focus",
    name: "Inventory Focus",
    description: "Availability-led card highlighting stock and urgency.",
    bestFor: "Drops, low-stock campaigns, and limited batches",
  },
  {
    id: "dense-row",
    name: "Dense Row",
    description: "Ultra-dense row card for high-volume result screens.",
    bestFor: "Power-user search and account/history screens",
  },
];

function resolveImage(product: ProductListItem): string | null {
  if (typeof product.primary_image === "string") return product.primary_image;
  return (product.primary_image as unknown as { image?: string | null })?.image || null;
}

function parseAmount(value?: string | null): number | null {
  if (!value) return null;
  const parsed = Number(String(value).replace(/,/g, ""));
  return Number.isFinite(parsed) ? parsed : null;
}

function getDiscountPercent(product: ProductListItem): number | null {
  const base = parseAmount(product.price);
  const current = parseAmount(product.current_price || product.sale_price || product.price);
  if (!base || !current || base <= current) return null;
  const percentage = Math.round(((base - current) / base) * 100);
  return percentage > 0 ? percentage : null;
}

function getCategoryLabel(product: ProductListItem): string {
  return product.primary_category_name || "Featured";
}

function StockBadge({ product }: { product: ProductListItem }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em]",
        product.is_in_stock
          ? "bg-success-500/10 text-success-700"
          : "bg-error-500/10 text-error-700"
      )}
    >
      {product.is_in_stock ? "In stock" : "Out of stock"}
    </span>
  );
}

function QuickViewButton({
  product,
  onQuickView,
  className,
}: {
  product: ProductListItem;
  onQuickView?: (slug: string) => void;
  className?: string;
}) {
  if (!onQuickView) return null;
  return (
    <Button
      size="sm"
      variant="secondary"
      className={cn("min-h-10", className)}
      onClick={() => onQuickView(product.slug)}
    >
      Quick view
    </Button>
  );
}

function BaseMedia({
  product,
  className,
  showBadges = true,
  showWishlist = true,
  onQuickView,
}: {
  product: ProductListItem;
  className?: string;
  showBadges?: boolean;
  showWishlist?: boolean;
  onQuickView?: (slug: string) => void;
}) {
  const image = resolveImage(product);
  const productHref = buildProductPath(product);
  const canQuickView = typeof onQuickView === "function";

  return (
    <div className={cn("relative overflow-hidden rounded-xl bg-muted", className)}>
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
          className="absolute inset-0 z-0"
          aria-label={`View ${product.name}`}
          target="_blank"
          rel="noopener noreferrer"
        />
      )}
      {showWishlist ? (
        <WishlistIconButton
          productId={product.id}
          size="md"
          variant="ghost"
          color="fixed-black"
          className="absolute right-2 top-2 z-20 bg-background/75 backdrop-blur"
        />
      ) : null}
      {showBadges ? (
        <div className="absolute left-2 top-2 z-10">
          <ProductBadges product={product} omitOnSale />
        </div>
      ) : null}
      {image ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={image}
          alt={product.name}
          loading="lazy"
          decoding="async"
          className="h-full w-full object-cover"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center text-sm text-foreground/55">
          No image
        </div>
      )}
    </div>
  );
}

function SharedTitle({
  product,
  className,
}: {
  product: ProductListItem;
  className?: string;
}) {
  return (
    <Link
      href={buildProductPath(product)}
      className={cn("block font-semibold leading-snug", className)}
      target="_blank"
      rel="noopener noreferrer"
    >
      {product.name}
    </Link>
  );
}

function StandardVariant({
  product,
  isInCompare,
  onToggleCompare,
  onQuickView,
  className,
}: RenderProps) {
  return (
    <Card variant="bordered" className={cn("group flex flex-col gap-4 p-4 sm:p-5", className)}>
      <BaseMedia product={product} className="aspect-[4/5]" onQuickView={onQuickView} />
      <div className="space-y-2">
        <p className="text-[11px] uppercase tracking-[0.16em] text-foreground/60">
          {getCategoryLabel(product)}
        </p>
        <SharedTitle product={product} className="text-base sm:text-lg" />
        <RatingStars rating={product.average_rating || 0} count={product.reviews_count} />
        <ProductPrice
          price={product.price}
          salePrice={product.sale_price}
          currentPrice={product.current_price}
          currency={product.currency}
        />
      </div>
      <div className="mt-auto grid grid-cols-2 gap-2">
        <AddToCartButton
          productId={product.id}
          label={product.is_in_stock ? "Add to bag" : "Out of stock"}
          disabled={!product.is_in_stock}
          size="sm"
          variant="secondary"
          className="w-full justify-center"
        />
        <Button
          size="sm"
          variant={isInCompare ? "primary" : "secondary"}
          className="w-full justify-center"
          onClick={onToggleCompare}
        >
          {isInCompare ? "Compared" : "Compare"}
        </Button>
      </div>
      <QuickViewButton product={product} onQuickView={onQuickView} className="w-full justify-center" />
    </Card>
  );
}

function CompactVariant({ product, onQuickView, className }: RenderProps) {
  return (
    <Card variant="bordered" className={cn("flex flex-col gap-3 p-3", className)}>
      <BaseMedia product={product} className="aspect-square" showBadges={false} onQuickView={onQuickView} />
      <div className="space-y-1">
        <SharedTitle product={product} className="line-clamp-2 text-sm" />
        <ProductPrice
          price={product.price}
          salePrice={product.sale_price}
          currentPrice={product.current_price}
          currency={product.currency}
          className="gap-1"
          priceClassName="text-base"
        />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <AddToCartButton
          productId={product.id}
          label={product.is_in_stock ? "Add" : "Out"}
          disabled={!product.is_in_stock}
          size="sm"
          className="w-full justify-center"
        />
        <QuickViewButton product={product} onQuickView={onQuickView} className="w-full justify-center" />
      </div>
    </Card>
  );
}

function HorizontalVariant({
  product,
  isInCompare,
  onToggleCompare,
  onQuickView,
  className,
}: RenderProps) {
  return (
    <Card variant="bordered" className={cn("flex flex-col gap-4 p-4 sm:flex-row sm:items-center", className)}>
      <BaseMedia product={product} className="h-40 w-full sm:h-36 sm:w-44" onQuickView={onQuickView} />
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <p className="text-xs uppercase tracking-[0.16em] text-foreground/60">{getCategoryLabel(product)}</p>
        <SharedTitle product={product} className="line-clamp-2 text-lg" />
        <RatingStars rating={product.average_rating || 0} count={product.reviews_count} />
        <ProductPrice
          price={product.price}
          salePrice={product.sale_price}
          currentPrice={product.current_price}
          currency={product.currency}
        />
        <div className="mt-1 flex flex-wrap gap-2">
          <AddToCartButton
            productId={product.id}
            label={product.is_in_stock ? "Add to bag" : "Out of stock"}
            disabled={!product.is_in_stock}
            size="sm"
            variant="secondary"
          />
          <Button size="sm" variant={isInCompare ? "primary" : "secondary"} onClick={onToggleCompare}>
            {isInCompare ? "Compared" : "Compare"}
          </Button>
        </div>
      </div>
    </Card>
  );
}

function OverlayVariant({ product, onQuickView, className }: RenderProps) {
  const href = buildProductPath(product);

  return (
    <Card
      variant="bordered"
      className={cn("relative overflow-hidden p-0 shadow-soft transition hover:shadow-soft-lg", className)}
    >
      <BaseMedia product={product} className="aspect-[4/5]" showBadges={false} onQuickView={onQuickView} />
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/75 via-black/20 to-transparent" />
      <div className="absolute inset-x-0 bottom-0 z-20 p-4 text-white">
        <p className="text-[10px] uppercase tracking-[0.2em] text-white/75">{getCategoryLabel(product)}</p>
        <Link
          href={href}
          className="mt-1 block text-lg font-semibold leading-tight"
          target="_blank"
          rel="noopener noreferrer"
        >
          {product.name}
        </Link>
        <div className="mt-2 flex items-center justify-between gap-2">
          <ProductPrice
            price={product.price}
            salePrice={product.sale_price}
            currentPrice={product.current_price}
            currency={product.currency}
            priceClassName="text-white"
            className="text-white [&>*:last-child]:text-white/70"
          />
          <div className="pointer-events-auto flex items-center gap-2">
            <WishlistIconButton productId={product.id} color="fixed-black" variant="default" size="sm" />
            <QuickViewButton product={product} onQuickView={onQuickView} />
          </div>
        </div>
      </div>
    </Card>
  );
}

function DealVariant({ product, onQuickView, className }: RenderProps) {
  const discount = getDiscountPercent(product);

  return (
    <Card variant="modern-gradient" className={cn("flex flex-col gap-4 border border-primary/20 p-4", className)}>
      <BaseMedia product={product} className="aspect-[16/10]" showBadges={false} onQuickView={onQuickView} />
      <div className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <StockBadge product={product} />
          {discount ? (
            <span className="rounded-full bg-accent-500 px-2.5 py-1 text-xs font-semibold text-white">
              Save {discount}%
            </span>
          ) : null}
        </div>
        <SharedTitle product={product} className="line-clamp-2 text-lg" />
        <ProductPrice
          price={product.price}
          salePrice={product.sale_price}
          currentPrice={product.current_price}
          currency={product.currency}
          priceClassName="text-2xl"
        />
      </div>
      <AddToCartButton
        productId={product.id}
        label={product.is_in_stock ? "Grab deal" : "Out of stock"}
        disabled={!product.is_in_stock}
        variant="primary-gradient"
        className="w-full justify-center"
      />
    </Card>
  );
}

function QuickAddVariant({ product, onQuickView, className }: RenderProps) {
  return (
    <Card variant="bordered" className={cn("space-y-4 p-4", className)}>
      <BaseMedia product={product} className="aspect-[3/2]" onQuickView={onQuickView} />
      <div className="space-y-2">
        <SharedTitle product={product} className="text-lg" />
        <ProductPrice
          price={product.price}
          salePrice={product.sale_price}
          currentPrice={product.current_price}
          currency={product.currency}
        />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <AddToCartButton
          productId={product.id}
          label={product.is_in_stock ? "Quick add" : "Out of stock"}
          disabled={!product.is_in_stock}
          variant="primary"
          className="w-full justify-center"
        />
        <QuickViewButton product={product} onQuickView={onQuickView} className="w-full justify-center" />
      </div>
    </Card>
  );
}

function MinimalVariant({ product, onQuickView, className }: RenderProps) {
  return (
    <Card variant="bordered" className={cn("flex flex-col gap-3 p-3", className)}>
      <div className="flex items-center gap-3">
        <BaseMedia
          product={product}
          className="h-20 w-20 shrink-0"
          showBadges={false}
          showWishlist={false}
          onQuickView={onQuickView}
        />
        <div className="min-w-0 flex-1">
          <p className="text-[11px] uppercase tracking-[0.16em] text-foreground/60">{getCategoryLabel(product)}</p>
          <SharedTitle product={product} className="line-clamp-2 text-sm font-medium" />
        </div>
      </div>
      <div className="flex items-center justify-between gap-2">
        <ProductPrice
          price={product.price}
          salePrice={product.sale_price}
          currentPrice={product.current_price}
          currency={product.currency}
          priceClassName="text-base"
        />
        <Link
          href={buildProductPath(product)}
          className="inline-flex min-h-10 items-center rounded-lg border border-border px-3 text-sm font-semibold hover:bg-muted"
          target="_blank"
          rel="noopener noreferrer"
        >
          View
        </Link>
      </div>
    </Card>
  );
}

function EditorialVariant({ product, onQuickView, className }: RenderProps) {
  return (
    <Card variant="glass" className={cn("space-y-4 border border-border/70 p-4 sm:p-5", className)}>
      <BaseMedia product={product} className="aspect-[5/4]" showBadges={false} onQuickView={onQuickView} />
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">Curated pick</p>
        <SharedTitle product={product} className="text-xl" />
        <p className="line-clamp-2 text-sm text-foreground/65">
          Handpicked from our latest artisan selections with a focus on quality, finish, and utility.
        </p>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <ProductBadges product={product} />
        <ProductPrice
          price={product.price}
          salePrice={product.sale_price}
          currentPrice={product.current_price}
          currency={product.currency}
        />
      </div>
      <div className="flex flex-wrap gap-2">
        <Link
          href={buildProductPath(product)}
          className="inline-flex min-h-10 items-center rounded-xl bg-primary px-4 text-sm font-semibold text-white hover:bg-primary/90"
          target="_blank"
          rel="noopener noreferrer"
        >
          Explore product
        </Link>
        <AddToCartButton
          productId={product.id}
          label={product.is_in_stock ? "Add to bag" : "Out of stock"}
          disabled={!product.is_in_stock}
          size="sm"
          variant="secondary"
        />
      </div>
    </Card>
  );
}

function RatingFocusVariant({
  product,
  isInCompare,
  onToggleCompare,
  onQuickView,
  className,
}: RenderProps) {
  return (
    <Card variant="bordered" className={cn("space-y-4 p-4", className)}>
      <BaseMedia product={product} className="aspect-square" onQuickView={onQuickView} />
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <RatingStars
            rating={product.average_rating || 0}
            count={product.reviews_count}
            className="text-sm"
          />
          <StockBadge product={product} />
        </div>
        <SharedTitle product={product} className="text-lg" />
        <ProductPrice
          price={product.price}
          salePrice={product.sale_price}
          currentPrice={product.current_price}
          currency={product.currency}
        />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <AddToCartButton
          productId={product.id}
          label={product.is_in_stock ? "Add to bag" : "Out of stock"}
          disabled={!product.is_in_stock}
          size="sm"
          variant="secondary"
          className="w-full justify-center"
        />
        <Button
          size="sm"
          variant={isInCompare ? "primary" : "secondary"}
          className="w-full justify-center"
          onClick={onToggleCompare}
        >
          {isInCompare ? "Compared" : "Compare"}
        </Button>
      </div>
    </Card>
  );
}

function CompareFocusVariant({
  product,
  isInCompare,
  onToggleCompare,
  onQuickView,
  className,
}: RenderProps) {
  return (
    <Card variant="bordered" className={cn("flex flex-col gap-4 p-4", className)}>
      <BaseMedia product={product} className="aspect-[4/3]" showBadges={false} onQuickView={onQuickView} />
      <div className="space-y-2">
        <SharedTitle product={product} className="line-clamp-2 text-lg" />
        <div className="flex flex-wrap gap-2 text-xs text-foreground/70">
          <span className="rounded-full border border-border bg-card px-2.5 py-1">{getCategoryLabel(product)}</span>
          <span className="rounded-full border border-border bg-card px-2.5 py-1">
            {product.average_rating ? `${product.average_rating.toFixed(1)} stars` : "No rating yet"}
          </span>
          <span className="rounded-full border border-border bg-card px-2.5 py-1">
            {product.is_in_stock ? "Available now" : "Unavailable"}
          </span>
        </div>
        <ProductPrice
          price={product.price}
          salePrice={product.sale_price}
          currentPrice={product.current_price}
          currency={product.currency}
        />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Button
          size="sm"
          variant={isInCompare ? "primary" : "secondary"}
          className="w-full justify-center"
          onClick={onToggleCompare}
        >
          {isInCompare ? "Compared" : "Add to compare"}
        </Button>
        <Link
          href={buildProductPath(product)}
          className="inline-flex min-h-10 items-center justify-center rounded-xl border border-border bg-card px-3 text-sm font-semibold hover:bg-muted"
          target="_blank"
          rel="noopener noreferrer"
        >
          View specs
        </Link>
      </div>
    </Card>
  );
}

function InventoryFocusVariant({
  product,
  inCart,
  onQuickView,
  className,
}: RenderProps) {
  return (
    <Card variant="bordered" className={cn("space-y-3 border border-border/80 p-4", className)}>
      <div className="flex items-center justify-between gap-2">
        <StockBadge product={product} />
        {inCart ? (
          <span className="rounded-full bg-primary/15 px-2.5 py-1 text-xs font-semibold text-primary">
            In your bag
          </span>
        ) : null}
      </div>
      <div className="flex items-center gap-3">
        <BaseMedia product={product} className="h-24 w-24 shrink-0" showBadges={false} onQuickView={onQuickView} />
        <div className="min-w-0 space-y-1">
          <SharedTitle product={product} className="line-clamp-2 text-base" />
          <ProductPrice
            price={product.price}
            salePrice={product.sale_price}
            currentPrice={product.current_price}
            currency={product.currency}
            priceClassName="text-base"
          />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <AddToCartButton
          productId={product.id}
          label={product.is_in_stock ? "Add one more" : "Notify me"}
          disabled={!product.is_in_stock}
          size="sm"
          className="w-full justify-center"
        />
        <WishlistIconButton
          productId={product.id}
          size="md"
          variant="ghost"
          className="h-10 w-full rounded-xl border border-border bg-card hover:bg-muted"
        />
      </div>
    </Card>
  );
}

function DenseRowVariant({
  product,
  isInCompare,
  onToggleCompare,
  onQuickView,
  className,
}: RenderProps) {
  return (
    <Card
      variant="bordered"
      className={cn("grid grid-cols-1 items-center gap-3 p-3 sm:grid-cols-[84px_minmax(0,1fr)_auto_auto_auto]", className)}
    >
      <BaseMedia
        product={product}
        className="h-20 w-20"
        showBadges={false}
        showWishlist={false}
        onQuickView={onQuickView}
      />
      <div className="min-w-0">
        <SharedTitle product={product} className="line-clamp-1 text-sm sm:text-base" />
        <p className="text-xs text-foreground/60">{getCategoryLabel(product)}</p>
      </div>
      <RatingStars
        rating={product.average_rating || 0}
        count={product.reviews_count}
        className="sm:justify-self-center"
      />
      <ProductPrice
        price={product.price}
        salePrice={product.sale_price}
        currentPrice={product.current_price}
        currency={product.currency}
        className="sm:justify-self-end"
        priceClassName="text-base"
      />
      <div className="flex items-center gap-2 sm:justify-self-end">
        <Button size="sm" variant={isInCompare ? "primary" : "secondary"} onClick={onToggleCompare}>
          {isInCompare ? "Compared" : "Compare"}
        </Button>
        <AddToCartButton
          productId={product.id}
          label={product.is_in_stock ? "Add" : "Out"}
          disabled={!product.is_in_stock}
          size="sm"
        />
      </div>
    </Card>
  );
}

type RenderProps = {
  product: ProductListItem;
  className?: string;
  inCart?: boolean;
  onQuickView?: (slug: string) => void;
  isInCompare: boolean;
  onToggleCompare: () => void;
};

export function ProductCardVariant({
  product,
  variant,
  className,
  inCart = false,
  onQuickView,
}: {
  product: ProductListItem;
  variant: ProductCardVariantName;
  className?: string;
  inCart?: boolean;
  onQuickView?: (slug: string) => void;
}) {
  const { isInCompare, toggleCompare } = useCompareToggle(product);
  const renderProps: RenderProps = {
    product,
    className,
    inCart,
    onQuickView,
    isInCompare,
    onToggleCompare: () => toggleCompare(compareItemFromProduct(product)),
  };

  switch (variant) {
    case "standard":
      return <StandardVariant {...renderProps} />;
    case "compact":
      return <CompactVariant {...renderProps} />;
    case "horizontal":
      return <HorizontalVariant {...renderProps} />;
    case "overlay":
      return <OverlayVariant {...renderProps} />;
    case "deal":
      return <DealVariant {...renderProps} />;
    case "quick-add":
      return <QuickAddVariant {...renderProps} />;
    case "minimal":
      return <MinimalVariant {...renderProps} />;
    case "editorial":
      return <EditorialVariant {...renderProps} />;
    case "rating-focus":
      return <RatingFocusVariant {...renderProps} />;
    case "compare-focus":
      return <CompareFocusVariant {...renderProps} />;
    case "inventory-focus":
      return <InventoryFocusVariant {...renderProps} />;
    case "dense-row":
      return <DenseRowVariant {...renderProps} />;
    default:
      return <StandardVariant {...renderProps} />;
  }
}
