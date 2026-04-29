"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiFetch, ApiError } from "@/lib/api";
import type {
  ProductDetail,
  ProductListItem,
  Review,
  ReviewStatistics,
  ShippingRateResponse,
  ShippingMethodOption,
} from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { AddToCartButton } from "@/components/cart/AddToCartButton";
import { RatingStars } from "@/components/products/RatingStars";
import { ProductPrice } from "@/components/products/ProductPrice";
import { formatMoney } from "@/lib/money";
import { useToast } from "@/components/ui/ToastProvider";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { addRecentlyViewed } from "@/lib/recentlyViewed";
import { cn } from "@/lib/utils";
import { RecentlyViewedSection } from "@/components/products/RecentlyViewedSection";
import { ProductGrid } from "@/components/products/ProductGrid";
import { buildProductPath } from "@/lib/productPaths";
import { buildCategoryPath } from "@/lib/categoryPaths";
import { ChevronLeft, ChevronRight, ChevronUp, ChevronDown, X } from "lucide-react";
import { getLazyImageProps } from "@/lib/lazyImage";

type Variant = NonNullable<ProductDetail["variants"]>[number];
type VariantOptionMap = Record<string, string>;

const COLOR_NAME_TO_HEX: Record<string, string> = {
  black: "#111827",
  white: "#ffffff",
  red: "#ef4444",
  green: "#16a34a",
  blue: "#2563eb",
  yellow: "#facc15",
  orange: "#f97316",
  purple: "#a855f7",
  pink: "#ec4899",
  gray: "#6b7280",
  grey: "#6b7280",
  brown: "#8b5a2b",
  beige: "#d4b48c",
  teal: "#0d9488",
  navy: "#1e3a8a",
  maroon: "#7f1d1d",
  magenta: "#db2777",
  olive: "#4d7c0f",
  gold: "#d97706",
  silver: "#94a3b8",
};

function toNumber(value: string | number | null | undefined) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = typeof value === "number" ? value : Number(String(value));
  return Number.isFinite(parsed) ? parsed : null;
}

function formatDateLabel(value?: string | null) {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(date);
}

function getVariantOptionMap(variant: Variant | null | undefined): VariantOptionMap {
  const map: VariantOptionMap = {};
  if (!variant?.option_values?.length) return map;
  variant.option_values.forEach((optionValue) => {
    if (!optionValue.option?.slug || !optionValue.value) return;
    map[optionValue.option.slug] = optionValue.value;
  });
  return map;
}

function getColorSwatch(value: string) {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (/^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(trimmed)) return trimmed;
  return COLOR_NAME_TO_HEX[trimmed.toLowerCase()] || null;
}

function getVariantInStock(variant: Variant | null | undefined, product: ProductDetail) {
  if (!variant) return product.is_in_stock;
  if (typeof variant.stock_quantity === "number") return variant.stock_quantity > 0;
  return product.is_in_stock;
}

function resolveDeliveryLabel(method: ShippingMethodOption | null | undefined) {
  if (!method) return null;
  if (method.delivery_estimate) return method.delivery_estimate;
  if (typeof method.min_days === "number" && typeof method.max_days === "number") {
    if (method.min_days === method.max_days) return `${method.min_days} day delivery`;
    return `${method.min_days}-${method.max_days} day delivery`;
  }
  if (typeof method.min_days === "number") return `${method.min_days}+ day delivery`;
  if (typeof method.max_days === "number") return `Up to ${method.max_days} days`;
  return null;
}

function DetailRow({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode | null | undefined;
}) {
  if (value === null || value === undefined || value === "") return null;
  return (
    <div className="flex items-center justify-between gap-4 text-[13px]">
      <span className="text-foreground/60">{label}</span>
      <span className="text-right">{value}</span>
    </div>
  );
}

function ProductGallery({
  product,
  layout = "default",
}: {
  product: ProductDetail;
  layout?: "default" | "minimal";
}) {
  const images = React.useMemo(() => {
    const next: Array<{ id: string; image: string; alt: string }> = [];
    const pushImage = (id: string, image: string | null | undefined, alt: string) => {
      if (!image) return;
      if (next.some((item) => item.image === image)) return;
      next.push({ id, image, alt });
    };

    const primaryImage =
      typeof product.primary_image === "string"
        ? product.primary_image
        : (product.primary_image as { image?: string | null } | null)?.image || null;

    pushImage("primary", primaryImage, product.name);
    (product.images || []).forEach((image) => {
      pushImage(image.id, image.image, image.alt_text || product.name);
    });
    return next;
  }, [product]);
  const [active, setActive] = React.useState(0);
  const activeImage = images[active] || images[0] || null;
  const [isZoomed, setIsZoomed] = React.useState(false);
  const [isHovering, setIsHovering] = React.useState(false);
  const [zoomOrigin, setZoomOrigin] = React.useState("center");
  const [lightboxOpen, setLightboxOpen] = React.useState(false);
  const thumbsRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    setActive(0);
    setIsZoomed(false);
    setLightboxOpen(false);
  }, [product.id]);

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!activeImage) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width) * 100;
    const y = ((event.clientY - bounds.top) / bounds.height) * 100;
    setZoomOrigin(`${x.toFixed(2)}% ${y.toFixed(2)}%`);
  };

  const zoomActive = isZoomed || isHovering;
  const hasMultipleImages = images.length > 1;
  const isMinimal = layout === "minimal";

  const goNext = React.useCallback(() => {
    if (!images.length) return;
    setActive((prev) => (prev + 1) % images.length);
  }, [images.length]);

  const goPrev = React.useCallback(() => {
    if (!images.length) return;
    setActive((prev) => (prev - 1 + images.length) % images.length);
  }, [images.length]);

  React.useEffect(() => {
    if (!lightboxOpen) return;
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setLightboxOpen(false);
      if (event.key === "ArrowRight") goNext();
      if (event.key === "ArrowLeft") goPrev();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [goNext, goPrev, lightboxOpen]);

  const scrollThumbs = (direction: "up" | "down") => {
    if (!thumbsRef.current) return;
    const delta = direction === "up" ? -120 : 120;
    thumbsRef.current.scrollBy({ top: delta, behavior: "smooth" });
  };

  return (
    <div className={cn(isMinimal ? "grid gap-4 lg:grid-cols-[96px_1fr]" : "space-y-4")}>
      {isMinimal ? (
        <div className="hidden lg:flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={() => scrollThumbs("up")}
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-border bg-background/85 text-foreground/70 transition hover:border-foreground hover:text-foreground"
            aria-label="Scroll thumbnails up"
          >
            <ChevronUp aria-hidden="true" className="h-4 w-4" strokeWidth={1.8} />
          </button>
          <div
            ref={thumbsRef}
            className="flex max-h-[520px] flex-col gap-2 overflow-y-auto pr-1 scrollbar-thin"
          >
            {images.map((image, index) => (
              <button
                key={image.id}
                type="button"
                onClick={() => setActive(index)}
                className={cn(
                  "aspect-square w-20 overflow-hidden border transition",
                  index === active ? "border-foreground" : "border-border hover:border-foreground"
                )}
                aria-label={`Show image ${index + 1}`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  {...getLazyImageProps(image.image, image.alt)}
                  alt={image.alt}
                  className="h-full w-full object-cover"
                />
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => scrollThumbs("down")}
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-border bg-background/85 text-foreground/70 transition hover:border-foreground hover:text-foreground"
            aria-label="Scroll thumbnails down"
          >
            <ChevronDown aria-hidden="true" className="h-4 w-4" strokeWidth={1.8} />
          </button>
        </div>
      ) : null}

      <div className="space-y-4">
        <div
          className={cn(
            "relative aspect-[4/5] w-full overflow-hidden bg-muted",
            isMinimal ? "rounded-none" : "rounded-2xl lg:mx-auto lg:max-w-[500px]"
          )}
          onMouseEnter={() => setIsHovering(true)}
          onMouseLeave={() => {
            setIsHovering(false);
            if (!isZoomed) setZoomOrigin("center");
          }}
          onMouseMove={handleMouseMove}
          onClick={() => setIsZoomed((prev) => !prev)}
        >
          {activeImage ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              {...getLazyImageProps(activeImage.image, activeImage.alt)}
              alt={activeImage.alt}
              className={cn(
                "h-full w-full object-cover transition-transform duration-300",
                zoomActive ? "scale-110" : "scale-100",
                isZoomed ? "cursor-zoom-out" : "cursor-zoom-in"
              )}
              style={{ transformOrigin: zoomOrigin }}
            />
          ) : null}
          {hasMultipleImages && !isMinimal ? (
            <>
              <button
                type="button"
                onClick={goPrev}
                className="absolute left-3 top-1/2 -translate-y-1/2 rounded-full bg-background/85 p-2 text-foreground transition hover:bg-background"
                aria-label="Previous image"
              >
                <ChevronLeft aria-hidden="true" className="h-4 w-4" strokeWidth={1.8} />
              </button>
              <button
                type="button"
                onClick={goNext}
                className="absolute right-3 top-1/2 -translate-y-1/2 rounded-full bg-background/85 p-2 text-foreground transition hover:bg-background"
                aria-label="Next image"
              >
                <ChevronRight aria-hidden="true" className="h-4 w-4" strokeWidth={1.8} />
              </button>
            </>
          ) : null}
          {!isMinimal ? (
            <div className="pointer-events-none absolute bottom-3 right-3">
              <span className="rounded-full bg-background/80 px-2 py-1 text-xs text-foreground/70">
                {images.length ? `${active + 1}/${images.length}` : "1/1"}
              </span>
            </div>
          ) : null}
        </div>

        {hasMultipleImages && !isMinimal ? (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="grid flex-1 grid-cols-4 gap-2">
              {images.slice(0, 8).map((image, index) => (
                <button
                  key={image.id}
                  type="button"
                  onClick={() => setActive(index)}
                  className={cn(
                    "aspect-square overflow-hidden rounded-xl border transition",
                    index === active ? "border-primary" : "border-border hover:border-primary/40"
                  )}
                  aria-label={`Show image ${index + 1}`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    {...getLazyImageProps(image.image, image.alt)}
                    alt={image.alt}
                    className="h-full w-full object-cover"
                  />
                </button>
              ))}
            </div>
            <Button size="sm" variant="secondary" onClick={() => setLightboxOpen(true)}>
              View fullscreen
            </Button>
          </div>
        ) : null}

        {hasMultipleImages && isMinimal ? (
          <div className="flex gap-2 overflow-x-auto lg:hidden">
            {images.map((image, index) => (
              <button
                key={image.id}
                type="button"
                onClick={() => setActive(index)}
                className={cn(
                  "aspect-square w-20 flex-shrink-0 overflow-hidden border",
                  index === active ? "border-foreground" : "border-border"
                )}
                aria-label={`Show image ${index + 1}`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  {...getLazyImageProps(image.image, image.alt)}
                  alt={image.alt}
                  className="h-full w-full object-cover"
                />
              </button>
            ))}
          </div>
        ) : null}

        {product.assets_3d && product.assets_3d.length && !isMinimal ? (
          <Card variant="bordered" className="space-y-2 p-4 text-sm">
            <p className="font-semibold">3D assets</p>
            <div className="space-y-1">
              {product.assets_3d.map((asset) => (
                <a
                  key={asset.id}
                  href={asset.ar_quicklook_url || asset.file || "#"}
                  className="text-primary"
                >
                  {asset.is_ar_compatible ? "View in AR" : "View 3D asset"}
                </a>
              ))}
            </div>
          </Card>
        ) : null}

        {lightboxOpen && activeImage && !isMinimal ? (
          <div className="fixed inset-0 z-70 flex items-center justify-center bg-black/85 p-4">
            <button
              type="button"
              className="absolute right-4 top-4 rounded-full bg-white/15 p-2 text-white transition hover:bg-white/25"
              onClick={() => setLightboxOpen(false)}
              aria-label="Close fullscreen gallery"
            >
              <X aria-hidden="true" className="h-5 w-5" strokeWidth={2} />
            </button>
            {hasMultipleImages ? (
              <button
                type="button"
                onClick={goPrev}
                className="absolute left-4 rounded-full bg-white/15 p-2 text-white transition hover:bg-white/25"
                aria-label="Previous image"
              >
                <ChevronLeft aria-hidden="true" className="h-6 w-6" strokeWidth={1.8} />
              </button>
            ) : null}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              {...getLazyImageProps(activeImage.image, activeImage.alt)}
              alt={activeImage.alt}
              className="max-h-[88vh] max-w-[92vw] rounded-xl object-contain"
            />
            {hasMultipleImages ? (
              <button
                type="button"
                onClick={goNext}
                className="absolute right-4 rounded-full bg-white/15 p-2 text-white transition hover:bg-white/25"
                aria-label="Next image"
              >
                <ChevronRight aria-hidden="true" className="h-6 w-6" strokeWidth={1.8} />
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function BackInStockForm({
  product,
  variantId,
}: {
  product: ProductDetail;
  variantId?: string | null;
}) {
  const { hasToken } = useAuthContext();
  const { push } = useToast();
  const [email, setEmail] = React.useState("");

  const requestNotification = useMutation({
    mutationFn: async () => {
      return apiFetch(`/catalog/products/${product.slug}/request-back-in-stock/`, {
        method: "POST",
        body: {
          variant_id: variantId || undefined,
          email: hasToken ? undefined : email,
        },
      });
    },
    onSuccess: (response) => {
      const message =
        response && typeof response === "object" && "detail" in response
          ? String((response as { detail?: string }).detail || "")
          : "We will notify you when it is back in stock.";
      push(message, "success");
    },
    onError: () => {
      push("Could not submit back in stock request.", "error");
    },
  });

  return (
    <Card variant="bordered" className="space-y-3" id="back-in-stock">
      <h3 className="text-sm font-semibold">Get notified</h3>
      <p className="text-xs text-foreground/60">
        Leave your email and we will let you know when this item is back.
      </p>
      {!hasToken ? (
        <input
          type="email"
          placeholder="Email address"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          className="h-10 rounded-xl border border-border bg-transparent px-3 text-sm"
        />
      ) : null}
      <Button
        size="sm"
        variant="secondary"
        onClick={() => requestNotification.mutate()}
        disabled={requestNotification.isPending || (!hasToken && !email)}
      >
        {requestNotification.isPending ? "Sending..." : "Notify me"}
      </Button>
    </Card>
  );
}

function ShippingEstimator({
  product,
  quantity,
  unitPrice,
}: {
  product: ProductDetail;
  quantity: number;
  unitPrice: string | number | null | undefined;
}) {
  const { push } = useToast();
  const [country, setCountry] = React.useState("Bangladesh");
  const [state, setState] = React.useState("Dhaka");
  const [postalCode, setPostalCode] = React.useState("");
  const [result, setResult] = React.useState<ShippingRateResponse | null>(null);
  const [loading, setLoading] = React.useState(false);
  const safeQuantity = Math.max(1, Number.isFinite(quantity) ? quantity : 1);
  const unitPriceValue =
    toNumber(unitPrice) ??
    toNumber(product.current_price) ??
    toNumber(product.price) ??
    0;
  const subtotal = unitPriceValue * safeQuantity;
  const subtotalLabel = formatMoney(subtotal, product.currency || "USD");
  const baseWeight = toNumber(product.weight);
  const totalWeight = baseWeight ? baseWeight * safeQuantity : undefined;

  React.useEffect(() => {
    setResult(null);
  }, [safeQuantity, unitPrice]);

  const orderedMethods = React.useMemo(() => {
    if (!result?.methods?.length) return [] as ShippingMethodOption[];
    return [...result.methods].sort((a, b) => {
      const aRate = toNumber(a.rate) ?? Number.POSITIVE_INFINITY;
      const bRate = toNumber(b.rate) ?? Number.POSITIVE_INFINITY;
      return aRate - bRate;
    });
  }, [result]);

  const handleEstimate = async () => {
    setLoading(true);
    try {
      const response = await apiFetch<ShippingRateResponse>("/shipping/calculate/", {
        method: "POST",
        body: {
          country,
          state: state || undefined,
          postal_code: postalCode || undefined,
          subtotal,
          item_count: safeQuantity,
          product_ids: [product.id],
          weight: totalWeight || undefined,
        },
      });
      setResult(response.data || null);
    } catch {
      push("Could not estimate shipping.", "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card variant="bordered" className="space-y-2 p-4">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold uppercase tracking-[0.16em] text-foreground/70">
          Shipping estimate
        </h3>
        <p className="text-[13px] text-foreground/60">
          Calculated for {safeQuantity} item{safeQuantity === 1 ? "" : "s"}
          {subtotalLabel ? ` - Subtotal ${subtotalLabel}` : ""}
        </p>
      </div>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <input
          value={country}
          onChange={(event) => setCountry(event.target.value)}
          className="h-11 w-full min-w-0 rounded-xl border border-border bg-transparent px-3 text-sm sm:h-11 sm:flex-1 sm:px-3 sm:text-sm"
          placeholder="Country"
        />
        <input
          value={state}
          onChange={(event) => setState(event.target.value)}
          className="h-11 w-full min-w-0 rounded-xl border border-border bg-transparent px-3 text-sm sm:h-11 sm:flex-1 sm:px-3 sm:text-sm"
          placeholder="State"
        />
        <input
          value={postalCode}
          onChange={(event) => setPostalCode(event.target.value)}
          className="h-11 w-full min-w-0 rounded-xl border border-border bg-transparent px-3 text-sm sm:h-11 sm:flex-1 sm:px-3 sm:text-sm"
          placeholder="Postal code"
        />
        <Button
          size="md"
          variant="secondary"
          onClick={handleEstimate}
          disabled={loading}
          className="h-11 w-full px-4 text-sm sm:h-11 sm:w-auto sm:shrink-0 sm:text-sm"
        >
          {loading ? "Estimating..." : "Get rates"}
        </Button>
      </div>
      {orderedMethods.length ? (
        <div className="space-y-1.5 text-[11px] text-foreground/70">
          {orderedMethods.map((method: ShippingMethodOption) => (
            <div
              key={method.code || method.name}
              className="rounded-xl border border-border/70 bg-background/60 px-2 py-1.5"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="font-medium">{method.name}</span>
                <span>{method.rate_display || method.rate || "-"}</span>
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-foreground/60">
                {resolveDeliveryLabel(method) ? <span>{resolveDeliveryLabel(method)}</span> : null}
                {method.is_express ? <span>Express</span> : null}
                {method.requires_signature ? <span>Signature required</span> : null}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </Card>
  );
}

function ProductReviews({ product, reviewStatsQuery: sharedReviewStatsQuery }: { product: ProductDetail; reviewStatsQuery: ReturnType<typeof useQuery<ReviewStatistics | undefined, Error>> }) {
  const { hasToken } = useAuthContext();
  const { push } = useToast();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [page, setPage] = React.useState(1);
  const [rating, setRating] = React.useState(5);
  const [title, setTitle] = React.useState("");
  const [body, setBody] = React.useState("");

  const reviewsQuery = useQuery({
    queryKey: ["product", product.id, "reviews", page],
    queryFn: async () => {
      const response = await apiFetch<Review[]>(
        `/reviews/product/${product.id}/`,
        { params: { page } }
      );
      return {
        reviews: response.data || [],
        pagination: response.meta?.pagination || null,
      };
    },
  });

  const addReview = useMutation({
    mutationFn: async () => {
      return apiFetch(`/reviews/`, {
        method: "POST",
        body: { product_id: product.id, rating, title, body },
      });
    },
    onSuccess: () => {
      push("Review submitted. Pending approval.", "success");
      setTitle("");
      setBody("");
      sharedReviewStatsQuery.refetch();
      reviewsQuery.refetch();
    },
    onError: (error) => {
      if (error instanceof ApiError && error.message) {
        push(error.message, "error");
        return;
      }
      push("Could not submit review.", "error");
    },
  });

  const summary = sharedReviewStatsQuery.data;
  const canReview = summary?.can_review;
  const canReviewReason = summary?.can_review_reason;
  const totalPages = Math.max(1, reviewsQuery.data?.pagination?.total_pages || 1);
  const search = searchParams?.toString();
  const nextHref = `${pathname}${search ? `?${search}` : ""}#reviews`;
  const loginHref = `/account/login/?next=${encodeURIComponent(nextHref)}`;
  const ratingRows = [5, 4, 3, 2, 1].map((star) => {
    const count = Number(summary?.distribution?.[String(star)] || 0);
    const total = summary?.total_count || 0;
    const percent = total > 0 ? Math.round((count / total) * 100) : 0;
    return { star, count, percent };
  });

  return (
    <Card variant="bordered" className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-[auto_1fr] sm:items-center">
        <div className="space-y-1">
          <h3 className="text-lg font-semibold">Customer reviews</h3>
          {summary ? (
            <p className="text-sm text-foreground/60">
              {summary.average_rating} out of 5 ({summary.total_count} reviews)
            </p>
          ) : null}
        </div>
        <div className="space-y-2">
          {ratingRows.map((row) => (
            <div key={row.star} className="flex items-center gap-2 text-xs">
              <span className="w-10 text-foreground/60">{row.star} star</span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-accent-500"
                  style={{ width: `${row.percent}%` }}
                />
              </div>
              <span className="w-8 text-right text-foreground/60">{row.count}</span>
            </div>
          ))}
        </div>
      </div>

      {reviewsQuery.data?.reviews?.length ? (
        <div className="space-y-4">
          {reviewsQuery.data.reviews.map((review) => (
            <article key={review.id} className="space-y-2 rounded-xl border border-border/70 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2 text-xs text-foreground/60">
                  <span className="font-medium text-foreground">
                    {review.user_name || "Customer"}
                  </span>
                  {review.verified_purchase ? (
                    <span className="rounded-full border border-success-500/40 bg-success-500/10 px-2 py-0.5 text-success-700">
                      Verified purchase
                    </span>
                  ) : null}
                  {review.created_at ? <span>{formatDateLabel(review.created_at)}</span> : null}
                </div>
                <RatingStars rating={review.rating} showCount={false} />
              </div>
              {review.title ? <p className="text-sm font-semibold">{review.title}</p> : null}
              {review.body ? <p className="text-sm text-foreground/70">{review.body}</p> : null}
            </article>
          ))}
        </div>
      ) : (
        <p className="text-sm text-foreground/60">
          {reviewsQuery.isFetching || sharedReviewStatsQuery.isFetching ? "Loading reviews..." : "No reviews yet."}
        </p>
      )}

      {totalPages > 1 ? (
        <div className="flex items-center justify-between text-xs text-foreground/60">
          <Button
            size="sm"
            variant="ghost"
            disabled={page <= 1}
            onClick={() => setPage((prev) => Math.max(1, prev - 1))}
          >
            Previous
          </Button>
          <span>
            Page {page} of {totalPages}
          </span>
          <Button
            size="sm"
            variant="ghost"
            disabled={page >= totalPages}
            onClick={() => setPage((prev) => prev + 1)}
          >
            Next
          </Button>
        </div>
      ) : null}

      <div className="space-y-3">
        <h4 className="text-sm font-semibold">Write a review</h4>
        {!hasToken ? (
          <p className="text-xs text-foreground/60">
            Log in to submit a review.{" "}
            <Link href={loginHref} className="text-primary underline-offset-2 hover:underline">
              Sign in
            </Link>
          </p>
        ) : canReview === false ? (
          <p className="text-xs text-foreground/60">
            {canReviewReason || "You cannot review this product right now."}
          </p>
        ) : (
          <div className="grid gap-3">
            <div className="space-y-2">
              <label className="text-xs text-foreground/60">Rating</label>
              <div className="flex flex-wrap gap-2">
                {[5, 4, 3, 2, 1].map((value) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setRating(value)}
                    className={cn(
                      "rounded-full border px-3 py-1 text-xs transition",
                      rating === value
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border text-foreground/70 hover:border-primary/40"
                    )}
                  >
                    {value} star{value === 1 ? "" : "s"}
                  </button>
                ))}
              </div>
            </div>
            <input
              type="text"
              placeholder="Review title"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              className="h-10 rounded-xl border border-border bg-transparent px-3 text-sm"
            />
            <textarea
              placeholder="Share your experience"
              value={body}
              onChange={(event) => setBody(event.target.value)}
              className="min-h-[100px] rounded-xl border border-border bg-transparent px-3 py-2 text-sm"
            />
            <div className="flex justify-end">
              <Button
                size="sm"
                variant="secondary"
                onClick={() => addReview.mutate()}
                disabled={addReview.isPending || (!title.trim() && !body.trim())}
              >
                {addReview.isPending ? "Submitting..." : "Submit review"}
              </Button>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

export function ProductDetailClient({
  product,
  relatedProducts,
}: {
  product: ProductDetail;
  relatedProducts?: ProductListItem[];
}) {
  const { hasToken } = useAuthContext();

  const variants = React.useMemo<Variant[]>(
    () => product.variants ?? [],
    [product.variants]
  );
  const defaultVariant = React.useMemo(
    () => variants.find((variant) => variant.is_default) || variants[0] || null,
    [variants]
  );
  const [variantId, setVariantId] = React.useState<string | null>(defaultVariant?.id || null);
  const [quantity, setQuantity] = React.useState(1);
  const [selectedOptions, setSelectedOptions] = React.useState<VariantOptionMap>(
    getVariantOptionMap(defaultVariant)
  );
  const [descriptionExpanded, setDescriptionExpanded] = React.useState(false);
  const [sizeChartExpanded, setSizeChartExpanded] = React.useState(false);
  const [moreInfoExpanded, setMoreInfoExpanded] = React.useState(false);
  const [returnsExpanded, setReturnsExpanded] = React.useState(false);

  // Review stats query - fetched at top level to use in multiple places
  const reviewStatsQuery = useQuery({
    queryKey: ["product", product.id, "review-stats"],
    queryFn: async () => {
      const response = await apiFetch<ReviewStatistics>(
        `/reviews/product/${product.id}/statistics/`
      );
      return response.data;
    },
  });

  const variantOptionMapById = React.useMemo(() => {
    const map = new Map<string, VariantOptionMap>();
    variants.forEach((variant) => {
      map.set(variant.id, getVariantOptionMap(variant));
    });
    return map;
  }, [variants]);

  const optionGroups = React.useMemo(() => {
    const groupMap = new Map<
      string,
      { slug: string; name: string; values: string[]; isColor: boolean }
    >();
    variants.forEach((variant) => {
      (variant.option_values || []).forEach((optionValue) => {
        const slug = optionValue.option?.slug || optionValue.option?.name || "";
        if (!slug) return;
        const existing = groupMap.get(slug) || {
          slug,
          name: optionValue.option?.name || slug,
          values: [],
          isColor: /color|colour|shade|tone/i.test(optionValue.option?.name || slug),
        };
        if (!existing.values.includes(optionValue.value)) {
          existing.values.push(optionValue.value);
        }
        groupMap.set(slug, existing);
      });
    });
    return Array.from(groupMap.values());
  }, [variants]);

  const selectedVariant = React.useMemo(
    () => variants.find((variant) => variant.id === variantId) || defaultVariant || null,
    [defaultVariant, variantId, variants]
  );
  const inStock = getVariantInStock(selectedVariant, product);
  const stockQty =
    typeof selectedVariant?.stock_quantity === "number"
      ? selectedVariant.stock_quantity
      : typeof product.available_stock === "number"
      ? product.available_stock
      : null;
  const isLowStock = Boolean(product.is_low_stock) || (stockQty !== null && stockQty > 0 && stockQty <= 5);

  React.useEffect(() => {
    setVariantId(defaultVariant?.id || null);
    setSelectedOptions(getVariantOptionMap(defaultVariant));
    setQuantity(1);
    setDescriptionExpanded(false);
    setSizeChartExpanded(false);
    setMoreInfoExpanded(false);
    setReturnsExpanded(false);
  }, [defaultVariant, product.id]);

  React.useEffect(() => {
    if (!selectedVariant) return;
    const nextSelection = getVariantOptionMap(selectedVariant);
    setSelectedOptions((prev) => {
      const prevKeys = Object.keys(prev);
      const nextKeys = Object.keys(nextSelection);
      if (
        prevKeys.length === nextKeys.length &&
        prevKeys.every((key) => prev[key] === nextSelection[key])
      ) {
        return prev;
      }
      return nextSelection;
    });
  }, [selectedVariant]);

  React.useEffect(() => {
    if (!variants.length) return;
    const inList = variantId ? variants.some((variant) => variant.id === variantId) : false;
    if (!inList) {
      setVariantId(defaultVariant?.id || variants[0]?.id || null);
    }
  }, [defaultVariant, variantId, variants]);

  React.useEffect(() => {
    if (!variants.length || typeof window === "undefined") return;
    const requestedVariant = new URLSearchParams(window.location.search).get("variant");
    if (!requestedVariant) return;
    if (!variants.some((variant) => variant.id === requestedVariant)) return;
    setVariantId(requestedVariant);
  }, [product.id, variants]);

  React.useEffect(() => {
    if (!variants.length || typeof window === "undefined") return;
    const url = new URL(window.location.href);
    const currentVariant = url.searchParams.get("variant");
    if (variantId) {
      if (currentVariant === variantId) return;
      url.searchParams.set("variant", variantId);
      window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
      return;
    }
    if (!currentVariant) return;
    url.searchParams.delete("variant");
    window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
  }, [variantId, variants.length]);

  const variantMatchesSelection = React.useCallback(
    (variant: Variant, selection: VariantOptionMap) => {
      const variantMap = variantOptionMapById.get(variant.id) || {};
      return Object.entries(selection)
        .filter(([, value]) => Boolean(value))
        .every(([slug, value]) => variantMap[slug] === value);
    },
    [variantOptionMapById]
  );

  const resolveVariantForSelection = React.useCallback(
    (selection: VariantOptionMap) => {
      const matched = variants
        .filter((variant) => variantMatchesSelection(variant, selection))
        .sort((a, b) => {
          const aScore = (getVariantInStock(a, product) ? 100 : 0) + (a.is_default ? 10 : 0);
          const bScore = (getVariantInStock(b, product) ? 100 : 0) + (b.is_default ? 10 : 0);
          return bScore - aScore;
        });
      return matched[0] || null;
    },
    [product, variantMatchesSelection, variants]
  );

  const isOptionAvailable = React.useCallback(
    (groupSlug: string, value: string) => {
      const selection = { ...selectedOptions, [groupSlug]: value };
      return variants.some((variant) => variantMatchesSelection(variant, selection));
    },
    [selectedOptions, variantMatchesSelection, variants]
  );

  const handleOptionSelect = (groupSlug: string, value: string) => {
    const nextSelection = { ...selectedOptions, [groupSlug]: value };
    setSelectedOptions(nextSelection);
    const nextVariant = resolveVariantForSelection(nextSelection);
    if (nextVariant) {
      setVariantId(nextVariant.id);
      return;
    }
    const fallbackVariant = variants.find((variant) => {
      const variantMap = variantOptionMapById.get(variant.id) || {};
      return variantMap[groupSlug] === value;
    });
    if (fallbackVariant) {
      setVariantId(fallbackVariant.id);
    }
  };

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const canonicalPath = buildProductPath(product);
    const url = new URL(window.location.href);
    const legacyPaths = [`/products/${product.slug}/`, `/products/${product.slug}`];
    if (!legacyPaths.includes(url.pathname)) return;
    url.pathname = canonicalPath;
    window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
  }, [product]);

  React.useEffect(() => {
    const image =
      typeof product.primary_image === "string"
        ? product.primary_image
        : (product.primary_image as unknown as { image?: string | null })?.image || null;
    const fallbackImage = product.images?.[0]?.image || null;
    addRecentlyViewed({
      id: product.id,
      slug: product.slug,
      name: product.name,
      primary_image: image || fallbackImage,
      current_price: product.current_price,
      currency: product.currency,
      average_rating: product.average_rating,
    });
  }, [product]);

  const maxQty = stockQty && stockQty > 0 ? stockQty : null;
  const clampQuantity = React.useCallback(
    (value: number) => {
      let next = Math.max(1, Math.floor(value || 1));
      if (maxQty !== null) {
        next = Math.min(next, maxQty);
      }
      return next;
    },
    [maxQty]
  );

  React.useEffect(() => {
    setQuantity((prev) => clampQuantity(prev));
  }, [variantId, maxQty, clampQuantity]);

  const unitPrice =
    selectedVariant?.current_price ||
    selectedVariant?.price ||
    product.current_price ||
    product.sale_price ||
    product.price ||
    "0";

  const stockLabel = !inStock ? "Out of stock" : isLowStock ? "Low stock" : "In stock";
  const dimensions = [product.length, product.width, product.height]
    .filter((value) => value !== null && value !== undefined && value !== "")
    .map((value) => String(value))
    .join(" x ");
  const categoryLabel =
    product.primary_category?.name ||
    (product.categories?.length
      ? product.categories.map((category) => category.name).join(", ")
      : null);
  const fullDescription = product.description || product.short_description || "";
  const sizeGroup = optionGroups.find((group) => group.name.toLowerCase().includes("size"));
  const otherGroups = optionGroups.filter((group) => group !== sizeGroup);
  const categoryBreadcrumbs = (() => {
    if (product.breadcrumbs?.length) {
      const seen = new Set<string>();
      const segments: string[] = [];
      return product.breadcrumbs
        .filter((crumb) => {
          if (!crumb?.id || !crumb?.slug) return false;
          if (seen.has(crumb.id)) return false;
          seen.add(crumb.id);
          return true;
        })
        .map((crumb) => {
          segments.push(crumb.slug);
          return {
            label: crumb.name,
            href: buildCategoryPath(segments.join("/")),
          };
        });
    }

    if (product.primary_category) {
      return [
        {
          label: product.primary_category.name,
          href: buildCategoryPath(
            product.primary_category_slug_path || product.primary_category.slug
          ),
        },
      ];
    }

    return [];
  })();
  const breadcrumbLinks = [
    { label: "Home", href: "/" },
    ...categoryBreadcrumbs,
    { label: product.name, href: buildProductPath(product) },
  ];
  const reviewStats = reviewStatsQuery.data;
  const reviewCount = reviewStats?.total_count || 0;
  const canReview = reviewStats?.can_review ?? false;

  return (
    <div className="space-y-10 pb-16">
      <nav className="text-xs uppercase tracking-[0.2em] text-foreground/60">
        {breadcrumbLinks.map((crumb, index) => (
          <span key={crumb.label}>
            {index < breadcrumbLinks.length - 1 ? (
              <>
                <Link href={crumb.href} className="hover:text-foreground">
                  {crumb.label}
                </Link>
                {" / "}
              </>
            ) : (
              <span>{crumb.label}</span>
            )}
          </span>
        ))}
      </nav>

      <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
        <ProductGallery product={product} layout="minimal" />

        <div className="space-y-4">
          <h1 className="text-2xl font-semibold uppercase tracking-[0.12em] sm:text-3xl">
            {product.name}
          </h1>
          <div className="text-xs uppercase tracking-[0.2em] text-foreground/60">
            {reviewCount} {reviewCount === 1 ? "Review" : "Reviews"}
            {hasToken && canReview ? (
              <>
                {" "}
                <a href="#reviews" className="ml-2 underline-offset-2 hover:underline">
                  Add Review
                </a>
              </>
            ) : null}
          </div>
          <ProductPrice
            price={product.price}
            salePrice={product.sale_price}
            currentPrice={
              selectedVariant?.current_price ||
              selectedVariant?.price ||
              product.current_price
            }
            currency={product.currency}
            priceClassName="text-2xl font-semibold"
          />
          <div className="space-y-1 text-sm text-foreground/70">
            <p>
              <span className="font-medium text-foreground/60">Brand:</span> Bunoraa
            </p>
            <p>
              <span className="font-medium text-foreground/60">Availability:</span>{" "}
              {stockLabel}
            </p>
            {(selectedVariant?.sku || product.sku) ? (
              <p>
                <span className="font-medium text-foreground/60">SKU:</span>{" "}
                {selectedVariant?.sku || product.sku}
              </p>
            ) : null}
          </div>

          {sizeGroup ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-foreground/60">
                  {sizeGroup.name}
                </p>
                <a href="#size-chart" className="text-xs uppercase tracking-[0.2em] underline">
                  Size Chart
                </a>
              </div>
              <div className="flex flex-wrap gap-2">
                {sizeGroup.values.map((value) => {
                  const selected = selectedOptions[sizeGroup.slug] === value;
                  const available = isOptionAvailable(sizeGroup.slug, value);
                  return (
                    <button
                      key={value}
                      type="button"
                      disabled={!available}
                      onClick={() => handleOptionSelect(sizeGroup.slug, value)}
                      className={cn(
                        "min-h-9 rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em]",
                        selected
                          ? "border-foreground bg-foreground text-background"
                          : "border-border text-foreground/70 hover:border-foreground",
                        !available && "cursor-not-allowed opacity-40"
                      )}
                    >
                      {value}
                    </button>
                  );
                })}
              </div>
            </div>
          ) : null}

          {otherGroups.length ? (
            <div className="space-y-3">
              {otherGroups.map((group) => (
                <div key={group.slug} className="space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-foreground/60">
                    {group.name}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {group.values.map((value) => {
                      const selected = selectedOptions[group.slug] === value;
                      const available = isOptionAvailable(group.slug, value);
                      const swatchColor = group.isColor ? getColorSwatch(value) : null;
                      return (
                        <button
                          key={value}
                          type="button"
                          disabled={!available}
                          onClick={() => handleOptionSelect(group.slug, value)}
                          className={cn(
                            "inline-flex min-h-9 items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em]",
                            selected
                              ? "border-foreground bg-foreground text-background"
                              : "border-border text-foreground/70 hover:border-foreground",
                            !available && "cursor-not-allowed opacity-40"
                          )}
                        >
                          {swatchColor ? (
                            <span
                              className="h-3 w-3 rounded-full border border-border"
                              style={{ backgroundColor: swatchColor }}
                            />
                          ) : null}
                          <span>{value}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          ) : null}

          <div className="pt-3">
            <AddToCartButton
              productId={product.id}
              variantId={variantId}
              quantity={quantity}
              size="sm"
              variant="primary"
              className="h-11 w-full justify-center"
              disabled={!inStock}
              label={inStock ? "Add to cart" : "Out of stock"}
            />
          </div>

          {!inStock ? <BackInStockForm product={product} variantId={variantId} /> : null}

          {/* Collapsible Sections */}
          <div className="border-t border-border pt-4 space-y-3">
            {/* Description */}
            <section id="description" className="space-y-2 border-b border-border pb-3">
              <button
                type="button"
                onClick={() => setDescriptionExpanded(!descriptionExpanded)}
                className="flex w-full items-center justify-between"
              >
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
                  Description
                </h2>
                <ChevronDown
                  size={16}
                  className={cn(
                    "transition-transform",
                    descriptionExpanded && "rotate-180"
                  )}
                />
              </button>
              {descriptionExpanded && (
                <div className="pt-2 space-y-2">
                  <p className="text-sm text-foreground/70">
                    {fullDescription || "No description available."}
                  </p>
                </div>
              )}
            </section>

            {/* Size Chart */}
            <section id="size-chart" className="space-y-2 border-b border-border pb-3">
              <button
                type="button"
                onClick={() => setSizeChartExpanded(!sizeChartExpanded)}
                className="flex w-full items-center justify-between"
              >
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
                  Size Chart
                </h2>
                <ChevronDown
                  size={16}
                  className={cn(
                    "transition-transform",
                    sizeChartExpanded && "rotate-180"
                  )}
                />
              </button>
              {sizeChartExpanded && (
                <div className="pt-2 space-y-4">
                  {product.size_charts?.length ? (
                    product.size_charts.map((link, chartIdx) => {
                      const chart = link.size_chart;
                      return (
                        <div key={chart.id || chartIdx} className="space-y-2">
                          <div className="flex items-center justify-between">
                            <p className="text-sm font-semibold text-foreground/80">{chart.name}</p>
                            <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] uppercase tracking-wider text-foreground/50">
                              {chart.unit}
                            </span>
                          </div>
                          {chart.description ? (
                            <p className="text-xs text-foreground/60">{chart.description}</p>
                          ) : null}
                          {chart.columns?.length && chart.rows?.length ? (
                            <div className="overflow-x-auto rounded-lg border border-border">
                              <table className="w-full text-sm">
                                <thead>
                                  <tr className="border-b border-border bg-muted/50">
                                    {chart.columns.map((col, colIdx) => (
                                      <th
                                        key={colIdx}
                                        className="whitespace-nowrap px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-foreground/60"
                                      >
                                        {col}
                                      </th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {chart.rows.map((row, rowIdx) => (
                                    <tr
                                      key={rowIdx}
                                      className={cn(
                                        "border-b border-border/50 transition-colors hover:bg-muted/30",
                                        rowIdx % 2 === 1 && "bg-muted/20"
                                      )}
                                    >
                                      {row.map((cell, cellIdx) => (
                                        <td
                                          key={cellIdx}
                                          className={cn(
                                            "whitespace-nowrap px-3 py-2 text-foreground/70",
                                            cellIdx === 0 && "font-medium text-foreground/90"
                                          )}
                                        >
                                          {cell}
                                        </td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          ) : null}
                          {chart.fit_notes ? (
                            <p className="text-xs italic text-foreground/50">💡 {chart.fit_notes}</p>
                          ) : null}
                        </div>
                      );
                    })
                  ) : product.attributes?.length ? (
                    <div className="grid gap-2 text-sm text-foreground/70">
                      {product.attributes.map((attr) => (
                        <div key={attr.id} className="flex items-center justify-between border-b border-border/70 py-1">
                          <span className="uppercase tracking-[0.12em] text-foreground/60">
                            {attr.attribute.name}
                          </span>
                          <span>{attr.value}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-foreground/60">Size chart not available.</p>
                  )}
                </div>
              )}
            </section>

            {/* More Information */}
            <section id="more-info" className="space-y-2 border-b border-border pb-3">
              <button
                type="button"
                onClick={() => setMoreInfoExpanded(!moreInfoExpanded)}
                className="flex w-full items-center justify-between"
              >
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
                  More Information
                </h2>
                <ChevronDown
                  size={16}
                  className={cn(
                    "transition-transform",
                    moreInfoExpanded && "rotate-180"
                  )}
                />
              </button>
              {moreInfoExpanded && (
                <div className="pt-2 space-y-1 text-sm text-foreground/70">
                  <DetailRow label="Category" value={categoryLabel} />
                  <DetailRow label="Stock status" value={stockLabel} />
                  <DetailRow label="Available stock" value={stockQty !== null ? stockQty : null} />
                  <DetailRow label="Dimensions (L x W x H)" value={dimensions || null} />
                  <DetailRow label="Weight" value={product.weight ?? null} />
                  <DetailRow label="Shipping material" value={product.shipping_material?.name} />
                </div>
              )}
            </section>

            {/* Returns & Exchange */}
            <section id="returns" className="space-y-2">
              <button
                type="button"
                onClick={() => setReturnsExpanded(!returnsExpanded)}
                className="flex w-full items-center justify-between"
              >
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
                  Returns & Exchange
                </h2>
                <ChevronDown
                  size={16}
                  className={cn(
                    "transition-transform",
                    returnsExpanded && "rotate-180"
                  )}
                />
              </button>
              {returnsExpanded && (
                <div className="pt-2">
                  <p className="text-sm text-foreground/70">
                    Returns and exchanges are accepted within 7 days of delivery when items are unused,
                    unwashed, and in original packaging.
                  </p>
                </div>
              )}
            </section>
          </div>
        </div>
      </div>

      <ShippingEstimator product={product} quantity={quantity} unitPrice={unitPrice} />

      <section className="space-y-2">
        <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
          Recently viewed
        </h2>
        <RecentlyViewedSection
          excludeProductId={product.id}
          excludeProductSlug={product.slug}
        />
      </section>

      {relatedProducts && relatedProducts.length ? (
        <section className="space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
            Related Products
          </h2>
          <ProductGrid products={relatedProducts} cardStyle="minimal" />
        </section>
      ) : null}

      <section id="reviews" className="space-y-2">
        <ProductReviews product={product} reviewStatsQuery={reviewStatsQuery} />
      </section>
    </div>
  );
}

