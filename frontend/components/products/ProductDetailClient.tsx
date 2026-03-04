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
  ProductQuestion,
  CustomerPhoto,
  ShippingRateResponse,
  ShippingMethodOption,
} from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { AddToCartButton } from "@/components/cart/AddToCartButton";
import { AddToWishlistButton } from "@/components/wishlist/AddToWishlistButton";
import { RatingStars } from "@/components/products/RatingStars";
import { ProductBadges } from "@/components/products/ProductBadges";
import { ProductPrice } from "@/components/products/ProductPrice";
import { formatMoney } from "@/lib/money";
import { ProductGrid } from "@/components/products/ProductGrid";
import { useToast } from "@/components/ui/ToastProvider";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { addRecentlyViewed } from "@/lib/recentlyViewed";
import { compareItemFromProduct, useCompareToggle } from "@/components/products/compareHelpers";
import { cn } from "@/lib/utils";
import { RecentlyViewedSection } from "@/components/products/RecentlyViewedSection";
import { buildProductPath } from "@/lib/productPaths";

type Variant = NonNullable<ProductDetail["variants"]>[number];
type VariantOptionMap = Record<string, string>;
type SectionLink = { id: string; label: string };

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

function buildVariantLabel(variant: Variant) {
  if (variant.option_values?.length) {
    return variant.option_values
      .map((value) => `${value.option.name}: ${value.value}`)
      .join(" / ");
  }
  return `Variant ${variant.id.slice(0, 6)}`;
}

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

function SectionAnchorNav({ sections }: { sections: SectionLink[] }) {
  const [activeSection, setActiveSection] = React.useState<string>(sections[0]?.id || "");

  React.useEffect(() => {
    if (!sections.length) {
      setActiveSection("");
      return;
    }
    setActiveSection(sections[0].id);
  }, [sections]);

  React.useEffect(() => {
    if (!sections.length) return;

    const sectionIds = new Set(sections.map((section) => section.id));
    const syncFromHash = () => {
      const hash = window.location.hash.replace("#", "");
      if (hash && sectionIds.has(hash)) {
        setActiveSection(hash);
      }
    };

    syncFromHash();
    window.addEventListener("hashchange", syncFromHash);

    const observer = new IntersectionObserver(
      (entries) => {
        const visibleEntries = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (!visibleEntries.length) return;
        const nextActive = visibleEntries[0].target.id;
        if (nextActive) setActiveSection(nextActive);
      },
      {
        rootMargin: "-35% 0px -55% 0px",
        threshold: [0.15, 0.35, 0.6],
      }
    );

    sections.forEach((section) => {
      const element = document.getElementById(section.id);
      if (element) observer.observe(element);
    });

    return () => {
      window.removeEventListener("hashchange", syncFromHash);
      observer.disconnect();
    };
  }, [sections]);

  if (!sections.length) return null;
  return (
    <nav
      aria-label="Product sections"
      className="rounded-2xl border border-border bg-card/90 px-2 py-2 shadow-soft backdrop-blur"
    >
      <div className="scrollbar-hide overflow-x-auto">
        <ul className="flex min-w-max items-center gap-2">
          {sections.map((section) => {
            const isActive = activeSection === section.id;
            return (
              <li key={section.id}>
                <a
                  href={`#${section.id}`}
                  aria-current={isActive ? "location" : undefined}
                  onClick={() => setActiveSection(section.id)}
                  className={cn(
                    "inline-flex min-h-10 items-center rounded-xl border px-3 py-2 text-sm font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                    isActive
                      ? "border-primary/40 bg-primary/10 text-primary shadow-sm"
                      : "border-border/70 bg-background/70 text-foreground/75 hover:border-primary/30 hover:bg-muted hover:text-foreground"
                  )}
                >
                  {section.label}
                </a>
              </li>
            );
          })}
        </ul>
      </div>
    </nav>
  );
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

function ProductGallery({ product }: { product: ProductDetail }) {
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

  return (
    <div className="space-y-4">
      <div
        className="relative aspect-[4/5] w-full overflow-hidden rounded-2xl bg-muted lg:mx-auto lg:max-w-[500px]"
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
            src={activeImage.image}
            alt={activeImage.alt}
            className={cn(
              "h-full w-full object-cover transition-transform duration-300",
              zoomActive ? "scale-110" : "scale-100",
              isZoomed ? "cursor-zoom-out" : "cursor-zoom-in"
            )}
            style={{ transformOrigin: zoomOrigin }}
          />
        ) : null}
        {hasMultipleImages ? (
          <>
            <button
              type="button"
              onClick={goPrev}
              className="absolute left-3 top-1/2 -translate-y-1/2 rounded-full bg-background/85 p-2 text-foreground transition hover:bg-background"
              aria-label="Previous image"
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 20 20"
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              >
                <path d="M12.5 4.5L7 10l5.5 5.5" />
              </svg>
            </button>
            <button
              type="button"
              onClick={goNext}
              className="absolute right-3 top-1/2 -translate-y-1/2 rounded-full bg-background/85 p-2 text-foreground transition hover:bg-background"
              aria-label="Next image"
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 20 20"
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              >
                <path d="M7.5 4.5L13 10l-5.5 5.5" />
              </svg>
            </button>
          </>
        ) : null}
        <div className="pointer-events-none absolute bottom-3 right-3">
          <span className="rounded-full bg-background/80 px-2 py-1 text-xs text-foreground/70">
            {images.length ? `${active + 1}/${images.length}` : "1/1"}
          </span>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2">
        {hasMultipleImages ? (
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
                <img src={image.image} alt={image.alt} className="h-full w-full object-cover" />
              </button>
            ))}
          </div>
        ) : null}
        <Button size="sm" variant="secondary" onClick={() => setLightboxOpen(true)}>
          View fullscreen
        </Button>
      </div>

      {product.assets_3d && product.assets_3d.length ? (
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

      {lightboxOpen && activeImage ? (
        <div className="fixed inset-0 z-70 flex items-center justify-center bg-black/85 p-4">
          <button
            type="button"
            className="absolute right-4 top-4 rounded-full bg-white/15 p-2 text-white transition hover:bg-white/25"
            onClick={() => setLightboxOpen(false)}
            aria-label="Close fullscreen gallery"
          >
            <svg
              aria-hidden="true"
              viewBox="0 0 20 20"
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M5 5l10 10M15 5L5 15" />
            </svg>
          </button>
          {hasMultipleImages ? (
            <button
              type="button"
              onClick={goPrev}
              className="absolute left-4 rounded-full bg-white/15 p-2 text-white transition hover:bg-white/25"
              aria-label="Previous image"
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 20 20"
                className="h-6 w-6"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              >
                <path d="M12.5 4.5L7 10l5.5 5.5" />
              </svg>
            </button>
          ) : null}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={activeImage.image}
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
              <svg
                aria-hidden="true"
                viewBox="0 0 20 20"
                className="h-6 w-6"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              >
                <path d="M7.5 4.5L13 10l-5.5 5.5" />
              </svg>
            </button>
          ) : null}
        </div>
      ) : null}
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
          className="h-10 w-full min-w-0 rounded-xl border border-border bg-transparent px-3 text-sm sm:h-9 sm:flex-1 sm:px-2 sm:text-xs"
          placeholder="Country"
        />
        <input
          value={state}
          onChange={(event) => setState(event.target.value)}
          className="h-10 w-full min-w-0 rounded-xl border border-border bg-transparent px-3 text-sm sm:h-9 sm:flex-1 sm:px-2 sm:text-xs"
          placeholder="State"
        />
        <input
          value={postalCode}
          onChange={(event) => setPostalCode(event.target.value)}
          className="h-10 w-full min-w-0 rounded-xl border border-border bg-transparent px-3 text-sm sm:h-9 sm:flex-1 sm:px-2 sm:text-xs"
          placeholder="Postal code"
        />
        <Button
          size="sm"
          variant="secondary"
          onClick={handleEstimate}
          disabled={loading}
          className="h-10 w-full px-3 text-sm sm:h-9 sm:w-auto sm:shrink-0 sm:text-xs"
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

function ProductReviews({ product }: { product: ProductDetail }) {
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

  const reviewStatsQuery = useQuery({
    queryKey: ["product", product.id, "review-stats"],
    queryFn: async () => {
      const response = await apiFetch<ReviewStatistics>(
        `/reviews/product/${product.id}/statistics/`
      );
      return response.data;
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
      reviewStatsQuery.refetch();
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

  const summary = reviewStatsQuery.data;
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
          {reviewsQuery.isFetching || reviewStatsQuery.isFetching ? "Loading reviews..." : "No reviews yet."}
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

function ProductQuestions({ product }: { product: ProductDetail }) {
  const { hasToken } = useAuthContext();
  const { push } = useToast();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [question, setQuestion] = React.useState("");

  const questionsQuery = useQuery({
    queryKey: ["product", product.id, "questions"],
    queryFn: async () => {
      const response = await apiFetch<ProductQuestion[]>(
        `/catalog/products/${product.id}/questions/`
      );
      return response.data;
    },
  });

  const askQuestion = useMutation({
    mutationFn: async () => {
      return apiFetch(`/catalog/products/${product.id}/questions/`, {
        method: "POST",
        body: { question_text: question },
      });
    },
    onSuccess: () => {
      push("Question submitted for review.", "success");
      setQuestion("");
      questionsQuery.refetch();
    },
    onError: () => push("Could not submit question.", "error"),
  });

  const items = React.useMemo(() => {
    const list = questionsQuery.data || [];
    return [...list].sort((a, b) => {
      const aTime = a.created_at ? new Date(a.created_at).getTime() : 0;
      const bTime = b.created_at ? new Date(b.created_at).getTime() : 0;
      return bTime - aTime;
    });
  }, [questionsQuery.data]);
  const search = searchParams?.toString();
  const nextHref = `${pathname}${search ? `?${search}` : ""}#questions`;
  const loginHref = `/account/login/?next=${encodeURIComponent(nextHref)}`;

  return (
    <Card variant="bordered" className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Questions and answers</h3>
        <span className="text-xs text-foreground/60">{items.length} questions</span>
      </div>
      {items.length ? (
        <div className="space-y-3">
          {items.map((item) => (
            <article key={item.id} className="rounded-xl border border-border/70 p-4">
              <p className="text-sm font-semibold">{item.question_text}</p>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-foreground/60">
                {item.user_name ? <span>{item.user_name}</span> : null}
                {item.created_at ? <span>{formatDateLabel(item.created_at)}</span> : null}
                <span>{item.answers?.length || 0} answer(s)</span>
              </div>
              {item.answers?.length ? (
                <div className="mt-3 space-y-2 border-l border-border pl-3 text-sm text-foreground/70">
                  {item.answers.map((answer) => (
                    <p key={answer.id}>{answer.answer_text}</p>
                  ))}
                </div>
              ) : (
                <p className="mt-2 text-xs text-foreground/60">No answers yet.</p>
              )}
            </article>
          ))}
        </div>
      ) : (
        <p className="text-sm text-foreground/60">
          {questionsQuery.isFetching ? "Loading questions..." : "No questions yet."}
        </p>
      )}

      {hasToken ? (
        <div className="grid gap-3">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            className="min-h-[90px] rounded-xl border border-border bg-transparent px-3 py-2 text-sm"
            placeholder="Ask a question about fit, material, or delivery."
          />
          <div className="flex justify-end">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => askQuestion.mutate()}
              disabled={askQuestion.isPending || !question.trim()}
            >
              {askQuestion.isPending ? "Sending..." : "Submit question"}
            </Button>
          </div>
        </div>
      ) : (
        <p className="text-xs text-foreground/60">
          Log in to ask a question.{" "}
          <Link href={loginHref} className="text-primary underline-offset-2 hover:underline">
            Sign in
          </Link>
        </p>
      )}
    </Card>
  );
}

function CustomerPhotos({ product }: { product: ProductDetail }) {
  const { hasToken } = useAuthContext();
  const { push } = useToast();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [description, setDescription] = React.useState("");
  const [file, setFile] = React.useState<File | null>(null);
  const [activePhoto, setActivePhoto] = React.useState<CustomerPhoto | null>(null);

  const photosQuery = useQuery({
    queryKey: ["product", product.slug, "photos"],
    queryFn: async () => {
      const response = await apiFetch<CustomerPhoto[]>(
        `/catalog/products/${product.slug}/customer-photos/`
      );
      return response.data;
    },
  });

  const upload = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("Missing file");
      const formData = new FormData();
      formData.append("product", product.id);
      formData.append("image", file);
      if (description) formData.append("description", description);
      return apiFetch(`/catalog/customer-photos/upload/`, {
        method: "POST",
        body: formData,
      });
    },
    onSuccess: () => {
      push("Photo uploaded and pending review.", "success");
      setDescription("");
      setFile(null);
      photosQuery.refetch();
    },
    onError: () => push("Could not upload photo.", "error"),
  });

  const handleFileChange = (nextFile: File | null) => {
    if (!nextFile) {
      setFile(null);
      return;
    }
    const maxSize = 8 * 1024 * 1024;
    if (nextFile.size > maxSize) {
      push("File is too large. Maximum size is 8MB.", "error");
      return;
    }
    setFile(nextFile);
  };
  const search = searchParams?.toString();
  const nextHref = `${pathname}${search ? `?${search}` : ""}#photos`;
  const loginHref = `/account/login/?next=${encodeURIComponent(nextHref)}`;

  return (
    <Card variant="bordered" className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Customer photos</h3>
        <span className="text-xs text-foreground/60">
          {photosQuery.data?.length || 0} photos
        </span>
      </div>
      {photosQuery.data?.length ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {photosQuery.data.map((photo) => (
            <button
              key={photo.id}
              type="button"
              className="group aspect-square overflow-hidden rounded-xl bg-muted"
              onClick={() => setActivePhoto(photo)}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={photo.image}
                alt={photo.description || "Customer photo"}
                className="h-full w-full object-cover transition group-hover:scale-105"
              />
            </button>
          ))}
        </div>
      ) : (
        <p className="text-sm text-foreground/60">
          {photosQuery.isFetching ? "Loading photos..." : "No photos yet."}
        </p>
      )}

      {hasToken ? (
        <div className="grid gap-3">
          <label className="space-y-1 text-xs text-foreground/60">
            <span>Upload your photo (JPG/PNG, max 8MB)</span>
            <input
              type="file"
              accept="image/*"
              onChange={(event) => handleFileChange(event.target.files?.[0] || null)}
              className="block w-full text-xs"
            />
          </label>
          <textarea
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            className="min-h-[80px] rounded-xl border border-border bg-transparent px-3 py-2 text-sm"
            placeholder="Describe your photo (optional)"
          />
          <div className="flex justify-end">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => upload.mutate()}
              disabled={upload.isPending || !file}
            >
              {upload.isPending ? "Uploading..." : "Upload photo"}
            </Button>
          </div>
        </div>
      ) : (
        <p className="text-xs text-foreground/60">
          Log in to upload a photo.{" "}
          <Link href={loginHref} className="text-primary underline-offset-2 hover:underline">
            Sign in
          </Link>
        </p>
      )}

      {activePhoto ? (
        <div className="fixed inset-0 z-70 flex items-center justify-center bg-black/85 p-4">
          <button
            type="button"
            className="absolute right-4 top-4 rounded-full bg-white/15 p-2 text-white transition hover:bg-white/25"
            onClick={() => setActivePhoto(null)}
            aria-label="Close photo preview"
          >
            <svg
              aria-hidden="true"
              viewBox="0 0 20 20"
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M5 5l10 10M15 5L5 15" />
            </svg>
          </button>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={activePhoto.image}
            alt={activePhoto.description || "Customer photo"}
            className="max-h-[88vh] max-w-[92vw] rounded-xl object-contain"
          />
        </div>
      ) : null}
    </Card>
  );
}


export function ProductDetailClient({
  product,
  relatedProducts,
}: {
  product: ProductDetail;
  relatedProducts: ProductListItem[];
}) {
  const { push } = useToast();
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
  const mobileStickyBarRef = React.useRef<HTMLDivElement | null>(null);

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
  }, [defaultVariant, product.id]);

  React.useEffect(() => {
    if (typeof document === "undefined") return;
    document.body.classList.add("has-mobile-sticky-cta");
    let resizeObserver: ResizeObserver | null = null;

    const updateFooterClearance = () => {
      const isMobile = window.matchMedia("(max-width: 1023px)").matches;
      if (!isMobile) {
        document.body.style.removeProperty("--mobile-sticky-footer-clearance");
        return;
      }
      const stickyHeight = mobileStickyBarRef.current?.getBoundingClientRect().height ?? 0;
      const nextClearance = Math.max(0, Math.ceil(stickyHeight));
      document.body.style.setProperty("--mobile-sticky-footer-clearance", `${nextClearance}px`);
    };

    const handleResize = () => updateFooterClearance();
    window.addEventListener("resize", handleResize);
    window.requestAnimationFrame(updateFooterClearance);

    if (typeof ResizeObserver !== "undefined" && mobileStickyBarRef.current) {
      resizeObserver = new ResizeObserver(updateFooterClearance);
      resizeObserver.observe(mobileStickyBarRef.current);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      resizeObserver?.disconnect();
      document.body.style.removeProperty("--mobile-sticky-footer-clearance");
      document.body.classList.remove("has-mobile-sticky-cta");
    };
  }, []);

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

  const isOptionInStock = React.useCallback(
    (groupSlug: string, value: string) => {
      const selection = { ...selectedOptions, [groupSlug]: value };
      return variants.some(
        (variant) =>
          variantMatchesSelection(variant, selection) && getVariantInStock(variant, product)
      );
    },
    [product, selectedOptions, variantMatchesSelection, variants]
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

  const { isInCompare, toggleCompare } = useCompareToggle(product);

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
  const unitPriceNumber = toNumber(unitPrice) ?? 0;
  const lineTotal = unitPriceNumber * quantity;
  const lineTotalLabel = formatMoney(lineTotal, product.currency || "USD");
  const comparePrice = toNumber(product.price);
  const savings =
    comparePrice !== null && comparePrice > unitPriceNumber ? comparePrice - unitPriceNumber : null;
  const savingsPercent =
    savings !== null && comparePrice
      ? Math.round((savings / comparePrice) * 100)
      : null;

  const handleShare = async () => {
    try {
      if (navigator.share) {
        await navigator.share({
          title: product.name,
          text: product.short_description || undefined,
          url: window.location.href,
        });
      } else {
        await navigator.clipboard.writeText(window.location.href);
        push("Link copied.", "success");
      }
    } catch {
      push("Could not share link.", "error");
    }
  };

  const handleCopySku = async () => {
    const sku = selectedVariant?.sku || product.sku;
    if (!sku) return;
    try {
      await navigator.clipboard.writeText(sku);
      push("SKU copied.", "success");
    } catch {
      push("Could not copy SKU.", "error");
    }
  };

  const stockLabel = !inStock ? "Out of stock" : isLowStock ? "Low stock" : "In stock";
  const stockHint = stockQty !== null ? `${stockQty} available` : null;
  const dimensions = [product.length, product.width, product.height]
    .filter((value) => value !== null && value !== undefined && value !== "")
    .map((value) => String(value))
    .join(" x ");
  const categoryLabel =
    product.primary_category?.name ||
    (product.categories?.length
      ? product.categories.map((category) => category.name).join(", ")
      : null);
  const variantLabel = selectedVariant ? buildVariantLabel(selectedVariant) : null;
  const fullDescription = product.description || product.short_description || "";
  const shouldClampDescription = fullDescription.length > 420;
  const visibleDescription =
    shouldClampDescription && !descriptionExpanded
      ? `${fullDescription.slice(0, 420).trimEnd()}...`
      : fullDescription;
  const highlightAttributes = (product.attributes || []).slice(0, 6);
  const sectionLinks: SectionLink[] = [
    { id: "overview", label: "Overview" },
    { id: "specs", label: "Specs" },
    { id: "shipping", label: "Shipping" },
    { id: "reviews", label: "Reviews" },
    { id: "questions", label: "Q&A" },
    { id: "photos", label: "Photos" },
  ];
  if (product.eco_certifications?.length || product.material_breakdown) {
    sectionLinks.splice(2, 0, { id: "sustainability", label: "Sustainability" });
  }
  return (
    <div className="space-y-8 pb-24 lg:pb-0">
      <SectionAnchorNav sections={sectionLinks} />
      <div className="grid gap-10 lg:grid-cols-[1.1fr_1fr]">
        <ProductGallery product={product} />

        <div className="flex flex-col gap-6 self-start lg:sticky lg:top-[calc(var(--header-offset)+1rem)]">
          <div id="overview">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="inline-flex items-center rounded-full border border-border/70 bg-muted/60 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-foreground/75">
                  {product.primary_category?.name || "Catalog"}
                </p>
                <h1 className="text-3xl font-semibold sm:text-4xl">
                  {product.name}
                </h1>
              </div>
            </div>
            <p className="mt-3 text-[13px] text-foreground/70">
              {visibleDescription || "No description available."}
            </p>
            {shouldClampDescription ? (
              <button
                type="button"
                className="mt-2 text-xs font-semibold text-primary"
                onClick={() => setDescriptionExpanded((prev) => !prev)}
              >
                {descriptionExpanded ? "Show less" : "Read more"}
              </button>
            ) : null}
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <RatingStars rating={product.average_rating || 0} count={product.reviews_count} />
              <ProductBadges product={product} badges={product.badges} omitOnSale />
            </div>
            {highlightAttributes.length ? (
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {highlightAttributes.map((attribute) => (
                  <div
                    key={attribute.id}
                    className="rounded-xl border border-border/70 bg-background/70 px-3 py-2"
                  >
                    <p className="text-xs uppercase tracking-[0.16em] text-foreground/55">
                      {attribute.attribute.name}
                    </p>
                    <p className="mt-1 text-sm">{attribute.value}</p>
                  </div>
                ))}
              </div>
            ) : null}
          </div>

          <Card variant="bordered" className="space-y-3">
            <ProductPrice
              price={product.price}
              salePrice={product.sale_price}
              currentPrice={
                selectedVariant?.current_price ||
                selectedVariant?.price ||
                product.current_price
              }
              currency={product.currency}
              priceClassName="text-3xl"
            />
            {savings !== null ? (
              <p className="text-sm text-success-600">
                Save {formatMoney(savings, product.currency || "USD")}
                {savingsPercent !== null ? ` (${savingsPercent}%)` : ""}
              </p>
            ) : null}
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className={cn(inStock ? "text-success-500" : "text-error-500")}>
                {stockLabel}
              </span>
              {stockHint ? (
                <span className="text-xs text-foreground/60">({stockHint})</span>
              ) : null}
            </div>
            {variantLabel ? <p className="text-xs text-foreground/60">{variantLabel}</p> : null}

            {optionGroups.length ? (
              <div className="space-y-3">
                {optionGroups.map((group) => (
                  <div key={group.slug} className="space-y-2">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-foreground/60">
                      {group.name}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {group.values.map((value) => {
                        const selected = selectedOptions[group.slug] === value;
                        const available = isOptionAvailable(group.slug, value);
                        const optionStock = isOptionInStock(group.slug, value);
                        const swatchColor = group.isColor ? getColorSwatch(value) : null;
                        return (
                          <button
                            key={value}
                            type="button"
                            disabled={!available}
                            onClick={() => handleOptionSelect(group.slug, value)}
                            className={cn(
                              "inline-flex min-h-10 items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition",
                              selected
                                ? "border-primary bg-primary/10 text-primary"
                                : "border-border text-foreground/75 hover:border-primary/40",
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
                            {available && !optionStock ? (
                              <span className="text-[10px] uppercase tracking-[0.16em] text-error-600">
                                Out
                              </span>
                            ) : null}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            ) : null}

            {variants.length ? (
              <div className="space-y-2">
                <label className="text-xs text-foreground/60">Variant</label>
                <select
                  value={variantId || ""}
                  onChange={(event) => setVariantId(event.target.value || null)}
                  className="h-10 w-full rounded-xl border border-border bg-card px-3 text-sm"
                >
                  {variants.map((variant) => (
                    <option key={variant.id} value={variant.id}>
                      {buildVariantLabel(variant)}
                    </option>
                  ))}
                </select>
              </div>
            ) : null}

            {inStock ? (
              <div className="space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <label className="text-xs text-foreground/60">Quantity</label>
                  <p className="text-xs text-foreground/60">Total: {lineTotalLabel}</p>
                </div>
                <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:gap-3">
                  <div className="grid w-full grid-cols-[2.75rem_minmax(0,1fr)_2.75rem] items-center overflow-hidden rounded-xl border border-border bg-card sm:w-auto sm:grid-cols-[2.5rem_4rem_2.5rem]">
                    <button
                      type="button"
                      className="h-11 w-full border-r border-border/70 text-2xl font-semibold text-foreground/70 transition hover:text-foreground disabled:opacity-40 sm:h-10"
                      onClick={() => setQuantity((prev) => clampQuantity(prev - 1))}
                      disabled={quantity <= 1}
                      aria-label="Decrease quantity"
                    >
                      -
                    </button>
                    <input
                      type="text"
                      value={quantity}
                      onChange={(event) => {
                        const digitsOnly = event.target.value.replace(/\D+/g, "");
                        setQuantity(clampQuantity(Number(digitsOnly || "1")));
                      }}
                      inputMode="numeric"
                      pattern="[0-9]*"
                      className="h-11 w-full min-w-0 appearance-none bg-transparent text-center text-base sm:h-10 sm:text-sm"
                    />
                    <button
                      type="button"
                      className="h-11 w-full border-l border-border/70 text-2xl font-semibold text-foreground/70 transition hover:text-foreground disabled:opacity-40 sm:h-10"
                      onClick={() => setQuantity((prev) => clampQuantity(prev + 1))}
                      disabled={maxQty !== null && quantity >= maxQty}
                      aria-label="Increase quantity"
                    >
                      +
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2 sm:flex sm:items-center sm:gap-3">
                    <AddToCartButton
                      productId={product.id}
                      variantId={variantId}
                      quantity={quantity}
                      size="sm"
                      variant="primary"
                      className="h-11 w-full sm:h-10 sm:w-auto"
                      disabled={!inStock}
                      label={inStock ? "Add to bag" : "Out of stock"}
                    />
                    <AddToWishlistButton
                      productId={product.id}
                      variantId={variantId}
                      size="sm"
                      hideIconOnMobile
                      className="h-11 w-full justify-center gap-2 sm:hidden"
                    />
                  </div>
                </div>
                {maxQty !== null ? (
                  <p className="text-xs text-foreground/60">
                    Max {maxQty} per order
                  </p>
                ) : null}
              </div>
            ) : null}

            <div className="grid grid-cols-2 gap-2 sm:flex sm:flex-nowrap sm:items-center sm:gap-3">
              <AddToWishlistButton
                productId={product.id}
                variantId={variantId}
                size="sm"
                hideIconOnMobile
                className={cn(
                  "h-11 w-full justify-center gap-2 sm:h-10 sm:w-auto",
                  inStock && "hidden sm:inline-flex"
                )}
              />
              <Button
                size="sm"
                variant={isInCompare ? "primary" : "secondary"}
                onClick={() => toggleCompare(compareItemFromProduct(product))}
                className="h-11 w-full justify-center gap-2 sm:h-10 sm:w-auto"
              >
                <svg
                  aria-hidden="true"
                  viewBox="0 0 24 24"
                  className="hidden h-4 w-4 sm:block"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="3" y="4" width="7" height="16" rx="2" />
                  <rect x="14" y="4" width="7" height="16" rx="2" />
                </svg>
                {isInCompare ? "Compared" : "Add to compare"}
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={handleShare}
                className="h-11 w-full justify-center gap-2 sm:h-10 sm:w-auto"
              >
                <svg
                  aria-hidden="true"
                  viewBox="0 0 24 24"
                  className="hidden h-5 w-5 sm:block"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M12 3v12" />
                  <path d="M8 7l4-4 4 4" />
                  <path d="M4 14v5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5" />
                </svg>
                Share
              </Button>
            </div>
          </Card>

          {!inStock ? <BackInStockForm product={product} variantId={variantId} /> : null}

          <Card variant="bordered" className="space-y-4" id="specs">
            <div className="flex flex-col items-start gap-2 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-lg font-semibold">Specifications and details</h2>
              {(selectedVariant?.sku || product.sku) ? (
                <button
                  type="button"
                  onClick={handleCopySku}
                  className="inline-flex items-center gap-1.5 rounded-full border border-border/70 bg-card/60 px-2.5 py-1 text-xs font-medium text-foreground/70 transition-colors hover:border-primary/40 hover:bg-muted/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 sm:ml-auto"
                  title="Click to copy SKU"
                >
                  <span className="uppercase tracking-[0.12em] text-foreground/55">SKU</span>
                  <span className="font-mono text-foreground/80">
                    {selectedVariant?.sku || product.sku}
                  </span>
                </button>
              ) : null}
            </div>
            <p className="text-justify text-[13px] text-foreground/70">
              {product.description ||
                product.short_description ||
                "Detailed specification information is currently being updated. Please contact support for complete details."}
            </p>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <h3 className="text-sm font-semibold">Specifications</h3>
                <div className="space-y-1">
                  <DetailRow label="Category" value={categoryLabel} />
                  <DetailRow label="Stock status" value={stockLabel} />
                  <DetailRow
                    label="Available stock"
                    value={stockQty !== null ? stockQty : null}
                  />
                  <DetailRow
                    label="Average rating"
                    value={
                      product.average_rating ? `${product.average_rating} / 5` : null
                    }
                  />
                  <DetailRow
                    label="Reviews"
                    value={typeof product.reviews_count === "number" ? product.reviews_count : null}
                  />
                  <DetailRow
                    label="Views"
                    value={typeof product.views_count === "number" ? product.views_count : null}
                  />
                  <DetailRow
                    label="Shipping material"
                    value={product.shipping_material?.name}
                  />
                </div>
              </div>
              <div className="space-y-2">
                <h3 className="text-sm font-semibold">Size & weight</h3>
                <div className="space-y-1">
                  <DetailRow
                    label="Dimensions (L x W x H)"
                    value={dimensions || null}
                  />
                  <DetailRow label="Weight" value={product.weight ?? null} />
                  <DetailRow
                    label="Packaging weight"
                    value={product.shipping_material?.packaging_weight ?? null}
                  />
                </div>
              </div>
            </div>
            {product.attributes?.length ? (
              <div className="space-y-2">
                <h3 className="text-sm font-semibold">Attributes</h3>
                <div className="grid gap-2 text-[13px]">
                  {product.attributes.map((attr) => (
                    <div key={attr.id} className="flex justify-between">
                      <span className="text-foreground/60">{attr.attribute.name}</span>
                      <span>{attr.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
            {product.shipping_material?.notes ? (
              <p className="text-justify text-xs text-foreground/60">
                {product.shipping_material.notes}
              </p>
            ) : null}
          </Card>

          <div id="shipping">
            <ShippingEstimator
              product={product}
              quantity={quantity}
              unitPrice={unitPrice}
            />
          </div>
        </div>
      </div>

      {(product.eco_certifications?.length || product.material_breakdown) ? (
        <Card variant="bordered" className="space-y-4" id="sustainability">
          <h3 className="text-lg font-semibold">Sustainability</h3>
          {product.eco_certifications?.length ? (
            <div className="flex flex-wrap gap-2">
              {product.eco_certifications.map((cert) => (
                <span
                  key={cert.id}
                  className="rounded-full border border-border px-3 py-1 text-xs"
                >
                  {cert.name}
                </span>
              ))}
            </div>
          ) : null}
          {product.material_breakdown ? (
            <div className="grid gap-2 text-sm">
              {Object.entries(product.material_breakdown).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-foreground/60">{key}</span>
                  <span>{String(value)}</span>
                </div>
              ))}
            </div>
          ) : null}
        </Card>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
        <div id="reviews">
          <ProductReviews product={product} />
        </div>
        <div id="questions">
          <ProductQuestions product={product} />
        </div>
      </div>

      <div id="photos">
        <CustomerPhotos product={product} />
      </div>

      {relatedProducts.length ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-semibold">Related products</h2>
            <Button asChild variant="ghost">
              <Link href="/products/">View all</Link>
            </Button>
          </div>
          <ProductGrid products={relatedProducts} />
        </div>
      ) : null}

      <RecentlyViewedSection
        excludeProductId={product.id}
        excludeProductSlug={product.slug}
      />

      <div
        ref={mobileStickyBarRef}
        className="fixed inset-x-0 bottom-0 z-60 border-t border-border bg-background/95 p-3 backdrop-blur lg:hidden"
      >
        <div className="mx-auto flex max-w-6xl items-center gap-3">
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs text-foreground/55">{product.name}</p>
            <p className="text-sm font-semibold">
              {formatMoney(unitPrice, product.currency || "USD")}
            </p>
          </div>
          {inStock ? (
            <AddToCartButton
              productId={product.id}
              variantId={variantId}
              quantity={quantity}
              size="sm"
              variant="primary"
              className="h-11 min-w-[10rem] justify-center"
              disabled={!inStock}
              label="Add to bag"
            />
          ) : (
            <Button
              size="sm"
              variant="secondary"
              className="h-11 min-w-[10rem]"
              onClick={() => {
                document.getElementById("back-in-stock")?.scrollIntoView({
                  behavior: "smooth",
                  block: "start",
                });
              }}
            >
              Notify me
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

