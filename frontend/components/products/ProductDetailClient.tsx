"use client";

import * as React from "react";
import Link from "next/link";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type {
  ProductDetail,
  ProductListItem,
  ProductReviewsResponse,
  ProductQuestion,
  CustomerPhoto,
} from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { AddToCartButton } from "@/components/cart/AddToCartButton";
import { AddToWishlistButton } from "@/components/wishlist/AddToWishlistButton";
import { RatingStars } from "@/components/products/RatingStars";
import { ProductBadges } from "@/components/products/ProductBadges";
import { ProductPrice } from "@/components/products/ProductPrice";
import { ProductGrid } from "@/components/products/ProductGrid";
import { useToast } from "@/components/ui/ToastProvider";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { addRecentlyViewed } from "@/lib/recentlyViewed";
import { compareItemFromProduct, useCompareToggle } from "@/components/products/compareHelpers";
import { cn } from "@/lib/utils";
import { RecentlyViewedSection } from "@/components/products/RecentlyViewedSection";

type Variant = NonNullable<ProductDetail["variants"]>[number];

function buildVariantLabel(variant: Variant) {
  if (variant.option_values?.length) {
    return variant.option_values
      .map((value) => `${value.option.name}: ${value.value}`)
      .join(" / ");
  }
  return variant.sku ? `SKU ${variant.sku}` : `Variant ${variant.id.slice(0, 6)}`;
}

function toNumber(value: string | number | null | undefined) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = typeof value === "number" ? value : Number(String(value));
  return Number.isFinite(parsed) ? parsed : null;
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
    <div className="flex items-center justify-between gap-4 text-sm">
      <span className="text-foreground/60">{label}</span>
      <span className="text-right">{value}</span>
    </div>
  );
}

function ProductGallery({ product }: { product: ProductDetail }) {
  const images = product.images || [];
  const [active, setActive] = React.useState(0);
  const activeImage = images[active]?.image || images[0]?.image || null;

  return (
    <div className="space-y-4">
      <div className="aspect-[4/5] overflow-hidden rounded-2xl bg-muted">
        {activeImage ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={activeImage} alt={product.name} className="h-full w-full object-cover" />
        ) : null}
      </div>
      {images.length > 1 ? (
        <div className="grid grid-cols-4 gap-2">
          {images.slice(0, 8).map((image, index) => (
            <button
              key={image.id}
              type="button"
              onClick={() => setActive(index)}
              className={cn(
                "aspect-square overflow-hidden rounded-xl border",
                index === active ? "border-primary" : "border-border"
              )}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={image.image} alt={image.alt_text || product.name} className="h-full w-full object-cover" />
            </button>
          ))}
        </div>
      ) : null}
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
                View 3D asset
              </a>
            ))}
          </div>
        </Card>
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
    <Card variant="bordered" className="space-y-3">
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
  const [country, setCountry] = React.useState("US");
  const [state, setState] = React.useState("");
  const [postalCode, setPostalCode] = React.useState("");
  const [result, setResult] = React.useState<any>(null);
  const [loading, setLoading] = React.useState(false);
  const safeQuantity = Math.max(1, Number.isFinite(quantity) ? quantity : 1);
  const unitPriceValue =
    toNumber(unitPrice) ??
    toNumber(product.current_price) ??
    toNumber(product.price) ??
    0;
  const subtotal = unitPriceValue * safeQuantity;
  const baseWeight = toNumber(product.weight);
  const totalWeight = baseWeight ? baseWeight * safeQuantity : undefined;

  React.useEffect(() => {
    setResult(null);
  }, [safeQuantity, unitPrice]);

  const handleEstimate = async () => {
    setLoading(true);
    try {
      const response = await apiFetch("/shipping/calculate/", {
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
      setResult((response as { data?: any }).data || null);
    } catch {
      push("Could not estimate shipping.", "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card variant="bordered" className="space-y-3">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold">Shipping estimate</h3>
        <p className="text-xs text-foreground/60">
          Calculated for {safeQuantity} item{safeQuantity === 1 ? "" : "s"}
        </p>
      </div>
      <div className="grid gap-2 sm:grid-cols-3">
        <input
          value={country}
          onChange={(event) => setCountry(event.target.value)}
          className="h-10 rounded-xl border border-border bg-transparent px-3 text-sm"
          placeholder="Country"
        />
        <input
          value={state}
          onChange={(event) => setState(event.target.value)}
          className="h-10 rounded-xl border border-border bg-transparent px-3 text-sm"
          placeholder="State"
        />
        <input
          value={postalCode}
          onChange={(event) => setPostalCode(event.target.value)}
          className="h-10 rounded-xl border border-border bg-transparent px-3 text-sm"
          placeholder="Postal code"
        />
      </div>
      <Button size="sm" variant="secondary" onClick={handleEstimate} disabled={loading}>
        {loading ? "Estimating..." : "Get rates"}
      </Button>
      {result?.methods?.length ? (
        <div className="space-y-2 text-xs text-foreground/70">
          {result.methods.map((method: any) => (
            <div key={method.code || method.name} className="flex items-center justify-between">
              <span>{method.name}</span>
              <span>{method.rate_display || method.rate || "-"}</span>
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
  const [page, setPage] = React.useState(1);
  const [rating, setRating] = React.useState(5);
  const [title, setTitle] = React.useState("");
  const [body, setBody] = React.useState("");

  const reviewsQuery = useQuery({
    queryKey: ["product", product.slug, "reviews", page],
    queryFn: async () => {
      const response = await apiFetch<ProductReviewsResponse>(
        `/catalog/products/${product.slug}/reviews/`,
        { params: { page } }
      );
      return response.data;
    },
  });

  const addReview = useMutation({
    mutationFn: async () => {
      return apiFetch(`/catalog/products/${product.slug}/add_review/`, {
        method: "POST",
        body: { rating, title, body },
      });
    },
    onSuccess: () => {
      push("Review submitted. Pending approval.", "success");
      setTitle("");
      setBody("");
      reviewsQuery.refetch();
    },
    onError: () => push("Could not submit review.", "error"),
  });

  const summary = reviewsQuery.data?.summary;

  return (
    <Card variant="bordered" className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold">Customer reviews</h3>
          {summary ? (
            <p className="text-sm text-foreground/60">
              {summary.average} out of 5 ({summary.total} reviews)
            </p>
          ) : null}
        </div>
        {summary ? <RatingStars rating={summary.average} count={summary.total} /> : null}
      </div>

      {reviewsQuery.data?.reviews?.length ? (
        <div className="space-y-4">
          {reviewsQuery.data.reviews.map((review) => (
            <div key={review.id} className="border-b border-border pb-4">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold">{review.user_name || "Customer"}</p>
                <RatingStars rating={review.rating} showCount={false} />
              </div>
              <p className="text-sm text-foreground/70">{review.title}</p>
              <p className="text-sm text-foreground/60">{review.body}</p>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-foreground/60">No reviews yet.</p>
      )}

      <div className="flex items-center justify-between text-xs text-foreground/60">
        <Button
          size="sm"
          variant="ghost"
          disabled={page <= 1}
          onClick={() => setPage((prev) => Math.max(1, prev - 1))}
        >
          Previous
        </Button>
        <span>Page {page}</span>
        <Button
          size="sm"
          variant="ghost"
          disabled={page >= (reviewsQuery.data?.total_pages || 1)}
          onClick={() => setPage((prev) => prev + 1)}
        >
          Next
        </Button>
      </div>

      <div className="space-y-3">
        <h4 className="text-sm font-semibold">Write a review</h4>
        {!hasToken ? (
          <p className="text-xs text-foreground/60">Log in to submit a review.</p>
        ) : (
          <div className="grid gap-3">
            <div className="space-y-2">
              <label className="text-xs text-foreground/60">Rating</label>
              <select
                value={rating}
                onChange={(event) => setRating(Number(event.target.value))}
                className="h-10 w-full rounded-xl border border-border bg-card px-3 text-sm"
              >
                {[5, 4, 3, 2, 1].map((value) => (
                  <option key={value} value={value}>
                    {value} stars
                  </option>
                ))}
              </select>
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
                disabled={addReview.isPending}
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

  return (
    <Card variant="bordered" className="space-y-4">
      <h3 className="text-lg font-semibold">Questions & answers</h3>
      {questionsQuery.data?.length ? (
        <div className="space-y-4">
          {questionsQuery.data.map((item) => (
            <div key={item.id} className="space-y-2">
              <p className="text-sm font-semibold">{item.question_text}</p>
              {item.answers?.length ? (
                <div className="space-y-1 text-sm text-foreground/70">
                  {item.answers.map((answer) => (
                    <p key={answer.id}>{answer.answer_text}</p>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-foreground/60">No answers yet.</p>
              )}
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-foreground/60">No questions yet.</p>
      )}

      {hasToken ? (
        <div className="grid gap-3">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            className="min-h-[90px] rounded-xl border border-border bg-transparent px-3 py-2 text-sm"
            placeholder="Ask a question"
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
        <p className="text-xs text-foreground/60">Log in to ask a question.</p>
      )}
    </Card>
  );
}

function CustomerPhotos({ product }: { product: ProductDetail }) {
  const { push } = useToast();
  const [description, setDescription] = React.useState("");
  const [file, setFile] = React.useState<File | null>(null);

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

  return (
    <Card variant="bordered" className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Customer photos</h3>
        <span className="text-xs text-foreground/60">
          {photosQuery.data?.length || 0} photos
        </span>
      </div>
      {photosQuery.data?.length ? (
        <div className="grid gap-3 sm:grid-cols-3">
          {photosQuery.data.map((photo) => (
            <div key={photo.id} className="aspect-square overflow-hidden rounded-xl bg-muted">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={photo.image} alt={photo.description || "Customer photo"} className="h-full w-full object-cover" />
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-foreground/60">No photos yet.</p>
      )}

      <div className="grid gap-3">
        <input
          type="file"
          accept="image/*"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
          className="text-xs"
        />
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
  const [variantId, setVariantId] = React.useState<string | null>(() => {
    const defaultVariant = product.variants?.find((variant) => variant.is_default) || product.variants?.[0];
    return defaultVariant?.id || null;
  });
  const [quantity, setQuantity] = React.useState(1);
  const selectedVariant = product.variants?.find((variant) => variant.id === variantId) || null;
  const inStock = selectedVariant?.stock_quantity
    ? selectedVariant.stock_quantity > 0
    : product.is_in_stock;
  const stockQty =
    typeof selectedVariant?.stock_quantity === "number"
      ? selectedVariant.stock_quantity
      : typeof product.available_stock === "number"
      ? product.available_stock
      : null;
  const isLowStock = Boolean(product.is_low_stock) || (stockQty !== null && stockQty > 0 && stockQty <= 5);

  const { isInCompare, toggleCompare } = useCompareToggle(product);

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
    product.price ||
    "0";

  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      push("Link copied.", "success");
    } catch {
      push("Could not copy link.", "error");
    }
  };

  const stockLabel = !inStock ? "Out of stock" : isLowStock ? "Low stock" : "In stock";
  const stockHint = stockQty !== null ? `${stockQty} available` : null;
  const dimensions = [product.length, product.width, product.height]
    .filter((value) => value !== null && value !== undefined && value !== "")
    .map((value) => String(value))
    .join(" × ");
  const categoryLabel =
    product.primary_category?.name ||
    (product.categories?.length
      ? product.categories.map((category) => category.name).join(", ")
      : null);

  return (
    <div className="space-y-12">
      <div className="grid gap-10 lg:grid-cols-[1.1fr_1fr]">
        <ProductGallery product={product} />

        <div className="flex flex-col gap-6">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              {product.primary_category?.name || "Catalog"}
            </p>
            <h1 className="text-3xl font-semibold sm:text-4xl">
              {product.name}
            </h1>
            <p className="mt-3 text-foreground/70">
              {product.short_description}
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <RatingStars rating={product.average_rating || 0} count={product.reviews_count} />
              <ProductBadges product={product} badges={product.badges} />
            </div>
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
              priceClassName="text-2xl"
            />
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className={cn(inStock ? "text-success-500" : "text-error-500")}>
                {stockLabel}
              </span>
              {stockHint ? (
                <span className="text-xs text-foreground/60">({stockHint})</span>
              ) : null}
            </div>

            {product.variants?.length ? (
              <div className="space-y-2">
                <label className="text-xs text-foreground/60">Variant</label>
                <select
                  value={variantId || ""}
                  onChange={(event) => setVariantId(event.target.value)}
                  className="h-10 w-full rounded-xl border border-border bg-card px-3 text-sm"
                >
                  {product.variants.map((variant) => (
                    <option key={variant.id} value={variant.id}>
                      {buildVariantLabel(variant)}
                    </option>
                  ))}
                </select>
              </div>
            ) : null}

            {inStock ? (
              <div className="space-y-2">
                <label className="text-xs text-foreground/60">Quantity</label>
                <div className="inline-flex items-center rounded-xl border border-border bg-card">
                  <button
                    type="button"
                    className="h-10 w-10 text-lg text-foreground/70 transition hover:text-foreground disabled:opacity-40"
                    onClick={() => setQuantity((prev) => clampQuantity(prev - 1))}
                    disabled={quantity <= 1}
                    aria-label="Decrease quantity"
                  >
                    -
                  </button>
                  <input
                    type="number"
                    min={1}
                    max={maxQty ?? undefined}
                    value={quantity}
                    onChange={(event) =>
                      setQuantity(clampQuantity(Number(event.target.value)))
                    }
                    className="h-10 w-16 bg-transparent text-center text-sm"
                  />
                  <button
                    type="button"
                    className="h-10 w-10 text-lg text-foreground/70 transition hover:text-foreground disabled:opacity-40"
                    onClick={() => setQuantity((prev) => clampQuantity(prev + 1))}
                    disabled={maxQty !== null && quantity >= maxQty}
                    aria-label="Increase quantity"
                  >
                    +
                  </button>
                </div>
                {maxQty !== null ? (
                  <p className="text-xs text-foreground/60">
                    Max {maxQty} per order
                  </p>
                ) : null}
              </div>
            ) : null}

            <div className="flex flex-nowrap items-center gap-3 overflow-x-auto pb-1 sm:overflow-visible">
              <AddToCartButton
                productId={product.id}
                variantId={variantId}
                quantity={quantity}
                size="sm"
                variant="primary"
                disabled={!inStock}
                label={inStock ? "Add to cart" : "Out of stock"}
              />
              <AddToWishlistButton
                productId={product.id}
                variantId={variantId}
                size="sm"
              />
              <Button
                size="sm"
                variant={isInCompare ? "primary" : "secondary"}
                onClick={() => toggleCompare(compareItemFromProduct(product))}
              >
                {isInCompare ? "Compare" : "Add to compare"}
              </Button>
              <Button size="sm" variant="secondary" onClick={handleShare}>
                <svg
                  aria-hidden="true"
                  viewBox="0 0 24 24"
                  className="h-4 w-4"
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

          <Card variant="bordered" className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-lg font-semibold">Details</h2>
              {product.sku ? (
                <span className="text-xs text-foreground/60">SKU {product.sku}</span>
              ) : null}
            </div>
            <p className="text-sm text-foreground/70">
              {product.description || "Product details will appear here."}
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
                    label="Dimensions (L×W×H)"
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
                <div className="grid gap-2 text-sm">
                  {product.attributes.map((attr) => (
                    <div key={attr.id} className="flex justify-between">
                      <span className="text-foreground/60">{attr.attribute.name}</span>
                      <span>{attr.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
            {product.tags?.length ? (
              <div className="space-y-2">
                <p className="text-xs text-foreground/60">Tags</p>
                <div className="flex flex-wrap gap-2">
                  {product.tags.map((tag) => (
                    <span
                      key={tag.id}
                      className="rounded-full border border-border px-3 py-1 text-xs"
                    >
                      {tag.name}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
            {product.shipping_material?.notes ? (
              <p className="text-xs text-foreground/60">
                {product.shipping_material.notes}
              </p>
            ) : null}
          </Card>

          <ShippingEstimator product={product} quantity={quantity} unitPrice={unitPrice} />
        </div>
      </div>

      {(product.eco_certifications?.length || product.material_breakdown) ? (
        <Card variant="bordered" className="space-y-4">
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
        <ProductReviews product={product} />
        <ProductQuestions product={product} />
      </div>

      <CustomerPhotos product={product} />

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

      <RecentlyViewedSection />
    </div>
  );
}
