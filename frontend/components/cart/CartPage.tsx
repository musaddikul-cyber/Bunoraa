"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { apiFetch, ApiError } from "@/lib/api";
import type { CartItem, ProductListItem } from "@/lib/types";
import { useCart } from "@/components/cart/useCart";
import { useWishlist } from "@/components/wishlist/useWishlist";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useToast } from "@/components/ui/ToastProvider";
import { cn } from "@/lib/utils";

type ValidationIssue = {
  type?: string;
  message?: string;
  item_id?: string;
  product_name?: string;
  available?: number;
  old_price?: string;
  new_price?: string;
  minimum?: string;
  current?: string;
};

type ValidationResult = {
  is_valid: boolean;
  issues: ValidationIssue[];
  warnings: ValidationIssue[];
  issue_count?: number;
  warning_count?: number;
};

type ShareResult = {
  share_url?: string;
  share_token?: string;
};

type GiftResponse = {
  gift_state?: {
    is_gift?: boolean;
    gift_message?: string;
    gift_wrap?: boolean;
    gift_wrap_cost?: string;
  };
  gift_wrap_amount?: string;
  formatted_gift_wrap_amount?: string;
  formatted_gift_wrap_cost?: string;
  gift_wrap_label?: string;
  gift_wrap_enabled?: boolean;
};

type RelatedQueryKey = ["cart", "related", string | null];

async function fetchRelatedProducts(slug: string) {
  const response = await apiFetch<ProductListItem[]>(
    `/catalog/products/${encodeURIComponent(slug)}/related/`,
    { params: { limit: 4 } }
  );
  return response.data;
}

function formatMoney(amount: string | number, currency: string) {
  if (typeof amount === "string") {
    const trimmed = amount.trim();
    if (!trimmed) return "";
    if (/[^0-9.,-]/.test(trimmed)) {
      return trimmed;
    }
    const normalized = trimmed.replace(/,/g, "");
    const parsed = Number(normalized);
    if (Number.isFinite(parsed)) {
      amount = parsed;
    }
  }
  const numeric = Number(amount);
  if (!Number.isFinite(numeric)) {
    return String(amount);
  }
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency,
      maximumFractionDigits: 2,
    }).format(numeric);
  } catch {
    return `${numeric.toFixed(2)} ${currency}`;
  }
}

function getProductImage(product: ProductListItem) {
  const primary = product.primary_image as unknown as
    | string
    | { image?: string | null }
    | null;
  if (!primary) return null;
  if (typeof primary === "string") return primary;
  return primary.image || null;
}

function getProductPrice(product: ProductListItem) {
  return product.current_price || product.sale_price || product.price || "";
}

function getApiErrorMessage(error: unknown, fallback: string) {
  const asRecord =
    error && typeof error === "object" ? (error as Record<string, unknown>) : null;
  if (error instanceof ApiError) {
    const data = error.data;
    if (typeof data === "string" && data.trim()) return data.trim();
    if (data && typeof data === "object") {
      const record = data as Record<string, unknown>;
      const directMessage = record.message || record.error || record.detail;
      if (directMessage) return String(directMessage);
      if (record._text && typeof record._text === "string") {
        const text = record._text.trim();
        if (text && !text.includes("<")) return text;
      }
      if (Array.isArray(record.non_field_errors) && record.non_field_errors.length) {
        return String(record.non_field_errors[0]);
      }
      if (record.errors && typeof record.errors === "object") {
        for (const value of Object.values(record.errors as Record<string, unknown>)) {
          if (Array.isArray(value) && value.length) return String(value[0]);
          if (typeof value === "string" && value.trim()) return value.trim();
        }
      }
      for (const value of Object.values(record)) {
        if (Array.isArray(value) && value.length) return String(value[0]);
        if (typeof value === "string" && value.trim()) return value.trim();
      }
    }
    return error.message || fallback;
  }
  if (asRecord && typeof asRecord.message === "string") return asRecord.message;
  if (asRecord && typeof asRecord.error === "string") return asRecord.error;
  return fallback;
}

function CartItemRow({
  item,
  currency,
  onUpdate,
  onRemove,
  onMoveToWishlist,
  isUpdating,
  isRemoving,
  resetSignal,
  resetTargetId,
}: {
  item: CartItem;
  currency: string;
  onUpdate: (itemId: string, quantity: number) => void;
  onRemove: (itemId: string) => void;
  onMoveToWishlist: (item: CartItem) => void;
  isUpdating: boolean;
  isRemoving: boolean;
  resetSignal: number;
  resetTargetId: string | null;
}) {
  const [quantity, setQuantity] = React.useState<number>(item.quantity);
  const manualEditRef = React.useRef(false);
  const debounceRef = React.useRef<number | null>(null);

  React.useEffect(() => {
    setQuantity(item.quantity);
  }, [item.quantity]);

  React.useEffect(() => {
    if (!resetTargetId || resetTargetId !== item.id) return;
    manualEditRef.current = false;
    if (debounceRef.current) {
      window.clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    setQuantity(item.quantity);
  }, [resetSignal, resetTargetId, item.id, item.quantity]);

  const commitQuantity = React.useCallback(
    (nextValue: number) => {
      manualEditRef.current = false;
      if (debounceRef.current) {
        window.clearTimeout(debounceRef.current);
        debounceRef.current = null;
      }
      const parsed = Number.isFinite(nextValue) ? nextValue : 1;
      const safe = Math.max(1, Math.floor(parsed));
      setQuantity(safe);
      if (safe !== item.quantity) {
        onUpdate(item.id, safe);
      }
    },
    [item.id, item.quantity, onUpdate]
  );
  
  React.useEffect(() => {
    if (!manualEditRef.current) return;
    if (debounceRef.current) {
      window.clearTimeout(debounceRef.current);
    }
    debounceRef.current = window.setTimeout(() => {
      commitQuantity(quantity);
    }, 400);
    return () => {
      if (debounceRef.current) {
        window.clearTimeout(debounceRef.current);
      }
    };
  }, [quantity, commitQuantity]);

  return (
    <Card
      variant="bordered"
      className="flex flex-col gap-4 p-3 sm:flex-row sm:items-center sm:justify-between sm:p-4"
    >
      <div className="flex items-start gap-3 sm:items-center sm:gap-4">
        <div className="h-20 w-20 shrink-0 overflow-hidden rounded-xl bg-muted">
          {item.product_image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={item.product_image}
              alt={item.product_name}
              className="h-full w-full object-cover"
            />
          ) : null}
        </div>
        <div className="min-w-0">
          <Link
            href={`/products/${item.product_slug}/`}
            className="line-clamp-2 text-sm font-semibold leading-5 hover:underline"
          >
            {item.product_name}
          </Link>
          {item.variant_name ? (
            <p className="text-xs text-foreground/60">{item.variant_name}</p>
          ) : null}
          {!item.in_stock ? (
            <p className="mt-1 text-xs font-semibold text-error-500">Out of stock</p>
          ) : null}
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-3 sm:items-end">
        <div className="grid w-full grid-cols-2 gap-2 sm:w-auto sm:grid-cols-[auto_auto] sm:gap-4">
          <div className="rounded-xl border border-border/70 bg-muted/20 px-3 py-2 sm:border-0 sm:bg-transparent sm:px-0 sm:py-0">
            <p className="text-[10px] uppercase tracking-[0.15em] text-foreground/50 sm:hidden">
              Unit price
            </p>
            <p className="text-sm text-foreground/70">
              {formatMoney(item.unit_price, currency)}
            </p>
          </div>
          <div className="rounded-xl border border-border/70 bg-muted/20 px-3 py-2 text-right sm:min-w-[120px] sm:border-0 sm:bg-transparent sm:px-0 sm:py-0">
            <p className="text-[10px] uppercase tracking-[0.15em] text-foreground/50 sm:hidden">
              Line total
            </p>
            <p className="text-sm font-semibold">{formatMoney(item.total, currency)}</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 sm:justify-end">
          <div className="flex items-center gap-2">
            <button
              type="button"
              className={cn(
                "h-10 w-10 rounded-full border border-border text-base sm:h-9 sm:w-9 sm:text-sm",
                "hover:bg-muted"
              )}
              onClick={() => {
                manualEditRef.current = false;
                if (item.quantity <= 1) {
                  onRemove(item.id);
                  return;
                }
                const next = item.quantity - 1;
                setQuantity(next);
                onUpdate(item.id, next);
              }}
              disabled={isUpdating || isRemoving}
              aria-label="Decrease quantity"
            >
              -
            </button>
            <input
              type="number"
              min={1}
              inputMode="numeric"
              pattern="[0-9]*"
              className="no-spin h-10 w-14 rounded-xl border border-border bg-transparent px-2 text-center text-sm sm:h-9 sm:w-16"
              value={quantity}
              onChange={(event) => {
                const raw = event.target.value;
                if (raw === "") {
                  setQuantity(1);
                  return;
                }
                const value = Number(raw);
                manualEditRef.current = true;
                setQuantity(Number.isFinite(value) ? value : 1);
              }}
              onBlur={() => commitQuantity(quantity)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.currentTarget.blur();
                }
              }}
              disabled={isUpdating || isRemoving}
              aria-label="Quantity"
            />
            <button
              type="button"
              className={cn(
                "h-10 w-10 rounded-full border border-border text-base sm:h-9 sm:w-9 sm:text-sm",
                "hover:bg-muted"
              )}
              onClick={() => {
                manualEditRef.current = false;
                const next = item.quantity + 1;
                setQuantity(next);
                onUpdate(item.id, next);
              }}
              disabled={isUpdating || isRemoving}
              aria-label="Increase quantity"
            >
              +
            </button>
          </div>
        </div>
        <div className="grid w-full grid-cols-2 gap-2 sm:flex sm:w-auto sm:flex-wrap sm:items-center sm:justify-end">
          <Button
            variant="ghost"
            size="sm"
            className="w-full sm:w-auto"
            onClick={() => onMoveToWishlist(item)}
            disabled={isUpdating || isRemoving}
          >
            Save for later
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="w-full sm:w-auto"
            onClick={() => onRemove(item.id)}
            disabled={isUpdating || isRemoving}
          >
            Remove
          </Button>
        </div>
      </div>
    </Card>
  );
}

export function CartPage() {
  const router = useRouter();
  const pathname = usePathname();
  const { push } = useToast();
  const { hasToken } = useAuthContext();
  const {
    cartQuery,
    cartSummaryQuery,
    addItem,
    updateItem,
    removeItem,
    clearCart,
    applyCoupon,
    validateCoupon,
    removeCoupon,
    updateGiftOptions,
    validateCart,
    lockPrices,
    shareCart,
  } = useCart();
  const { addItem: addWishlistItem } = useWishlist({ enabled: hasToken });

  const [couponCode, setCouponCode] = React.useState("");
  const [giftState, setGiftState] = React.useState({
    is_gift: false,
    gift_message: "",
    gift_wrap: false,
  });
  const [giftResponse, setGiftResponse] = React.useState<GiftResponse | null>(null);
  const [validationResult, setValidationResult] = React.useState<ValidationResult | null>(null);
  const [shareState, setShareState] = React.useState({
    name: "",
    permission: "view",
    expires_days: 7,
    password: "",
  });
  const [shareResult, setShareResult] = React.useState<ShareResult | null>(null);
  const [updatingItemId, setUpdatingItemId] = React.useState<string | null>(null);
  const [removingItemId, setRemovingItemId] = React.useState<string | null>(null);
  const [resetCounter, setResetCounter] = React.useState(0);
  const [resetItemId, setResetItemId] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (cartQuery.data?.coupon_code) {
      setCouponCode(cartQuery.data.coupon_code);
    }
  }, [cartQuery.data?.coupon_code]);

  const cart = cartQuery.data;
  const summary = cartSummaryQuery.data;
  const currency = summary?.currency_code || cart?.currency || "";
  const itemCount = cart?.item_count ?? 0;

  const formattedSubtotal =
    summary?.formatted_subtotal ??
    (cart ? formatMoney(cart.subtotal, currency) : formatMoney(0, currency));
  const formattedDiscount =
    summary?.formatted_discount ??
    (cart ? formatMoney(cart.discount_amount, currency) : formatMoney(0, currency));
  const formattedShipping =
    summary?.formatted_shipping ??
    (summary?.shipping_cost
      ? formatMoney(summary.shipping_cost, currency)
      : formatMoney(0, currency));
  const formattedTax =
    summary?.formatted_tax ??
    (summary?.tax_amount ? formatMoney(summary.tax_amount, currency) : formatMoney(0, currency));
  const formattedGiftWrap =
    summary?.formatted_gift_wrap ??
    (summary?.gift_wrap_cost
      ? formatMoney(summary.gift_wrap_cost, currency)
      : formatMoney(0, currency));
  const formattedTotal =
    summary?.formatted_total ??
    (cart ? formatMoney(cart.total, currency) : formatMoney(0, currency));

  const showShipping = Boolean(summary);
  const showTax = summary?.tax_amount !== undefined && Number(summary.tax_amount) > 0;
  const showGiftWrap =
    summary?.gift_wrap_cost !== undefined &&
    (Number(summary.gift_wrap_cost) > 0 || giftState.gift_wrap);
  const shippingLabel = summary?.shipping_estimate ? "Estimated shipping" : "Shipping";
  const taxRateLabel = summary?.tax_rate
    ? `VAT (${Number(summary.tax_rate).toFixed(0)}%)`
    : "VAT";
  const shippingEstimateLabel = summary?.shipping_estimate
    ? summary?.shipping_estimate_label || ""
    : "";

  const relatedSlug = cart?.items?.[0]?.product_slug ?? null;
  const relatedQuery = useQuery<ProductListItem[], unknown, ProductListItem[], RelatedQueryKey>({
    queryKey: ["cart", "related", relatedSlug],
    queryFn: () => fetchRelatedProducts(relatedSlug || ""),
    enabled: Boolean(relatedSlug),
    staleTime: 5 * 60 * 1000,
  });

  const handleUpdateItem = async (itemId: string, quantity: number) => {
    try {
      setUpdatingItemId(itemId);
      await updateItem.mutateAsync({ itemId, quantity });
    } catch (error) {
      push(getApiErrorMessage(error, "Could not update item."), "error");
      cartQuery.refetch();
      setResetItemId(itemId);
      setResetCounter((prev) => prev + 1);
    } finally {
      setUpdatingItemId(null);
    }
  };

  const handleRemoveItem = async (itemId: string, options?: { silent?: boolean }) => {
    try {
      setRemovingItemId(itemId);
      await removeItem.mutateAsync(itemId);
      if (!options?.silent) {
        push("Item removed from cart.", "info");
      }
    } catch (error) {
      push(getApiErrorMessage(error, "Could not remove item."), "error");
    } finally {
      setRemovingItemId(null);
    }
  };

  const handleMoveToWishlist = async (item: CartItem) => {
    if (!hasToken) {
      router.push(`/account/login/?next=${encodeURIComponent(pathname || "/")}`);
      return;
    }
    try {
      await addWishlistItem.mutateAsync({
        productId: item.product_id,
        variantId: item.variant_id,
      });
      await handleRemoveItem(item.id, { silent: true });
      push("Moved to wishlist.", "success");
    } catch (error) {
      push(getApiErrorMessage(error, "Could not move item to wishlist."), "error");
    }
  };

  const handleApplyCoupon = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!couponCode.trim()) {
      push("Enter a coupon code.", "error");
      return;
    }
    const code = couponCode.trim();
    try {
      if (cart?.subtotal) {
        const validation = await validateCoupon.mutateAsync({
          code,
          subtotal: cart.subtotal,
        });
        if (!validation.success) {
          push(validation.message || "Coupon is not valid.", "error");
          return;
        }
      }
      const response = await applyCoupon.mutateAsync({ code });
      const message =
        response && typeof response === "object" && "message" in response
          ? String((response as { message?: string }).message || "")
          : "";
      push(message || "Coupon applied.", "success");
    } catch (error) {
      push(getApiErrorMessage(error, "Could not apply coupon."), "error");
    }
  };

  const handleRemoveCoupon = async () => {
    try {
      await removeCoupon.mutateAsync();
      setCouponCode("");
      push("Coupon removed.", "info");
    } catch (error) {
      push(getApiErrorMessage(error, "Could not remove coupon."), "error");
    }
  };

  const handleGiftSave = async () => {
    try {
      const response = await updateGiftOptions.mutateAsync(giftState);
      const data =
        response && typeof response === "object" && "data" in response
          ? (response as { data: GiftResponse }).data
          : null;
      setGiftResponse(data);
      push("Gift options saved.", "success");
    } catch (error) {
      push(getApiErrorMessage(error, "Could not update gift options."), "error");
    }
  };

  const handleValidateCart = async () => {
    try {
      const response = await validateCart.mutateAsync();
      const data =
        response && typeof response === "object" && "data" in response
          ? (response as { data: ValidationResult }).data
          : null;
      setValidationResult(data);
      if (data?.is_valid) {
        push("Cart validated successfully.", "success");
      } else {
        push("Cart validation completed.", "info");
      }
    } catch (error) {
      push(getApiErrorMessage(error, "Could not validate cart."), "error");
    }
  };

  const handleLockPrices = async () => {
    try {
      const response = await lockPrices.mutateAsync(24);
      const data =
        response && typeof response === "object" && "data" in response
          ? (response as { data: { locked_count?: number } }).data
          : null;
      const locked = data?.locked_count ?? 0;
      push(`Locked prices for ${locked} item(s).`, "success");
    } catch (error) {
      push(getApiErrorMessage(error, "Could not lock prices."), "error");
    }
  };

  const handleShareCart = async () => {
    try {
      const response = await shareCart.mutateAsync({
        name: shareState.name || undefined,
        permission: shareState.permission || undefined,
        expires_days: shareState.expires_days || undefined,
        password: shareState.password || undefined,
      });
      const data =
        response && typeof response === "object" && "data" in response
          ? (response as { data: ShareResult }).data
          : null;
      setShareResult(data);
      push("Share link created.", "success");
    } catch (error) {
      push(getApiErrorMessage(error, "Could not create share link."), "error");
    }
  };

  const handleClearCart = async () => {
    if (!window.confirm("Clear all items from your cart?")) return;
    try {
      await clearCart.mutateAsync();
      push("Cart cleared.", "info");
    } catch (error) {
      push(getApiErrorMessage(error, "Could not clear cart."), "error");
    }
  };

  const handleAddRecommended = async (product: ProductListItem) => {
    try {
      await addItem.mutateAsync({ productId: product.id });
      push("Added to cart.", "success");
    } catch (error) {
      push(getApiErrorMessage(error, "Could not add to cart."), "error");
    }
  };

  const handleCopyShare = async () => {
    if (!shareResult?.share_url) return;
    try {
      await navigator.clipboard.writeText(shareResult.share_url);
      push("Share link copied.", "success");
    } catch (error) {
      push(getApiErrorMessage(error, "Could not copy link."), "error");
    }
  };

  if (cartQuery.isLoading) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-[1400px] px-4 sm:px-6 py-12">
            <div className="grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
            <Card variant="bordered" className="h-32" />
            <Card variant="bordered" className="h-48" />
          </div>
        </div>
      </div>
    );
  }

  if (cartQuery.isError || !cart) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-[1100px] px-4 sm:px-6 py-12">
          <Card variant="bordered" className="space-y-4 text-center">
            <h1 className="text-2xl font-semibold">Unable to load your cart</h1>
            <p className="text-sm text-foreground/70">
              Please refresh or try again in a moment.
            </p>
            <Button onClick={() => cartQuery.refetch()}>Retry</Button>
          </Card>
        </div>
      </div>
    );
  }

  if (cart.items.length === 0) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-[1100px] px-4 sm:px-6 py-12">
          <Card variant="bordered" className="space-y-5 text-center">
            <h1 className="text-2xl font-semibold">Your cart is empty</h1>
            <p className="text-sm text-foreground/70">
              Explore new arrivals and curated collections from Bunoraa artisans.
            </p>
            <Button
              asChild
              variant="primary-gradient"
              className="shadow-soft hover:shadow-soft-lg"
            >
              <Link href="/products/">Continue shopping</Link>
            </Button>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-[1400px] px-4 sm:px-6 py-10">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-foreground/60">
              Bunoraa cart
            </p>
            <h1 className="mt-2 text-2xl font-semibold sm:text-3xl">Shopping cart</h1>
            <p className="text-sm text-foreground/70">
              {itemCount} item{itemCount === 1 ? "" : "s"} in your cart.
            </p>
          </div>
          <div className="grid w-full grid-cols-1 gap-2 sm:flex sm:w-auto sm:flex-wrap sm:items-center">
            <Button
              asChild
              variant="ghost"
              className="w-full border border-border bg-card text-foreground hover:bg-muted sm:w-auto"
            >
              <Link href="/products/">Continue shopping</Link>
            </Button>
            <Button variant="secondary" className="w-full sm:w-auto" onClick={handleClearCart}>
              Clear cart
            </Button>
          </div>
        </div>

        <div className="mt-8 grid gap-8 lg:grid-cols-[1.25fr_0.75fr]">
          <div className="space-y-4">
            {cart.items.map((item) => (
              <CartItemRow
                key={item.id}
                item={item}
                currency={currency}
                onUpdate={handleUpdateItem}
                onRemove={handleRemoveItem}
                onMoveToWishlist={handleMoveToWishlist}
                isUpdating={updatingItemId === item.id}
                isRemoving={removingItemId === item.id}
                resetSignal={resetCounter}
                resetTargetId={resetItemId}
              />
            ))}

            {validationResult ? (
              <Card variant="bordered" className="space-y-3 text-sm">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">Cart validation</h3>
                  <span
                    className={cn(
                      "text-xs font-semibold",
                      validationResult.is_valid ? "text-success-500" : "text-warning-500"
                    )}
                  >
                    {validationResult.is_valid ? "All good" : "Needs attention"}
                  </span>
                </div>
                {validationResult.issues.length > 0 ? (
                  <div>
                    <p className="mb-1 text-xs uppercase tracking-[0.2em] text-foreground/60">
                      Issues
                    </p>
                    <ul className="space-y-1">
                      {validationResult.issues.map((issue, index) => (
                        <li key={`${issue.type}-${index}`} className="text-error-500">
                          {issue.message || "Issue detected."}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {validationResult.warnings.length > 0 ? (
                  <div>
                    <p className="mb-1 text-xs uppercase tracking-[0.2em] text-foreground/60">
                      Warnings
                    </p>
                    <ul className="space-y-1 text-warning-500">
                      {validationResult.warnings.map((warn, index) => (
                        <li key={`${warn.type}-${index}`}>
                          {warn.message || "Warning detected."}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </Card>
            ) : null}
          </div>

          <div className="space-y-4">
            <Card variant="bordered" className="space-y-4">
              <h2 className="text-xl font-semibold">Order summary</h2>
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-foreground/70">Subtotal</span>
                  <span>{formattedSubtotal}</span>
                </div>
                {cart?.coupon_code || summary?.coupon_code ? (
                  <div className="flex items-center justify-between text-success-500">
                    <span>Discount</span>
                    <span>-{formattedDiscount}</span>
                  </div>
                ) : null}
                {showShipping ? (
                  <div className="flex items-center justify-between">
                    <span className="text-foreground/70">
                      {shippingLabel}
                      {shippingEstimateLabel ? ` (${shippingEstimateLabel})` : ""}
                    </span>
                    <span>{formattedShipping}</span>
                  </div>
                ) : null}
                {showTax ? (
                  <div className="flex items-center justify-between">
                    <span className="text-foreground/70">{taxRateLabel}</span>
                    <span>{formattedTax}</span>
                  </div>
                ) : null}
                {showGiftWrap ? (
                  <div className="flex items-center justify-between">
                    <span className="text-foreground/70">
                      {summary?.gift_wrap_label || "Gift wrap"}
                    </span>
                    <span>{formattedGiftWrap}</span>
                  </div>
                ) : null}
                <div className="flex items-center justify-between border-t border-border pt-3 text-base font-semibold">
                  <span>Total</span>
                  <span>{formattedTotal}</span>
                </div>
                {summary ? null : (
                  <p className="text-xs text-foreground/60">
                    Shipping and taxes calculated at checkout.
                  </p>
                )}
              </div>
              <Button asChild variant="primary-gradient" className="w-full">
                <Link href="/checkout/">Proceed to checkout</Link>
              </Button>
              <div className="grid gap-2 text-xs text-foreground/60">
                <p>Secure checkout with encrypted payments.</p>
                  <p>7-day easy returns on eligible items.</p>
              </div>
            </Card>

            <Card variant="bordered" className="space-y-4">
              <h3 className="text-base font-semibold">Promo code</h3>
              <form onSubmit={handleApplyCoupon} className="flex flex-col gap-2">
                <input
                  type="text"
                  value={couponCode}
                  onChange={(event) => setCouponCode(event.target.value)}
                  placeholder="Enter coupon code"
                  className="h-11 rounded-xl border border-border bg-transparent px-3 text-sm"
                />
                <div className="grid gap-2 sm:flex sm:items-center">
                  <Button
                    type="submit"
                    size="sm"
                    variant="secondary"
                    className="w-full sm:w-auto"
                  >
                    Apply
                  </Button>
                  {cart.coupon_code ? (
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      className="w-full sm:w-auto"
                      onClick={handleRemoveCoupon}
                    >
                      Remove
                    </Button>
                  ) : null}
                </div>
              </form>
              {cart.coupon_code ? (
                <p className="text-xs text-foreground/60">
                  Applied coupon: {cart.coupon_code}
                </p>
              ) : null}
            </Card>

            <Card variant="bordered" className="space-y-3">
              <h3 className="text-base font-semibold">Gift options</h3>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-border bg-card text-primary accent-primary focus:ring-2 focus:ring-primary/30"
                  checked={giftState.is_gift}
                  onChange={(event) =>
                    setGiftState((prev) => ({
                      ...prev,
                      is_gift: event.target.checked,
                    }))
                  }
                />
                This order is a gift
              </label>
              {giftState.is_gift ? (
                <div className="space-y-2">
                  <textarea
                    value={giftState.gift_message}
                    onChange={(event) =>
                      setGiftState((prev) => ({
                        ...prev,
                        gift_message: event.target.value,
                      }))
                    }
                    placeholder="Add a gift message (optional)"
                    className="min-h-[90px] w-full rounded-xl border border-border bg-transparent px-3 py-2 text-sm"
                  />
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      className="h-4 w-4 rounded border-border bg-card text-primary accent-primary focus:ring-2 focus:ring-primary/30"
                      checked={giftState.gift_wrap}
                      onChange={(event) =>
                        setGiftState((prev) => ({
                          ...prev,
                          gift_wrap: event.target.checked,
                        }))
                      }
                    />
                    {giftResponse?.gift_wrap_label || "Add gift wrap"}
                  </label>
                </div>
              ) : null}
              <Button size="sm" variant="secondary" className="w-full sm:w-auto" onClick={handleGiftSave}>
                Save gift options
              </Button>
              {giftResponse?.formatted_gift_wrap_cost ? (
                <p className="text-xs text-foreground/60">
                  Gift wrap cost: {giftResponse.formatted_gift_wrap_cost}
                </p>
              ) : null}
            </Card>

            <Card variant="bordered" className="space-y-3">
              <h3 className="text-base font-semibold">Share cart</h3>
              <div className="grid gap-2">
                <input
                  type="text"
                  placeholder="Name (optional)"
                  value={shareState.name}
                  onChange={(event) =>
                    setShareState((prev) => ({ ...prev, name: event.target.value }))
                  }
                  className="h-10 rounded-xl border border-border bg-transparent px-3 text-sm"
                />
                <div className="grid gap-2 sm:grid-cols-2">
                  <select
                    value={shareState.permission}
                    onChange={(event) =>
                      setShareState((prev) => ({
                        ...prev,
                        permission: event.target.value,
                      }))
                    }
                    className="h-10 rounded-xl border border-border bg-card px-3 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                  >
                    <option value="view">View only</option>
                    <option value="edit">Can edit items</option>
                    <option value="purchase">Can purchase</option>
                  </select>
                  <div className="relative">
                    <input
                      type="number"
                      min={1}
                      max={365}
                      value={shareState.expires_days}
                      onChange={(event) =>
                        setShareState((prev) => ({
                          ...prev,
                          expires_days: Number(event.target.value || 7),
                        }))
                      }
                      className="no-spin h-10 w-full rounded-xl border border-border bg-card px-3 pr-10 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                      placeholder="Expires in days"
                      inputMode="numeric"
                      pattern="[0-9]*"
                    />
                    <div className="absolute right-1 top-1 flex h-8 w-8 flex-col overflow-hidden rounded-lg border border-border bg-muted/40">
                      <button
                        type="button"
                        className="flex h-4 items-center justify-center text-[10px] text-foreground/70 hover:bg-muted"
                        aria-label="Increase days"
                        onClick={() =>
                          setShareState((prev) => ({
                            ...prev,
                            expires_days: Math.min(365, (prev.expires_days || 7) + 1),
                          }))
                        }
                      >
                        ^
                      </button>
                      <button
                        type="button"
                        className="flex h-4 items-center justify-center text-[10px] text-foreground/70 hover:bg-muted"
                        aria-label="Decrease days"
                        onClick={() =>
                          setShareState((prev) => ({
                            ...prev,
                            expires_days: Math.max(1, (prev.expires_days || 7) - 1),
                          }))
                        }
                      >
                        v
                      </button>
                    </div>
                  </div>
                </div>
                <input
                  type="password"
                  placeholder="Password (optional)"
                  value={shareState.password}
                  onChange={(event) =>
                    setShareState((prev) => ({
                      ...prev,
                      password: event.target.value,
                    }))
                  }
                  className="h-10 rounded-xl border border-border bg-transparent px-3 text-sm"
                />
                <div className="grid gap-2 sm:flex sm:flex-wrap sm:items-center">
                  <Button
                    size="sm"
                    variant="secondary"
                    className="w-full sm:w-auto"
                    onClick={handleShareCart}
                  >
                    Create link
                  </Button>
                  {shareResult?.share_url ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="w-full sm:w-auto"
                      onClick={handleCopyShare}
                    >
                      Copy link
                    </Button>
                  ) : null}
                </div>
                {shareResult?.share_url ? (
                  <p className="break-all text-xs text-foreground/60">
                    {shareResult.share_url}
                  </p>
                ) : null}
              </div>
            </Card>

            <Card variant="bordered" className="space-y-2">
              <h3 className="text-base font-semibold">Cart health</h3>
              <div className="grid gap-2 sm:flex sm:flex-wrap sm:items-center">
                <Button
                  size="sm"
                  variant="secondary"
                  className="w-full sm:w-auto"
                  onClick={handleValidateCart}
                >
                  Validate cart
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={handleLockPrices}
                  className="w-full border border-border bg-card text-foreground hover:bg-muted sm:w-auto"
                >
                  Lock prices for 24h
                </Button>
              </div>
            </Card>
          </div>
        </div>

        {relatedQuery.data && relatedQuery.data.length > 0 ? (
          <div className="mt-12">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-2xl font-semibold">Recommended for you</h2>
              <Button asChild variant="ghost" className="w-full sm:w-auto">
                <Link href="/products/">View all</Link>
              </Button>
            </div>
            <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
              {relatedQuery.data.map((product) => (
                <Card key={product.id} variant="bordered" className="flex flex-col gap-3">
                  <div className="aspect-[4/5] overflow-hidden rounded-xl bg-muted">
                    {getProductImage(product) ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={getProductImage(product) || ""}
                        alt={product.name}
                        className="h-full w-full object-cover"
                      />
                    ) : null}
                  </div>
                  <div className="flex flex-1 flex-col gap-1">
                    <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                      {product.primary_category_name || "Recommended"}
                    </p>
                    <p className="font-medium">{product.name}</p>
                    <p className="text-sm text-foreground/70">
                      {formatMoney(getProductPrice(product), product.currency)}
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      size="sm"
                      variant="secondary"
                      className="w-full"
                      onClick={() => handleAddRecommended(product)}
                    >
                      Add to cart
                    </Button>
                    <Button asChild size="sm" variant="ghost" className="w-full">
                      <Link href={`/products/${product.slug}/`}>View</Link>
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
