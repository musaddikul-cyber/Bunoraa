"use client";

import { usePathname, useRouter } from "next/navigation";
import { useWishlist } from "@/components/wishlist/useWishlist";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { useToast } from "@/components/ui/ToastProvider";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";

export function WishlistIconButton({
  productId,
  variantId,
  className,
  variant = "default",
  size = "md",
  color = "default",
}: {
  productId: string;
  variantId?: string | null;
  className?: string;
  variant?: "default" | "ghost";
  size?: "sm" | "md" | "lg";
  color?: "default" | "fixed-black";
}) {
  const router = useRouter();
  const pathname = usePathname();
  const { hasToken } = useAuthContext();
  const { wishlistQuery, addItem, removeItem } = useWishlist({ enabled: hasToken });
  const { push } = useToast();

  const wishlistItems = wishlistQuery.data?.data ?? [];
  const existingItem = wishlistItems.find((item) => item.product_id === productId);
  const isInWishlist = Boolean(existingItem);
  const isBusy = addItem.isPending || removeItem.isPending;

  const resolveMessage = (response: unknown, fallback: string) => {
    if (response && typeof response === "object" && "message" in response) {
      const message = String((response as { message?: string }).message || "").trim();
      if (message && message.toLowerCase() !== "ok") return message;
    }
    return fallback;
  };

  const handleClick = async () => {
    if (!hasToken) {
      router.push(`/account/login/?next=${encodeURIComponent(pathname || "/")}`);
      return;
    }
    try {
      if (isInWishlist && existingItem) {
        const response = await removeItem.mutateAsync(existingItem.id);
        push(resolveMessage(response, "Removed from wishlist."), "success");
      } else {
        const response = await addItem.mutateAsync({ productId, variantId });
        push(resolveMessage(response, "Added to wishlist."), "success");
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        router.push(`/account/login/?next=${encodeURIComponent(pathname || "/")}`);
        return;
      }
      push("Could not update wishlist.", "error");
    }
  };

  const sizeClasses = {
    sm: "h-8 w-8",
    md: "h-9 w-9",
    lg: "h-10 w-10",
  };

  const iconClasses = {
    sm: "h-4 w-4",
    md: "h-5 w-5",
    lg: "h-6 w-6",
  };

  const iconTone =
    color === "fixed-black"
      ? isInWishlist
        ? "fill-error-500 text-error-500"
        : "fill-transparent text-black group-hover/heart:fill-error-500 group-hover/heart:text-error-500"
      : isInWishlist
      ? "fill-error-500 text-error-500"
      : "fill-transparent text-foreground/70 group-hover/heart:fill-error-500 group-hover/heart:text-error-500";

  return (
    <button
      type="button"
      onClick={handleClick}
      className={cn(
        "group/heart inline-flex items-center justify-center rounded-full text-foreground transition",
        sizeClasses[size],
        variant === "default"
          ? "bg-background/80 shadow-sm backdrop-blur hover:bg-background"
          : "bg-transparent hover:text-primary",
        className
      )}
      aria-pressed={isInWishlist}
      aria-label={isInWishlist ? "Remove from wishlist" : "Add to wishlist"}
      disabled={isBusy}
    >
      <svg
        aria-hidden="true"
        viewBox="0 0 24 24"
        className={cn(
          iconClasses[size],
          "transition",
          iconTone
        )}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M20.8 6.6a5.5 5.5 0 0 0-9.1-3.9L12 3l.3-.3a5.5 5.5 0 0 0-7.7 7.7L12 18.8l7.4-7.4a5.5 5.5 0 0 0 1.4-4.8z" />
      </svg>
    </button>
  );
}
