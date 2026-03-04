"use client";

import { useWishlist } from "@/components/wishlist/useWishlist";
import { useToast } from "@/components/ui/ToastProvider";
import { Heart } from "lucide-react";
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
  const { wishlistQuery, addItem, removeItem } = useWishlist();
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
    try {
      if (isInWishlist && existingItem) {
        const response = await removeItem.mutateAsync(existingItem.id);
        push(resolveMessage(response, "Removed from wishlist."), "success");
      } else {
        const response = await addItem.mutateAsync({ productId, variantId });
        push(resolveMessage(response, "Added to wishlist."), "success");
      }
    } catch {
      push("Could not update wishlist.", "error");
    }
  };

  const sizeClasses = {
    sm: "h-9 w-9",
    md: "h-10 w-10",
    lg: "h-11 w-11",
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
      <Heart
        aria-hidden="true"
        className={cn(iconClasses[size], "transition", iconTone)}
        strokeWidth={1.8}
      />
    </button>
  );
}
