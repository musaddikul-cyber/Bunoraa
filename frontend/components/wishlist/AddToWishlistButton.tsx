"use client";

import { useWishlist } from "@/components/wishlist/useWishlist";
import { Button, type ButtonProps } from "@/components/ui/Button";
import { useToast } from "@/components/ui/ToastProvider";
import { Heart } from "lucide-react";
import { cn } from "@/lib/utils";

type AddToWishlistButtonProps = {
  productId: string;
  variantId?: string | null;
  label?: string;
  hideIconOnMobile?: boolean;
} & Omit<ButtonProps, "onClick">;

export function AddToWishlistButton({
  productId,
  variantId,
  label = "Add to wishlist",
  hideIconOnMobile = false,
  variant = "secondary",
  size = "sm",
  className,
  ...props
}: AddToWishlistButtonProps) {
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

  const buttonLabel = isInWishlist ? "Wishlisted" : label;
  const pendingLabel = isInWishlist ? "Removing..." : "Saving...";

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleClick}
      disabled={isBusy}
      aria-pressed={isInWishlist}
      aria-label={isInWishlist ? "Remove from wishlist" : "Add to wishlist"}
      className={cn("group/wishlist", className)}
      {...props}
    >
      <Heart
        aria-hidden="true"
        className={cn(
          "h-5 w-5 transition",
          hideIconOnMobile && "hidden sm:block",
          isInWishlist
            ? "fill-error-500 text-error-500"
            : "fill-transparent text-foreground/70 group-hover/wishlist:fill-error-500 group-hover/wishlist:text-error-500"
        )}
        strokeWidth={1.8}
      />
      <span>{isBusy ? pendingLabel : buttonLabel}</span>
    </Button>
  );
}
