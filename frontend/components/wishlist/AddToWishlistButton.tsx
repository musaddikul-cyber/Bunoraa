"use client";

import { useRouter, usePathname } from "next/navigation";
import { useWishlist } from "@/components/wishlist/useWishlist";
import { Button, type ButtonProps } from "@/components/ui/Button";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { ApiError } from "@/lib/api";
import { useToast } from "@/components/ui/ToastProvider";
import { cn } from "@/lib/utils";

type AddToWishlistButtonProps = {
  productId: string;
  variantId?: string | null;
  label?: string;
} & Omit<ButtonProps, "onClick">;

export function AddToWishlistButton({
  productId,
  variantId,
  label = "Add to wishlist",
  variant = "secondary",
  size = "sm",
  className,
  ...props
}: AddToWishlistButtonProps) {
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
      <svg
        aria-hidden="true"
        viewBox="0 0 24 24"
        className={cn(
          "h-4 w-4 transition",
          isInWishlist
            ? "fill-error-500 text-error-500"
            : "fill-transparent text-foreground/70 group-hover/wishlist:fill-error-500 group-hover/wishlist:text-error-500"
        )}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M20.8 6.6a5.5 5.5 0 0 0-9.1-3.9L12 3l.3-.3a5.5 5.5 0 0 0-7.7 7.7L12 18.8l7.4-7.4a5.5 5.5 0 0 0 1.4-4.8z" />
      </svg>
      <span>{isBusy ? pendingLabel : buttonLabel}</span>
    </Button>
  );
}
