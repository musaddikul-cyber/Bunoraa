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
}: {
  productId: string;
  variantId?: string | null;
  className?: string;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const { addItem } = useWishlist();
  const { hasToken } = useAuthContext();
  const { push } = useToast();

  const handleClick = async () => {
    if (!hasToken) {
      router.push(`/account/login/?next=${encodeURIComponent(pathname || "/")}`);
      return;
    }
    try {
      const response = await addItem.mutateAsync({ productId, variantId });
      const message =
        response && typeof response === "object" && "message" in response
          ? String((response as { message?: string }).message || "")
          : "";
      push(message || "Added to wishlist.", "success");
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        router.push(`/account/login/?next=${encodeURIComponent(pathname || "/")}`);
        return;
      }
      push("Could not update wishlist.", "error");
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      className={cn(
        "inline-flex h-9 w-9 items-center justify-center rounded-full bg-background/80 text-foreground shadow-sm backdrop-blur",
        "hover:bg-background",
        className
      )}
      aria-label="Add to wishlist"
    >
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
        <path d="M20.8 6.6a5.5 5.5 0 0 0-9.1-3.9L12 3l.3-.3a5.5 5.5 0 0 0-7.7 7.7L12 18.8l7.4-7.4a5.5 5.5 0 0 0 1.4-4.8z" />
      </svg>
    </button>
  );
}
