"use client";

import { useRouter, usePathname } from "next/navigation";
import { useWishlist } from "@/components/wishlist/useWishlist";
import { Button } from "@/components/ui/Button";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { ApiError } from "@/lib/api";
import { useToast } from "@/components/ui/ToastProvider";

export function AddToWishlistButton({
  productId,
  variantId,
}: {
  productId: string;
  variantId?: string | null;
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
    <Button variant="ghost" onClick={handleClick} disabled={addItem.isPending}>
      {addItem.isPending ? "Saving..." : "Wishlist"}
    </Button>
  );
}
