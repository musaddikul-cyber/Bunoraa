import { apiFetch, ApiError } from "@/lib/api";
import type { WishlistItem } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import Link from "next/link";
import { notFound } from "next/navigation";

export const revalidate = 300;

type SharedWishlistResponse = {
  wishlist: { items: WishlistItem[] };
};

async function getSharedWishlist(token: string) {
  try {
    const response = await apiFetch<SharedWishlistResponse>(
      `/commerce/wishlist/shared/${token}/`,
      { next: { revalidate } }
    );
    return response.data.wishlist.items || [];
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }
}

export default async function SharedWishlistPage({
  params,
}: {
  params: Promise<{ token: string }>;
}) {
  const { token } = await params;
  const items = await getSharedWishlist(token);

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-12">
      <div className="mb-8">
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Shared wishlist
        </p>
        <h1 className="text-3xl font-semibold">Wishlist items</h1>
      </div>
      {items.length === 0 ? (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          This wishlist is empty.
        </Card>
      ) : (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((item) => (
            <Card key={item.id} variant="bordered" className="flex flex-col gap-4">
              <div className="aspect-[4/5] overflow-hidden rounded-xl bg-muted">
                {item.product_image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={item.product_image}
                    alt={item.product_name}
                    className="h-full w-full object-cover"
                  />
                ) : null}
              </div>
              <div className="flex flex-1 flex-col gap-2">
                <h2 className="text-lg font-semibold">{item.product_name}</h2>
                <p className="text-sm text-foreground/70">
                  {item.current_price}
                </p>
              </div>
              <Button asChild size="sm" variant="secondary">
                <Link href={`/products/${item.product_slug}/`}>View product</Link>
              </Button>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
