"use client";

import Link from "next/link";
import { AuthGate } from "@/components/auth/AuthGate";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useWishlist } from "@/components/wishlist/useWishlist";

export default function WishlistPage() {
  const { wishlistQuery, removeItem, moveToCart } = useWishlist();

  return (
    <AuthGate title="Wishlist" description="Sign in to view your wishlist.">
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-5xl px-4 sm:px-6 py-16">
          <div className="mb-8">
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Wishlist
            </p>
            <h1 className="text-2xl font-semibold">Saved items</h1>
          </div>

          {wishlistQuery.isLoading ? (
            <div className="text-sm text-foreground/60">Loading wishlist...</div>
          ) : wishlistQuery.isError ? (
            <div className="text-sm text-foreground/60">
              Could not load wishlist.
            </div>
          ) : (
            <div className="grid gap-6 md:grid-cols-2">
              {wishlistQuery.data?.data?.length ? (
                wishlistQuery.data.data.map((item) => (
                  <Card key={item.id} variant="bordered" className="flex gap-4">
                    <div className="h-24 w-24 overflow-hidden rounded-xl bg-muted">
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
                      <Link href={`/products/${item.product_slug}/`}>
                        <h2 className="text-lg font-semibold">
                          {item.product_name}
                        </h2>
                      </Link>
                      <p className="text-sm text-foreground/60">
                        {item.current_price || item.price_at_add}
                      </p>
                      <div className="flex flex-wrap gap-2">
                        <Button
                          variant="primary-gradient"
                          size="sm"
                          onClick={() => moveToCart.mutate(item.id)}
                        >
                          Move to cart
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeItem.mutate(item.id)}
                        >
                          Remove
                        </Button>
                      </div>
                    </div>
                  </Card>
                ))
              ) : (
                <Card variant="bordered">
                  <p className="text-sm text-foreground/60">
                    Your wishlist is empty.
                  </p>
                </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </AuthGate>
  );
}
