import type { Metadata } from "next";
import { apiFetch, ApiError } from "@/lib/api";
import type { ProductDetail } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { AddToCartButton } from "@/components/cart/AddToCartButton";
import { AddToWishlistButton } from "@/components/wishlist/AddToWishlistButton";
import { notFound } from "next/navigation";

export const revalidate = 900;

async function getProduct(slug: string) {
  try {
    const response = await apiFetch<ProductDetail>(`/catalog/products/${slug}/`, {
      next: { revalidate },
    });
    return response.data;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const product = await getProduct(slug);
  return {
    title: product.meta_title || product.name,
    description: product.meta_description || product.short_description || "",
  };
}

export default async function ProductDetailPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const product = await getProduct(slug);
  const defaultVariant =
    product.variants?.find((variant) => variant.is_default) ||
    product.variants?.[0] ||
    null;
  const variantId = defaultVariant?.id || null;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-5xl px-6 py-12">
        <div className="grid gap-10 lg:grid-cols-[1.1fr_1fr]">
          <div className="space-y-4">
            <div className="aspect-[4/5] overflow-hidden rounded-2xl bg-muted">
              {product.images?.[0]?.image ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={product.images[0].image}
                  alt={product.name}
                  className="h-full w-full object-cover"
                />
              ) : null}
            </div>
            <div className="grid grid-cols-3 gap-3">
              {product.images?.slice(1, 4).map((image) => (
                <div
                  key={image.id}
                  className="aspect-square overflow-hidden rounded-xl bg-muted"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={image.image}
                    alt={image.alt_text || product.name}
                    className="h-full w-full object-cover"
                  />
                </div>
              ))}
            </div>
          </div>

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
            </div>

            <Card variant="bordered" className="space-y-3">
              <div className="text-2xl font-semibold">
                {product.current_price} {product.currency}
              </div>
              <p className="text-sm text-foreground/60">
                {product.is_in_stock ? "In stock" : "Out of stock"}
              </p>
              <div className="flex flex-wrap gap-3">
                <AddToCartButton
                  productId={product.id}
                  variantId={variantId}
                  variant="primary-gradient"
                />
                <AddToWishlistButton productId={product.id} variantId={variantId} />
              </div>
            </Card>

            <Card variant="default">
              <h2 className="text-lg font-semibold">Details</h2>
              <div className="mt-3 text-sm text-foreground/70">
                {product.description || "Product details will appear here."}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
