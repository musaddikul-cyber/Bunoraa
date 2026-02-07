import { apiFetch, ApiError } from "@/lib/api";
import type { Bundle, ProductListItem } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import Link from "next/link";
import { notFound } from "next/navigation";
import { WishlistIconButton } from "@/components/wishlist/WishlistIconButton";

export const revalidate = 600;

async function getBundle(slug: string) {
  try {
    const response = await apiFetch<Bundle>(`/catalog/bundles/${slug}/`, {
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

async function getBundleProducts(slug: string) {
  const response = await apiFetch<ProductListItem[]>(
    `/catalog/bundles/${slug}/`,
    { next: { revalidate } }
  );
  const data = response.data as unknown as { items?: ProductListItem[] };
  return data.items || [];
}

export default async function BundleDetailPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const [bundle, products] = await Promise.all([
    getBundle(slug),
    getBundleProducts(slug),
  ]);

  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-12">
      <div className="mb-8">
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Bundle
        </p>
        <h1 className="text-3xl font-semibold">{bundle.name}</h1>
        <p className="mt-2 text-foreground/70">{bundle.description}</p>
      </div>

      {products.length > 0 ? (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {products.map((product) => (
            <Card key={product.id} variant="bordered" className="flex flex-col gap-4">
              <div className="relative aspect-[4/5] overflow-hidden rounded-xl bg-muted">
                <WishlistIconButton
                  productId={product.id}
                  className="absolute right-3 top-3"
                />
                {product.primary_image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={product.primary_image}
                    alt={product.name}
                    className="h-full w-full object-cover"
                  />
                ) : null}
              </div>
              <div className="flex flex-1 flex-col gap-2">
                <h2 className="text-lg font-semibold">{product.name}</h2>
                <p className="text-sm text-foreground/70">{product.short_description}</p>
              </div>
              <Button asChild size="sm" variant="secondary">
                <Link href={`/products/${product.slug}/`}>View product</Link>
              </Button>
            </Card>
          ))}
        </div>
      ) : (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          Bundle details are available, but product list is not exposed via API yet.
        </Card>
      )}
    </div>
  );
}
