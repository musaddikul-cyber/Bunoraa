import Link from "next/link";
import { apiFetch, ApiError } from "@/lib/api";
import type { Bundle, ProductListItem } from "@/lib/types";
import { ProductGrid } from "@/components/products/ProductGrid";
import { Button } from "@/components/ui/Button";
import { notFound } from "next/navigation";

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
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-7xl px-6 py-12">
        <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Bundle
            </p>
            <h1 className="text-3xl font-semibold">{bundle.name}</h1>
            <p className="mt-2 text-foreground/70">{bundle.description}</p>
          </div>
          <Button asChild variant="secondary">
            <Link href="/products/">Shop all products</Link>
          </Button>
        </div>

        {products.length ? (
          <ProductGrid products={products} />
        ) : (
          <p className="text-sm text-foreground/60">
            Bundle details are available, but product list is not exposed via API yet.
          </p>
        )}
      </div>
    </div>
  );
}