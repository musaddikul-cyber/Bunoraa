import Link from "next/link";
import { apiFetch, ApiError } from "@/lib/api";
import type { Collection, ProductListItem } from "@/lib/types";
import { ProductGrid } from "@/components/products/ProductGrid";
import { Button } from "@/components/ui/Button";
import { notFound } from "next/navigation";

export const revalidate = 600;

async function getCollection(slug: string) {
  try {
    const response = await apiFetch<Collection>(`/catalog/collections/${slug}/`, {
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

async function getCollectionProducts(slug: string) {
  const response = await apiFetch<ProductListItem[]>(
    `/catalog/collections/${slug}/products/`,
    { next: { revalidate } }
  );
  return response.data;
}

export default async function CollectionDetailPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const [collection, products] = await Promise.all([
    getCollection(slug),
    getCollectionProducts(slug),
  ]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-7xl px-6 py-12">
        <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Collection
            </p>
            <h1 className="text-3xl font-semibold">{collection.name}</h1>
            <p className="mt-2 text-foreground/70">{collection.description}</p>
          </div>
          <Button asChild variant="secondary">
            <Link href="/products/">Shop all products</Link>
          </Button>
        </div>

        <ProductGrid products={products} />
      </div>
    </div>
  );
}