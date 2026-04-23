import type { Metadata } from "next";
import dynamic from "next/dynamic";
import Link from "next/link";
import { apiFetch, ApiError } from "@/lib/api";
import type { Collection, ProductListItem } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { notFound } from "next/navigation";
import { getServerLocaleHeaders } from "@/lib/serverLocale";
import { JsonLd } from "@/components/seo/JsonLd";
import { buildBreadcrumbList, buildCollectionPage, buildItemList, buildPageMetadata } from "@/lib/seo";
import { buildProductPath } from "@/lib/productPaths";

const ProductGrid = dynamic(
  () => import("@/components/products/ProductGrid").then((mod) => mod.ProductGrid)
);

export const revalidate = 600;

async function getCollection(slug: string) {
  try {
    const response = await apiFetch<Collection>(`/catalog/collections/${slug}/`, {
      headers: await getServerLocaleHeaders(),
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
    { headers: await getServerLocaleHeaders(), next: { revalidate } }
  );
  return response.data;
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const collection = await getCollection(slug);
  return buildPageMetadata({
    title: collection.name,
    description:
      collection.description || `Explore curated items in the ${collection.name} collection.`,
    path: `/collections/${collection.slug}/`,
    images: [collection.image],
  });
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
  const collectionUrl = `/collections/${collection.slug}/`;
  const breadcrumbs = buildBreadcrumbList([
    { name: "Home", url: "/" },
    { name: "Collections", url: "/collections/" },
    { name: collection.name, url: collectionUrl },
  ]);
  const itemListId = `${collectionUrl}#itemlist`;
  const productList = buildItemList(
    products.slice(0, 50).map((product) => ({
      name: product.name,
      url: buildProductPath(product),
      image: (product.primary_image as string | undefined) || undefined,
      description: product.short_description || undefined,
    })),
    `${collection.name} products`,
    itemListId
  );
  const collectionPage = buildCollectionPage({
    name: collection.name,
    description: collection.description || undefined,
    url: collectionUrl,
    itemListId,
  });

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-7xl px-3 sm:px-5 py-12">
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
      <JsonLd data={[collectionPage, breadcrumbs, productList]} />
    </div>
  );
}
