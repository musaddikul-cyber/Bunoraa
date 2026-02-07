import type { Metadata } from "next";
import { apiFetch, ApiError } from "@/lib/api";
import type { ProductDetail, ProductListItem } from "@/lib/types";
import { notFound } from "next/navigation";
import { ProductDetailClient } from "@/components/products/ProductDetailClient";

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

async function getRelated(slug: string) {
  const response = await apiFetch<ProductListItem[]>(
    `/catalog/products/${slug}/related/`,
    { params: { limit: 4 }, next: { revalidate } }
  );
  return response.data;
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
  const [product, relatedProducts] = await Promise.all([
    getProduct(slug),
    getRelated(slug).catch(() => []),
  ]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-6xl px-6 py-12">
        <ProductDetailClient product={product} relatedProducts={relatedProducts} />
      </div>
      {product.schema_org ? (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(product.schema_org) }}
        />
      ) : null}
    </div>
  );
}