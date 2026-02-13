import type { Metadata } from "next";
import { apiFetch, ApiError } from "@/lib/api";
import type { ProductDetail, ProductListItem } from "@/lib/types";
import { notFound } from "next/navigation";
import { ProductDetailClient } from "@/components/products/ProductDetailClient";
import { getServerLocaleHeaders } from "@/lib/serverLocale";
import { JsonLd } from "@/components/seo/JsonLd";
import { buildBreadcrumbList, buildProductSchema } from "@/lib/seo";

export const revalidate = 900;

async function getProduct(slug: string) {
  try {
    const response = await apiFetch<ProductDetail>(`/catalog/products/${slug}/`, {
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

async function getRelated(slug: string) {
  const response = await apiFetch<ProductListItem[]>(
    `/catalog/products/${slug}/related/`,
    { params: { limit: 4 }, headers: await getServerLocaleHeaders(), next: { revalidate } }
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

  const breadcrumbItems = [
    { name: "Home", url: "/" },
    { name: "Products", url: "/products/" },
  ];
  if (product.breadcrumbs && product.breadcrumbs.length) {
    product.breadcrumbs.forEach((crumb) => {
      breadcrumbItems.push({ name: crumb.name, url: `/categories/${crumb.slug}/` });
    });
  } else if (product.primary_category) {
    breadcrumbItems.push({
      name: product.primary_category.name,
      url: `/categories/${product.primary_category.slug}/`,
    });
  }
  breadcrumbItems.push({ name: product.name, url: `/products/${product.slug}/` });
  const breadcrumbs = buildBreadcrumbList(breadcrumbItems);
  const productSchema = product.schema_org || buildProductSchema(product);
  const jsonLd = [breadcrumbs, ...(productSchema ? [productSchema] : [])];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-6xl px-6 py-12">
        <ProductDetailClient product={product} relatedProducts={relatedProducts} />
      </div>
      <JsonLd data={jsonLd} />
    </div>
  );
}
