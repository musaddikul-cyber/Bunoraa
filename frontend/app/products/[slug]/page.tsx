import type { Metadata } from "next";
import { apiFetch, ApiError } from "@/lib/api";
import type { ProductDetail } from "@/lib/types";
import { notFound, redirect } from "next/navigation";
import { ProductDetailClient } from "@/components/products/ProductDetailClient";
import { getServerLocaleHeaders } from "@/lib/serverLocale";
import { JsonLd } from "@/components/seo/JsonLd";
import { buildBreadcrumbList, buildPageMetadata, buildProductSchema } from "@/lib/seo";
import { buildCategoryPath } from "@/lib/categoryPaths";
import { buildProductCategoryTrail, buildProductPath, getProductCategoryPath } from "@/lib/productPaths";

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

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const product = await getProduct(slug);
  const metadataImages = [
    product.primary_image || undefined,
    ...(product.images?.slice(0, 5).map((image) => image.image) || []),
  ];
  return buildPageMetadata({
    title: product.meta_title || product.name,
    description:
      product.meta_description ||
      product.short_description ||
      product.description ||
      "Explore product details on Bunoraa.",
    path: buildProductPath(product),
    images: metadataImages,
  });
}

export default async function ProductDetailPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const product = await getProduct(slug);

  const canonicalPath = buildProductPath(product);
  const legacyPath = `/products/${product.slug}/`;
  if (canonicalPath !== legacyPath) {
    redirect(canonicalPath);
  }

  const canonicalCategoryPath = getProductCategoryPath(product);
  const categoryTrail = buildProductCategoryTrail(product);
  const breadcrumbItems = [{ name: "Home", url: "/" }];
  if (categoryTrail.length) {
    categoryTrail.forEach((crumb) => {
      breadcrumbItems.push({ name: crumb.name, url: buildCategoryPath(crumb.slugPath) });
    });
  } else if (product.primary_category) {
    breadcrumbItems.push({
      name: product.primary_category.name,
      url: buildCategoryPath(canonicalCategoryPath),
    });
  } else {
    breadcrumbItems.push({ name: "Products", url: "/products/" });
  }
  breadcrumbItems.push({ name: product.name, url: canonicalPath });
  const breadcrumbs = buildBreadcrumbList(breadcrumbItems);
  const productSchema = product.schema_org || buildProductSchema(product);
  const jsonLd = [breadcrumbs, ...(productSchema ? [productSchema] : [])];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-6xl px-3 sm:px-5 py-12">
        <ProductDetailClient product={product} />
      </div>
      <JsonLd data={jsonLd} />
    </div>
  );
}
