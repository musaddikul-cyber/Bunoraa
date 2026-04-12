import type { Metadata } from "next";
import { apiFetch, ApiError } from "@/lib/api";
import type { ProductDetail, ProductListItem } from "@/lib/types";
import { notFound, redirect } from "next/navigation";
import { ProductDetailClient } from "@/components/products/ProductDetailClient";
import { getServerLocaleHeaders } from "@/lib/serverLocale";
import { JsonLd } from "@/components/seo/JsonLd";
import { buildBreadcrumbList, buildPageMetadata, buildProductSchema } from "@/lib/seo";
import { buildProductCategoryTrail, buildProductPath, getProductCategoryPath } from "@/lib/productPaths";
import { buildCategoryPath } from "@/lib/categoryPaths";
import {
  buildCategoryMetadataForPath,
  renderCategoryPageForPath,
  type CategorySearchParams,
} from "@/app/categories/[...slug]/categoryPageShared";
import { categoryPathExists } from "@/lib/routeLookup";

export const revalidate = 900;

async function getProduct(slug: string) {
  try {
    const response = await apiFetch<ProductDetail>(`/catalog/products/${slug}/`, {
      headers: await getServerLocaleHeaders(),
      next: { revalidate },
      suppressError: true,
      suppressErrorStatus: [404],
    });
    return response.data;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null;
    }
    throw error;
  }
}

async function getRelated(slug: string) {
  const response = await apiFetch<ProductListItem[]>(
    `/catalog/products/${slug}/related/`,
    { params: { limit: 8 }, headers: await getServerLocaleHeaders(), next: { revalidate } }
  );
  return response.data;
}

function toRequestedCategoryPath(rootCategory: string, rest: string[]) {
  return [rootCategory, ...rest.slice(0, -1)].filter(Boolean).join("/");
}

function toProductSlug(rest: string[]) {
  return rest.at(-1) || "";
}

export async function generateMetadata({
  params,
  searchParams,
}: {
  params: Promise<{ slug: string; rest: string[] }>;
  searchParams: Promise<CategorySearchParams>;
}): Promise<Metadata> {
  const [{ slug: rootCategory, rest }, resolvedSearchParams] = await Promise.all([
    params,
    searchParams,
  ]);
  const requestedPath = [rootCategory, ...(rest || [])].filter(Boolean).join("/");

  if (requestedPath && (await categoryPathExists(requestedPath))) {
    return buildCategoryMetadataForPath(requestedPath, resolvedSearchParams);
  }

  const productSlug = toProductSlug(rest || []);

  if (!productSlug || !rest?.length) {
    return buildCategoryMetadataForPath(requestedPath || rootCategory, resolvedSearchParams);
  }

  const product = await getProduct(productSlug);
  if (!product) {
    return buildCategoryMetadataForPath(requestedPath, resolvedSearchParams);
  }

  const canonicalCategoryPath = getProductCategoryPath(product);
  const requestedCategoryPath = toRequestedCategoryPath(rootCategory, rest);
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
    path:
      canonicalCategoryPath === requestedCategoryPath
        ? `/${requestedCategoryPath}/${product.slug}/`
        : buildProductPath(product),
    images: metadataImages,
  });
}

export default async function NestedCategoryProductDetailPage({
  params,
  searchParams,
}: {
  params: Promise<{ slug: string; rest: string[] }>;
  searchParams: Promise<CategorySearchParams>;
}) {
  const [{ slug: rootCategory, rest }, resolvedSearchParams] = await Promise.all([
    params,
    searchParams,
  ]);
  const requestedCategoryListingPath = [rootCategory, ...(rest || [])].join("/");
  if (requestedCategoryListingPath && (await categoryPathExists(requestedCategoryListingPath))) {
    return renderCategoryPageForPath(requestedCategoryListingPath, resolvedSearchParams);
  }

  if (!rest?.length) {
    notFound();
  }

  const productSlug = toProductSlug(rest);
  if (!productSlug) {
    notFound();
  }

  const product = await getProduct(productSlug);
  if (!product) {
    notFound();
  }
  const relatedProducts = await getRelated(productSlug).catch(() => []);

  const requestedCategoryPath = toRequestedCategoryPath(rootCategory, rest);
  const canonicalCategoryPath = getProductCategoryPath(product);
  const canonicalPath = buildProductPath(product);

  if (requestedCategoryPath !== canonicalCategoryPath) {
    redirect(canonicalPath);
  }

  const categoryTrail = buildProductCategoryTrail(product);
  const breadcrumbItems = [{ name: "Home", url: "/" }];
  categoryTrail.forEach((crumb) => {
    breadcrumbItems.push({ name: crumb.name, url: buildCategoryPath(crumb.slugPath) });
  });
  breadcrumbItems.push({ name: product.name, url: canonicalPath });

  const breadcrumbs = buildBreadcrumbList(breadcrumbItems);
  const productSchema = product.schema_org || buildProductSchema(product);
  const jsonLd = [breadcrumbs, ...(productSchema ? [productSchema] : [])];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-6xl px-3 sm:px-5 py-12">
        <ProductDetailClient product={product} relatedProducts={relatedProducts} />
      </div>
      <JsonLd data={jsonLd} />
    </div>
  );
}
