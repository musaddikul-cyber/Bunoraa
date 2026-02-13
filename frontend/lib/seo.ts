import type { ProductDetail } from "@/lib/types";

type UrlLike = string | null | undefined;

export const SITE_URL = (process.env.NEXT_PUBLIC_SITE_URL || "https://bunoraa.com").replace(
  /\/$/,
  ""
);

export function absoluteUrl(path: UrlLike): string {
  if (!path) return SITE_URL;
  if (path.startsWith("//")) return `https:${path}`;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  if (!path.startsWith("/")) return `${SITE_URL}/${path}`;
  return `${SITE_URL}${path}`;
}

export function cleanObject<T extends Record<string, unknown>>(obj: T): T {
  const entries = Object.entries(obj).filter(([_, value]) => {
    if (value === null || value === undefined || value === "") return false;
    if (Array.isArray(value)) return value.length > 0;
    return true;
  });
  return Object.fromEntries(entries) as T;
}

export function buildBreadcrumbList(items: Array<{ name: string; url: string }>) {
  return {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: items.map((item, index) =>
      cleanObject({
        "@type": "ListItem",
        position: index + 1,
        name: item.name,
        item: absoluteUrl(item.url),
      })
    ),
  };
}

export function buildItemList(
  items: Array<{
    name: string;
    url: string;
    image?: string | null;
    description?: string | null;
  }>,
  listName?: string,
  listId?: string
) {
  return cleanObject({
    "@context": "https://schema.org",
    "@type": "ItemList",
    "@id": listId ? absoluteUrl(listId) : undefined,
    name: listName,
    itemListElement: items.map((item, index) =>
      cleanObject({
        "@type": "ListItem",
        position: index + 1,
        url: absoluteUrl(item.url),
        name: item.name,
        image: item.image ? absoluteUrl(item.image) : undefined,
        description: item.description,
      })
    ),
  });
}

export function buildCollectionPage({
  name,
  description,
  url,
  itemListId,
}: {
  name: string;
  description?: string | null;
  url: string;
  itemListId?: string;
}) {
  return cleanObject({
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    name,
    description,
    url: absoluteUrl(url),
    mainEntity: itemListId ? { "@id": absoluteUrl(itemListId) } : undefined,
  });
}

export function buildSearchResultsPage({
  name,
  description,
  url,
  itemListId,
}: {
  name: string;
  description?: string | null;
  url: string;
  itemListId?: string;
}) {
  return cleanObject({
    "@context": "https://schema.org",
    "@type": "SearchResultsPage",
    name,
    description,
    url: absoluteUrl(url),
    mainEntity: itemListId ? { "@id": absoluteUrl(itemListId) } : undefined,
  });
}

export function buildProductSchema(product: ProductDetail) {
  const url = absoluteUrl(`/products/${product.slug}/`);
  const images = [
    product.primary_image || undefined,
    ...(product.images?.map((image) => image.image) || []),
  ].filter(Boolean) as string[];

  const price =
    product.current_price ||
    product.sale_price ||
    product.price ||
    undefined;
  const offers =
    price && product.currency
      ? cleanObject({
          "@type": "Offer",
          price,
          priceCurrency: product.currency,
          availability: product.is_in_stock
            ? "https://schema.org/InStock"
            : "https://schema.org/OutOfStock",
          url,
        })
      : undefined;

  const aggregateRating =
    typeof product.average_rating === "number" && product.reviews_count
      ? cleanObject({
          "@type": "AggregateRating",
          ratingValue: product.average_rating,
          reviewCount: product.reviews_count,
        })
      : undefined;

  return cleanObject({
    "@context": "https://schema.org",
    "@type": "Product",
    name: product.meta_title || product.name,
    description:
      product.meta_description ||
      product.short_description ||
      product.description ||
      undefined,
    sku: product.sku || undefined,
    image: images.length ? images.map((image) => absoluteUrl(image)) : undefined,
    url,
    category: product.primary_category?.name,
    offers,
    aggregateRating,
  });
}
