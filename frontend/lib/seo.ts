import type { ProductDetail } from "@/lib/types";
import type { Metadata } from "next";
import { buildProductPath } from "@/lib/productPaths";

type UrlLike = string | null | undefined;

export const SITE_URL = (process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000").replace(
  /\/$/,
  ""
);
export const SITE_NAME = "Bunoraa";
export const DEFAULT_OG_IMAGE_PATH = "/opengraph-image";

export function absoluteUrl(path: UrlLike): string {
  if (!path) return SITE_URL;
  if (path.startsWith("//")) return `https:${path}`;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  if (!path.startsWith("/")) return `${SITE_URL}/${path}`;
  return `${SITE_URL}${path}`;
}

function normalizePath(path: string): string {
  if (!path) return "/";
  if (path.startsWith("http://") || path.startsWith("https://")) {
    try {
      const url = new URL(path);
      return `${url.pathname}${url.search}` || "/";
    } catch {
      return "/";
    }
  }
  return path.startsWith("/") ? path : `/${path}`;
}

export function buildPageMetadata({
  title,
  description,
  path,
  images,
  type = "website",
}: {
  title: string;
  description: string;
  path: string;
  images?: Array<string | null | undefined>;
  type?: "website" | "article";
}): Metadata {
  const canonicalPath = normalizePath(path);
  const canonicalUrl = absoluteUrl(canonicalPath);
  const imageUrls = (images || []).filter(Boolean).map((image) => absoluteUrl(image as string));
  const shareImages = imageUrls.length ? imageUrls : [absoluteUrl(DEFAULT_OG_IMAGE_PATH)];

  return {
    title,
    description,
    alternates: {
      canonical: canonicalPath,
    },
    openGraph: {
      type,
      url: canonicalUrl,
      siteName: SITE_NAME,
      title,
      description,
      images: shareImages,
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: shareImages,
    },
  };
}

export function buildNoIndexMetadata({
  title,
  description,
  path,
}: {
  title: string;
  description: string;
  path: string;
}): Metadata {
  return {
    ...buildPageMetadata({ title, description, path }),
    robots: {
      index: false,
      follow: false,
      googleBot: {
        index: false,
        follow: false,
        noimageindex: true,
        "max-snippet": 0,
        "max-image-preview": "none",
        "max-video-preview": 0,
      },
    },
  };
}

export function cleanObject<T extends Record<string, unknown>>(obj: T): T {
  const entries = Object.entries(obj).filter(([, value]) => {
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
  const url = absoluteUrl(buildProductPath(product));
  const images = [
    product.primary_image || undefined,
    ...(product.images?.map((image) => image.image) || []),
  ].filter(Boolean) as string[];

  const parsePrice = (value: string | number | null | undefined) => {
    if (value === null || value === undefined || value === "") return null;
    const parsed =
      typeof value === "number" ? value : Number(String(value).replace(/,/g, ""));
    return Number.isFinite(parsed) ? parsed : null;
  };

  const fallbackPrice = product.current_price || product.sale_price || product.price || undefined;

  const variantOffers = (product.variants || [])
    .map((variant) => {
      const variantPrice = variant.current_price || variant.price || fallbackPrice || undefined;
      if (!variantPrice || !product.currency) return null;
      const optionLabel =
        variant.option_values?.map((item) => `${item.option.name}: ${item.value}`).join(" / ") ||
        undefined;
      return cleanObject({
        "@type": "Offer",
        sku: variant.sku || undefined,
        name: optionLabel ? `${product.name} - ${optionLabel}` : undefined,
        price: variantPrice,
        priceCurrency: product.currency,
        availability:
          typeof variant.stock_quantity === "number"
            ? variant.stock_quantity > 0
              ? "https://schema.org/InStock"
              : "https://schema.org/OutOfStock"
            : product.is_in_stock
            ? "https://schema.org/InStock"
            : "https://schema.org/OutOfStock",
        itemCondition: "https://schema.org/NewCondition",
        seller: {
          "@type": "Organization",
          name: SITE_NAME,
        },
        url,
      });
    })
    .filter(Boolean);

  const variantPrices = variantOffers
    .map((offer) => parsePrice((offer as { price?: string | number }).price))
    .filter((value): value is number => typeof value === "number");

  const offerFallback =
    fallbackPrice && product.currency
      ? cleanObject({
          "@type": "Offer",
          price: fallbackPrice,
          priceCurrency: product.currency,
          availability: product.is_in_stock
            ? "https://schema.org/InStock"
            : "https://schema.org/OutOfStock",
          itemCondition: "https://schema.org/NewCondition",
          seller: {
            "@type": "Organization",
            name: SITE_NAME,
          },
          url,
        })
      : undefined;

  const offers =
    variantOffers.length > 1 && variantPrices.length
      ? cleanObject({
          "@type": "AggregateOffer",
          priceCurrency: product.currency || undefined,
          lowPrice: Math.min(...variantPrices).toFixed(2),
          highPrice: Math.max(...variantPrices).toFixed(2),
          offerCount: variantOffers.length,
          offers: variantOffers.slice(0, 20),
          availability: product.is_in_stock
            ? "https://schema.org/InStock"
            : "https://schema.org/OutOfStock",
        })
      : variantOffers[0] || offerFallback;

  const attributeBySlug = new Map<string, string>();
  (product.attributes || []).forEach((item) => {
    if (!item.attribute?.slug || !item.value) return;
    attributeBySlug.set(item.attribute.slug.toLowerCase(), item.value);
  });
  const color = attributeBySlug.get("color") || attributeBySlug.get("colour") || undefined;
  const size = attributeBySlug.get("size") || undefined;
  const pattern = attributeBySlug.get("pattern") || undefined;

  const additionalProperty =
    product.attributes?.length
      ? product.attributes.slice(0, 20).map((item) =>
          cleanObject({
            "@type": "PropertyValue",
            name: item.attribute.name,
            value: item.value,
          })
        )
      : undefined;
  const material =
    product.material_breakdown && Object.keys(product.material_breakdown).length
      ? Object.keys(product.material_breakdown).join(", ")
      : undefined;

  const aggregateRating =
    typeof product.average_rating === "number" && product.reviews_count
      ? cleanObject({
          "@type": "AggregateRating",
          ratingValue: product.average_rating,
          reviewCount: product.reviews_count,
        })
      : undefined;

  const hasVariant =
    product.variants?.length
      ? product.variants.slice(0, 20).map((variant) =>
          cleanObject({
            "@type": "ProductModel",
            sku: variant.sku || undefined,
            name: variant.option_values?.length
              ? `${product.name} - ${variant.option_values
                  .map((value) => `${value.option.name}: ${value.value}`)
                  .join(" / ")}`
              : undefined,
            offers: variantOffers.find((offer) => {
              const sku = (offer as { sku?: string }).sku;
              if (sku && variant.sku) return sku === variant.sku;
              return false;
            }),
          })
        )
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
    mpn: product.sku || undefined,
    image: images.length ? images.map((image) => absoluteUrl(image)) : undefined,
    url,
    mainEntityOfPage: url,
    category: product.primary_category?.name,
    brand: {
      "@type": "Brand",
      name: SITE_NAME,
    },
    color,
    size,
    pattern,
    material,
    additionalProperty,
    offers,
    aggregateRating,
    hasVariant,
  });
}
