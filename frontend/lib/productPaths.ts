import type { ProductDetail } from "@/lib/types";

type ProductPathSource = Pick<ProductDetail, "slug" | "primary_category" | "breadcrumbs"> & {
  primary_category_slug_path?: string | null;
};

function cleanSegment(value: string | null | undefined): string {
  return String(value || "")
    .trim()
    .replace(/^\/+|\/+$/g, "");
}

export function getProductCategoryPath(product: ProductPathSource): string {
  const precomputedPath = cleanSegment(product.primary_category_slug_path);
  if (precomputedPath) return precomputedPath;

  const breadcrumbPath = (product.breadcrumbs || [])
    .map((crumb) => cleanSegment(crumb.slug))
    .filter(Boolean)
    .join("/");
  if (breadcrumbPath) return breadcrumbPath;

  const primaryCategorySlug = cleanSegment(product.primary_category?.slug);
  if (primaryCategorySlug) return primaryCategorySlug;

  return "products";
}

export function buildProductPath(product: ProductPathSource): string {
  const categoryPath = getProductCategoryPath(product);
  const slug = cleanSegment(product.slug);
  if (!slug) return `/${categoryPath}/`;
  return `/${categoryPath}/${slug}/`;
}

export function buildProductCategoryTrail(
  product: ProductPathSource
): Array<{ name: string; slugPath: string }> {
  if (product.breadcrumbs?.length) {
    const trail: Array<{ name: string; slugPath: string }> = [];
    const parts: string[] = [];

    product.breadcrumbs.forEach((crumb) => {
      const slug = cleanSegment(crumb.slug);
      const name = (crumb.name || "").trim();
      if (!slug || !name) return;
      parts.push(slug);
      trail.push({ name, slugPath: parts.join("/") });
    });

    if (trail.length) return trail;
  }

  const primarySlug = cleanSegment(product.primary_category?.slug);
  const primaryName = (product.primary_category?.name || "").trim();
  if (primarySlug && primaryName) {
    return [{ name: primaryName, slugPath: primarySlug }];
  }

  return [];
}
