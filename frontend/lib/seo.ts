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
  listName?: string
) {
  return cleanObject({
    "@context": "https://schema.org",
    "@type": "ItemList",
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

