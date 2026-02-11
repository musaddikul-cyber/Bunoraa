import type { ProductListItem } from "@/lib/types";

export type RecentlyViewedItem = Pick<
  ProductListItem,
  "id" | "slug" | "name" | "primary_image" | "current_price" | "currency" | "average_rating"
> & {
  viewed_at: string;
};

const KEY = "bunoraa-recently-viewed";
const MAX_ITEMS = 12;

function normalizeSlug(slug: string): string {
  return slug.trim().toLowerCase();
}

function normalizeItems(items: RecentlyViewedItem[]): RecentlyViewedItem[] {
  const seen = new Set<string>();
  const normalized: RecentlyViewedItem[] = [];

  for (const item of items) {
    if (!item || typeof item.id !== "string" || typeof item.slug !== "string") continue;
    const slug = item.slug.trim();
    if (!slug) continue;
    const key = normalizeSlug(slug);
    if (seen.has(key)) continue;
    seen.add(key);
    normalized.push({ ...item, slug });
  }

  return normalized;
}

function safeParse(value: string | null): RecentlyViewedItem[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as RecentlyViewedItem[];
    if (!Array.isArray(parsed)) return [];
    return normalizeItems(parsed);
  } catch {
    return [];
  }
}

function notify() {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent("recently-viewed-updated"));
}

export function getRecentlyViewed(): RecentlyViewedItem[] {
  if (typeof window === "undefined") return [];
  return safeParse(window.localStorage.getItem(KEY));
}

export function addRecentlyViewed(item: Omit<RecentlyViewedItem, "viewed_at">) {
  if (typeof window === "undefined") return;
  if (typeof item.slug !== "string" || !item.slug.trim()) return;
  const slugKey = normalizeSlug(item.slug);
  const items = getRecentlyViewed().filter(
    (existing) => existing.id !== item.id && normalizeSlug(existing.slug) !== slugKey
  );
  const next: RecentlyViewedItem[] = [
    { ...item, slug: item.slug.trim(), viewed_at: new Date().toISOString() },
    ...items,
  ];
  setRecentlyViewed(next);
}

export function setRecentlyViewed(items: RecentlyViewedItem[]) {
  if (typeof window === "undefined") return;
  const normalized = normalizeItems(items).slice(0, MAX_ITEMS);
  window.localStorage.setItem(KEY, JSON.stringify(normalized));
  notify();
}

export function clearRecentlyViewed() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(KEY);
  notify();
}
