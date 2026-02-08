import type { ProductListItem } from "@/lib/types";

export type RecentlyViewedItem = Pick<
  ProductListItem,
  "id" | "slug" | "name" | "primary_image" | "current_price" | "currency" | "average_rating"
> & {
  viewed_at: string;
};

const KEY = "bunoraa-recently-viewed";
const MAX_ITEMS = 12;

function safeParse(value: string | null): RecentlyViewedItem[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as RecentlyViewedItem[];
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item) => item && typeof item.id === "string");
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
  const items = getRecentlyViewed().filter((existing) => existing.id !== item.id);
  const next: RecentlyViewedItem[] = [
    { ...item, viewed_at: new Date().toISOString() },
    ...items,
  ].slice(0, MAX_ITEMS);
  window.localStorage.setItem(KEY, JSON.stringify(next));
  notify();
}

export function clearRecentlyViewed() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(KEY);
  notify();
}