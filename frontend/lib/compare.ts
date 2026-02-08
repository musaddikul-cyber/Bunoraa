import type { ProductListItem } from "@/lib/types";

export type CompareItem = Pick<
  ProductListItem,
  "id" | "slug" | "name" | "primary_image" | "current_price" | "currency" | "average_rating" | "reviews_count" | "is_in_stock" | "primary_category_name"
>;

const KEY = "bunoraa-compare";
const MAX_ITEMS = 4;

function safeParse(value: string | null): CompareItem[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as CompareItem[];
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item) => item && typeof item.id === "string");
  } catch {
    return [];
  }
}

function notify() {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent("compare-updated"));
}

export function getCompareItems(): CompareItem[] {
  if (typeof window === "undefined") return [];
  return safeParse(window.localStorage.getItem(KEY));
}

export function setCompareItems(items: CompareItem[]) {
  if (typeof window === "undefined") return;
  const next = items.slice(0, MAX_ITEMS);
  window.localStorage.setItem(KEY, JSON.stringify(next));
  notify();
}

export function addCompareItem(item: CompareItem) {
  const items = getCompareItems().filter((existing) => existing.id !== item.id);
  const next = [item, ...items].slice(0, MAX_ITEMS);
  setCompareItems(next);
}

export function removeCompareItem(id: string) {
  const next = getCompareItems().filter((item) => item.id !== id);
  setCompareItems(next);
}

export function toggleCompareItem(item: CompareItem) {
  const items = getCompareItems();
  if (items.some((existing) => existing.id === item.id)) {
    removeCompareItem(item.id);
    return false;
  }
  addCompareItem(item);
  return true;
}

export function isInCompare(id: string) {
  return getCompareItems().some((item) => item.id === id);
}

export function clearCompareItems() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(KEY);
  notify();
}