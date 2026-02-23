function cleanSegment(value: string | null | undefined): string {
  return String(value || "")
    .trim()
    .replace(/^\/+|\/+$/g, "");
}

export function normalizeCategorySlugPath(slugPath: string | null | undefined): string {
  return String(slugPath || "")
    .split("/")
    .map((segment) => cleanSegment(segment))
    .filter(Boolean)
    .join("/");
}

export function buildCategoryPath(slugPath: string | null | undefined): string {
  const normalized = normalizeCategorySlugPath(slugPath);
  if (!normalized) return "/categories/";
  return `/${normalized}/`;
}
