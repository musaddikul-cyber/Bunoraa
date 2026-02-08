export type ProductFilterState = {
  q?: string;
  ordering?: string;
  view?: "grid" | "list";
  priceMin?: string;
  priceMax?: string;
  inStock?: boolean;
  onSale?: boolean;
  minRating?: string;
  newArrivals?: boolean;
  tags: string[];
  attrs: Record<string, string[]>;
};

export type AppliedFilter = {
  key: string;
  label: string;
  value?: string;
};

function parseBool(value: string | null) {
  return value === "1" || value === "true";
}

function splitValues(value: string | null) {
  if (!value) return [];
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function parseFilters(searchParams: URLSearchParams): ProductFilterState {
  const attrs: Record<string, string[]> = {};
  searchParams.forEach((value, key) => {
    if (!key.startsWith("attr_")) return;
    const slug = key.replace("attr_", "");
    const values = splitValues(value);
    if (!values.length) return;
    attrs[slug] = Array.from(new Set([...(attrs[slug] || []), ...values]));
  });

  return {
    q: searchParams.get("q") || "",
    ordering: searchParams.get("ordering") || "",
    view: (searchParams.get("view") as "grid" | "list") || "grid",
    priceMin: searchParams.get("price_min") || "",
    priceMax: searchParams.get("price_max") || "",
    inStock: parseBool(searchParams.get("in_stock")),
    onSale: parseBool(searchParams.get("on_sale")),
    minRating: searchParams.get("min_rating") || "",
    newArrivals: parseBool(searchParams.get("new_arrivals")),
    tags: splitValues(searchParams.get("tags")),
    attrs,
  };
}

export function buildSearchParams(filters: ProductFilterState) {
  const params = new URLSearchParams();
  if (filters.q) params.set("q", filters.q);
  if (filters.ordering) params.set("ordering", filters.ordering);
  if (filters.view) params.set("view", filters.view);
  if (filters.priceMin) params.set("price_min", filters.priceMin);
  if (filters.priceMax) params.set("price_max", filters.priceMax);
  if (filters.inStock) params.set("in_stock", "true");
  if (filters.onSale) params.set("on_sale", "true");
  if (filters.minRating) params.set("min_rating", filters.minRating);
  if (filters.newArrivals) params.set("new_arrivals", "true");
  if (filters.tags && filters.tags.length) params.set("tags", filters.tags.join(","));
  Object.entries(filters.attrs || {}).forEach(([slug, values]) => {
    if (!values.length) return;
    params.set(`attr_${slug}`, values.join(","));
  });
  return params;
}

export function getAppliedFilters(filters: ProductFilterState): AppliedFilter[] {
  const applied: AppliedFilter[] = [];
  if (filters.q) applied.push({ key: "q", label: `Search: ${filters.q}` });
  if (filters.priceMin || filters.priceMax) {
    applied.push({
      key: "price",
      label: `Price: ${filters.priceMin || "min"} - ${filters.priceMax || "max"}`,
    });
  }
  if (filters.inStock) applied.push({ key: "in_stock", label: "In stock" });
  if (filters.onSale) applied.push({ key: "on_sale", label: "On sale" });
  if (filters.newArrivals) applied.push({ key: "new_arrivals", label: "New arrivals" });
  if (filters.minRating) applied.push({ key: "min_rating", label: `${filters.minRating}+ stars` });

  filters.tags.forEach((tag) => {
    applied.push({ key: "tags", label: `Tag: ${tag}`, value: tag });
  });

  Object.entries(filters.attrs).forEach(([slug, values]) => {
    values.forEach((value) => {
      applied.push({ key: `attr_${slug}`, label: `${slug}: ${value}`, value });
    });
  });

  return applied;
}

export function updateParamValue(
  searchParams: URLSearchParams,
  key: string,
  value: string | null
) {
  const params = new URLSearchParams(searchParams.toString());
  if (!value) {
    params.delete(key);
  } else {
    params.set(key, value);
  }
  params.delete("page");
  return params;
}

export function toggleMultiValue(
  searchParams: URLSearchParams,
  key: string,
  value: string
) {
  const params = new URLSearchParams(searchParams.toString());
  const current = splitValues(params.get(key));
  const exists = current.includes(value);
  const next = exists
    ? current.filter((item) => item !== value)
    : [...current, value];
  if (next.length) {
    params.set(key, next.join(","));
  } else {
    params.delete(key);
  }
  params.delete("page");
  return params;
}

export function clearAllFilters(searchParams: URLSearchParams) {
  const params = new URLSearchParams(searchParams.toString());
  [
    "q",
    "price_min",
    "price_max",
    "in_stock",
    "on_sale",
    "min_rating",
    "new_arrivals",
    "tags",
  ].forEach((key) => params.delete(key));

  Array.from(params.keys())
    .filter((key) => key.startsWith("attr_"))
    .forEach((key) => params.delete(key));

  params.delete("page");
  return params;
}

export function removeAppliedFilter(
  searchParams: URLSearchParams,
  filter: AppliedFilter
) {
  const params = new URLSearchParams(searchParams.toString());
  if (filter.key === "price") {
    params.delete("price_min");
    params.delete("price_max");
  } else if (filter.key === "tags" && filter.value) {
    const next = splitValues(params.get("tags")).filter((item) => item !== filter.value);
    if (next.length) params.set("tags", next.join(","));
    else params.delete("tags");
  } else if (filter.key.startsWith("attr_") && filter.value) {
    const next = splitValues(params.get(filter.key)).filter(
      (item) => item !== filter.value
    );
    if (next.length) params.set(filter.key, next.join(","));
    else params.delete(filter.key);
  } else {
    params.delete(filter.key);
  }
  params.delete("page");
  return params;
}