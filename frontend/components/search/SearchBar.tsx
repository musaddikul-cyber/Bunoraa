"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { ProductListItem } from "@/lib/types";

function useDebouncedValue<T>(value: T, delay = 300) {
  const [debounced, setDebounced] = React.useState(value);
  React.useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

type SearchResponse = {
  products: ProductListItem[];
  categories: Array<{ id: string; name: string; slug: string }>;
  query: string;
};

export function SearchBar() {
  const router = useRouter();
  const [query, setQuery] = React.useState("");
  const debounced = useDebouncedValue(query, 400);

  const suggestions = useQuery({
    queryKey: ["search", "suggestions", debounced],
    queryFn: async () => {
      const response = await apiFetch<SearchResponse>("/catalog/search/", {
        params: { q: debounced },
      });
      return response.data;
    },
    enabled: debounced.trim().length >= 2,
  });

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    router.push(`/search/?q=${encodeURIComponent(trimmed)}`);
    setQuery("");
  };
  const handleSelection = (href: string) => {
    setQuery("");
    router.push(href);
  };

  return (
    <div className="relative w-full max-w-md">
      <form onSubmit={onSubmit}>
        <input
          className="h-9 w-full min-h-0 rounded-full border border-border bg-card px-3 py-1 text-sm"
          placeholder="Search products"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </form>

      {query.trim().length >= 2 &&
      suggestions.data &&
      (suggestions.data.products.length > 0 || suggestions.data.categories.length > 0) ? (
        <div className="absolute left-0 right-0 top-full z-50 mt-2 rounded-xl border border-border bg-card p-3 shadow-lg">
          {suggestions.data.products.length > 0 ? (
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">Products</p>
              <ul className="mt-2 space-y-1">
                {suggestions.data.products.slice(0, 5).map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      className="w-full text-left text-sm text-foreground/80 hover:text-foreground"
                      onClick={() => handleSelection(`/products/${item.slug}/`)}
                    >
                      {item.name}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {suggestions.data.categories.length > 0 ? (
            <div className="mt-3">
              <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">Categories</p>
              <ul className="mt-2 space-y-1">
                {suggestions.data.categories.slice(0, 5).map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      className="w-full text-left text-sm text-foreground/80 hover:text-foreground"
                      onClick={() => handleSelection(`/categories/${item.slug}/`)}
                    >
                      {item.name}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
