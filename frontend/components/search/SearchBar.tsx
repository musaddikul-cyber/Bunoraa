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
  const [activeIndex, setActiveIndex] = React.useState(-1);
  const [isInputFocused, setIsInputFocused] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const listboxId = React.useId();
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

  const productSuggestions = suggestions.data?.products ?? [];
  const categorySuggestions = suggestions.data?.categories ?? [];
  const visibleProductCount = productSuggestions.slice(0, 5).length;
  const options = [
    ...productSuggestions.slice(0, 5).map((item) => ({
      id: `product-${item.id}`,
      label: item.name,
      href: `/products/${item.slug}/`,
    })),
    ...categorySuggestions.slice(0, 5).map((item) => ({
      id: `category-${item.id}`,
      label: item.name,
      href: `/categories/${item.slug}/`,
    })),
  ];
  const trimmedQuery = query.trim();
  const hasQuery = trimmedQuery.length >= 2;
  const hasSuggestions = options.length > 0;
  const showSuggestions = isInputFocused && hasQuery;

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (activeIndex >= 0 && options[activeIndex]) {
      handleSelection(options[activeIndex].href);
      return;
    }
    if (!trimmedQuery) return;
    router.push(`/search/?q=${encodeURIComponent(trimmedQuery)}`);
    setQuery("");
    setActiveIndex(-1);
    setIsInputFocused(false);
  };

  const handleSelection = (href: string) => {
    setQuery("");
    setActiveIndex(-1);
    setIsInputFocused(false);
    router.push(href);
  };

  React.useEffect(() => {
    setActiveIndex(-1);
  }, [debounced]);

  React.useEffect(() => {
    if (!showSuggestions) return;
    const handlePointer = (event: MouseEvent | TouchEvent) => {
      const target = event.target as Node | null;
      if (target && containerRef.current?.contains(target)) return;
      setIsInputFocused(false);
      setActiveIndex(-1);
    };

    document.addEventListener("mousedown", handlePointer);
    document.addEventListener("touchstart", handlePointer, { passive: true });
    return () => {
      document.removeEventListener("mousedown", handlePointer);
      document.removeEventListener("touchstart", handlePointer);
    };
  }, [showSuggestions]);

  return (
    <div ref={containerRef} className="relative w-full max-w-md">
      <form onSubmit={onSubmit}>
        <input
          className="h-11 w-full min-h-0 rounded-full border border-border bg-card px-4 py-1 text-sm shadow-sm transition focus-visible:border-primary/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 sm:h-10"
          placeholder="Search products"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onFocus={() => setIsInputFocused(true)}
          onKeyDown={(event) => {
            if (!hasQuery) return;
            if (event.key === "ArrowDown") {
              event.preventDefault();
              if (!options.length) return;
              setActiveIndex((prev) => (prev + 1) % options.length);
              return;
            }
            if (event.key === "ArrowUp") {
              event.preventDefault();
              if (!options.length) return;
              setActiveIndex((prev) => (prev <= 0 ? options.length - 1 : prev - 1));
              return;
            }
            if (event.key === "Escape") {
              setIsInputFocused(false);
              setActiveIndex(-1);
            }
          }}
          role="combobox"
          aria-haspopup="listbox"
          aria-expanded={showSuggestions}
          aria-controls={listboxId}
          aria-autocomplete="list"
          aria-activedescendant={activeIndex >= 0 ? options[activeIndex]?.id : undefined}
        />
      </form>

      {showSuggestions ? (
        <div
          id={listboxId}
          className="absolute left-0 right-0 top-full z-50 mt-2 rounded-xl border border-border bg-card p-3 shadow-lg"
          role="listbox"
          aria-label="Search suggestions"
        >
          {suggestions.isFetching ? (
            <p className="text-sm text-foreground/60">Searching...</p>
          ) : null}

          {productSuggestions.length > 0 ? (
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">Products</p>
              <ul className="mt-2 space-y-1">
                {productSuggestions.slice(0, 5).map((item, index) => (
                  <li key={item.id}>
                    <button
                      id={`product-${item.id}`}
                      type="button"
                      role="option"
                      aria-selected={activeIndex === index}
                      className={`w-full rounded-lg px-2 py-2 text-left text-sm transition ${
                        activeIndex === index
                          ? "bg-muted text-foreground"
                          : "text-foreground/80 hover:bg-muted hover:text-foreground"
                      }`}
                      onClick={() => handleSelection(`/products/${item.slug}/`)}
                      onMouseEnter={() => setActiveIndex(index)}
                    >
                      {item.name}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {categorySuggestions.length > 0 ? (
            <div className="mt-3">
              <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">Categories</p>
              <ul className="mt-2 space-y-1">
                {categorySuggestions.slice(0, 5).map((item, index) => {
                  const optionIndex = visibleProductCount + index;
                  return (
                    <li key={item.id}>
                      <button
                        id={`category-${item.id}`}
                        type="button"
                        role="option"
                        aria-selected={activeIndex === optionIndex}
                        className={`w-full rounded-lg px-2 py-2 text-left text-sm transition ${
                          activeIndex === optionIndex
                            ? "bg-muted text-foreground"
                            : "text-foreground/80 hover:bg-muted hover:text-foreground"
                        }`}
                        onClick={() => handleSelection(`/categories/${item.slug}/`)}
                        onMouseEnter={() => setActiveIndex(optionIndex)}
                      >
                        {item.name}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : null}

          {!suggestions.isFetching && !hasSuggestions ? (
            <p className="text-sm text-foreground/60">
              No direct matches. Press Enter to search all results.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
