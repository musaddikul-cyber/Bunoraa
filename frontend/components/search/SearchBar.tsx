"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { ArrowUpRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { ProductListItem } from "@/lib/types";
import { buildCategoryPath } from "@/lib/categoryPaths";
import { buildProductPath } from "@/lib/productPaths";
import { getLazyImageProps } from "@/lib/lazyImage";

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
  categories: Array<{
    id: string;
    name: string;
    slug: string;
    image?: string | null;
    product_count?: number | null;
  }>;
  query: string;
};

type SearchBarProps = {
  hideSubmitButtonOnDesktop?: boolean;
};

type SuggestionOption = {
  id: string;
  label: string;
  href: string;
  kind: "product" | "category";
  newTab?: boolean;
};

function getProductImage(product: ProductListItem) {
  const primary = product.primary_image as unknown as
    | string
    | { image?: string | null }
    | null;
  if (!primary) return null;
  if (typeof primary === "string") return primary;
  return primary.image || null;
}

function getProductPrice(product: ProductListItem) {
  return product.current_price || product.sale_price || product.price || "";
}

export function SearchBar({ hideSubmitButtonOnDesktop = false }: SearchBarProps) {
  const router = useRouter();
  const [query, setQuery] = React.useState("");
  const [activeIndex, setActiveIndex] = React.useState(-1);
  const [isInputFocused, setIsInputFocused] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const inputId = React.useId();
  const listboxId = React.useId();
  const debounced = useDebouncedValue(query, 350);

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
  const productOptions: SuggestionOption[] = productSuggestions.slice(0, 6).map((item) => ({
    id: `product-${item.id}`,
    label: item.name,
    href: buildProductPath(item),
    kind: "product",
    newTab: true,
  }));
  const categoryOptions: SuggestionOption[] = categorySuggestions
    .slice(0, 5)
    .map((item) => ({
      id: `category-${item.id}`,
      label: item.name,
      href: buildCategoryPath(item.slug),
      kind: "category",
      newTab: false,
    }));
  const options = [...productOptions, ...categoryOptions];

  const trimmedQuery = query.trim();
  const hasQuery = trimmedQuery.length >= 2;
  const hasSuggestions = options.length > 0;
  const showSuggestions = isInputFocused && hasQuery;

  const handleSelection = (href: string, newTab = false) => {
    setQuery("");
    setActiveIndex(-1);
    setIsInputFocused(false);
    if (newTab) {
      window.open(href, "_blank", "noopener,noreferrer");
      return;
    }
    router.push(href);
  };

  const handleSearchResults = () => {
    if (!trimmedQuery) return;
    router.push(`/search/?q=${encodeURIComponent(trimmedQuery)}`);
    setQuery("");
    setActiveIndex(-1);
    setIsInputFocused(false);
  };

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (activeIndex >= 0 && options[activeIndex]) {
      const selected = options[activeIndex];
      handleSelection(selected.href, Boolean(selected.newTab));
      return;
    }
    handleSearchResults();
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
      <form onSubmit={onSubmit} className="relative">
        <label htmlFor={inputId} className="sr-only">
          Search products and categories
        </label>
        <input
          id={inputId}
          className={`${
            hideSubmitButtonOnDesktop ? "h-10 min-h-10 lg:h-9 lg:min-h-9" : "h-10 min-h-10"
          } w-full rounded-full border border-border bg-card px-4 py-1 text-sm shadow-sm transition focus-visible:border-primary/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 ${
            hideSubmitButtonOnDesktop ? "pr-24 lg:pr-4" : "pr-24"
          }`}
          placeholder="Search products"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onFocus={() => setIsInputFocused(true)}
          autoComplete="off"
          enterKeyHint="search"
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
        <button
          type="submit"
          className={`absolute right-1 top-1/2 inline-flex h-8 min-h-8 -translate-y-1/2 items-center justify-center rounded-full bg-primary px-3 text-xs font-semibold text-white transition hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 ${
            hideSubmitButtonOnDesktop ? "lg:hidden" : ""
          }`}
          aria-label="Search"
        >
          Search
        </button>
      </form>

      {showSuggestions ? (
        <div
          id={listboxId}
          className="absolute left-0 right-0 top-full z-50 mt-2 rounded-xl border border-border bg-card p-3 shadow-lg"
          role="listbox"
          aria-label="Search suggestions"
        >
          <div className="mb-3 flex items-center justify-between text-xs text-foreground/60">
            <p className="uppercase tracking-[0.18em]">Live results</p>
            <p>
              {productSuggestions.length} products • {categorySuggestions.length} categories
            </p>
          </div>

          {suggestions.isFetching ? (
            <p className="text-sm text-foreground/60">Searching...</p>
          ) : null}

          {productSuggestions.length > 0 ? (
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">Products</p>
              <ul className="mt-2 space-y-1.5">
                {productSuggestions.slice(0, 6).map((item, index) => {
                  const image = getProductImage(item);
                  return (
                    <li key={item.id}>
                      <button
                        id={`product-${item.id}`}
                        type="button"
                        role="option"
                        aria-selected={activeIndex === index}
                        className={`flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-sm transition ${
                          activeIndex === index
                            ? "bg-muted text-foreground"
                            : "text-foreground/80 hover:bg-muted hover:text-foreground"
                        }`}
                        onClick={() => handleSelection(buildProductPath(item), true)}
                        onMouseEnter={() => setActiveIndex(index)}
                      >
                        <div className="h-10 w-10 shrink-0 overflow-hidden rounded-md bg-muted">
                          {image ? (
                            <img
                              {...getLazyImageProps(image, item.name)}
                              className="h-full w-full object-cover"
                            />
                          ) : null}
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="line-clamp-1 font-medium">{item.name}</p>
                          <p className="line-clamp-1 text-xs text-foreground/60">
                            {item.primary_category_name || "Product"} • {getProductPrice(item)}{" "}
                            {item.currency}
                          </p>
                        </div>
                        <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-foreground/60" />
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : null}

          {categorySuggestions.length > 0 ? (
            <div className="mt-3">
              <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">Categories</p>
              <ul className="mt-2 space-y-1">
                {categorySuggestions.slice(0, 5).map((item, index) => {
                  const optionIndex = productOptions.length + index;
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
                        onClick={() => handleSelection(buildCategoryPath(item.slug))}
                        onMouseEnter={() => setActiveIndex(optionIndex)}
                      >
                        <p className="font-medium">{item.name}</p>
                        <p className="text-xs text-foreground/60">Category</p>
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

          <div className="mt-3 border-t border-border pt-2">
            <button
              type="button"
              className="w-full rounded-lg px-2 py-2 text-left text-sm font-medium text-primary transition hover:bg-primary/10"
              onClick={handleSearchResults}
            >
              View all results for "{trimmedQuery}"
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
