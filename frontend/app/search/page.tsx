import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { ProductListItem, ProductFilterResponse } from "@/lib/types";
import { ProductGrid } from "@/components/products/ProductGrid";
import { FilterPanel } from "@/components/products/FilterPanel";
import { FilterDrawer } from "@/components/products/FilterDrawer";
import { AppliedFilters } from "@/components/products/AppliedFilters";
import { SortMenu } from "@/components/products/SortMenu";
import { ViewToggle } from "@/components/products/ViewToggle";
import { Button } from "@/components/ui/Button";
import { RecentlyViewedSection } from "@/components/products/RecentlyViewedSection";
import type { CategoryFacet } from "@/components/products/FilterPanel";

export const revalidate = 60;

type SearchParams = Record<string, string | string[] | undefined>;

type SearchResponse = {
  products: ProductListItem[];
  categories: Array<{ id: string; name: string; slug: string }>;
  query: string;
};

async function getSearchMeta(query: string) {
  const response = await apiFetch<SearchResponse>("/catalog/search/", {
    params: { q: query },
    next: { revalidate },
  });
  return response.data;
}

async function getProducts(searchParams: SearchParams) {
  const params: Record<string, string | number | boolean | Array<string | number | boolean> | undefined> = {};
  Object.entries(searchParams).forEach(([key, value]) => {
    if (key === "view") return;
    if (value === undefined) return;
    if (Array.isArray(value)) {
      params[key] = value;
      return;
    }
    if (value !== "") {
      if (key === "q") {
        params.search = value;
      } else {
        params[key] = key === "page" ? Number(value) || 1 : value;
      }
    }
  });

  return apiFetch<ProductListItem[]>("/catalog/products/", {
    params,
    next: { revalidate },
  });
}

async function getFilters(query: string) {
  const response = await apiFetch<ProductFilterResponse>("/catalog/products/filters/", {
    params: query ? { q: query } : undefined,
    next: { revalidate },
  });
  return response.data;
}

async function getCategoryFacets(slug: string) {
  const response = await apiFetch<CategoryFacet[]>(
    `/catalog/categories/${slug}/facets/`,
    { next: { revalidate } }
  );
  return response.data;
}

export default async function SearchPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const resolved = await searchParams;
  const query = typeof resolved.q === "string" ? resolved.q : "";
  const view = resolved.view === "list" ? "list" : "grid";
  const currentPage = Number(resolved.page || 1) || 1;

  if (!query) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-6xl px-6 py-12">
          <div className="mb-8">
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Search
            </p>
            <h1 className="text-3xl font-semibold">Search the catalog</h1>
          </div>
          <p className="text-sm text-foreground/60">Add a query using ?q=your-search.</p>
        </div>
      </div>
    );
  }

  const [meta, productsResponse, filterData] = await Promise.all([
    getSearchMeta(query),
    getProducts(resolved),
    getFilters(query).catch(() => null),
  ]);

  const facetCategory =
    (typeof resolved.category === "string" && resolved.category) ||
    meta.categories[0]?.slug ||
    "";

  const facets = facetCategory
    ? await getCategoryFacets(facetCategory).catch(() => [])
    : [];

  const rawData = productsResponse.data as
    | ProductListItem[]
    | {
        results?: ProductListItem[];
        count?: number;
        next?: string | null;
        previous?: string | null;
      };
  const products = Array.isArray(rawData)
    ? rawData
    : Array.isArray(rawData?.results)
    ? rawData.results
    : [];

  const pagination =
    productsResponse.meta?.pagination ||
    (rawData && !Array.isArray(rawData)
      ? {
          count: rawData.count ?? products.length,
          next: rawData.next ?? null,
          previous: rawData.previous ?? null,
          page: currentPage,
          page_size: products.length,
          total_pages: rawData.count
            ? Math.max(1, Math.ceil(rawData.count / Math.max(products.length, 1)))
            : 1,
        }
      : undefined);

  const baseParams = new URLSearchParams();
  Object.entries(resolved).forEach(([key, value]) => {
    if (key === "page" || value === undefined) return;
    if (Array.isArray(value)) {
      value.forEach((item) => baseParams.append(key, item));
    } else if (value !== "") {
      baseParams.set(key, value);
    }
  });

  const pageLink = (page: number) => {
    const params = new URLSearchParams(baseParams.toString());
    params.set("page", String(page));
    return `?${params.toString()}`;
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-7xl px-6 py-12">
        <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Search
            </p>
            <h1 className="text-3xl font-semibold">Results for "{query}"</h1>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <FilterDrawer filters={filterData} facets={facets} className="lg:hidden" />
            <SortMenu />
            <ViewToggle />
          </div>
        </div>

        {meta.categories.length ? (
          <div className="mb-6 flex flex-wrap gap-2">
            {meta.categories.map((category) => (
              <Link
                key={category.id}
                className="rounded-full border border-border px-4 py-2 text-sm"
                href={`/categories/${category.slug}/`}
              >
                {category.name}
              </Link>
            ))}
          </div>
        ) : null}

        <div className="grid gap-8 lg:grid-cols-[260px_1fr]">
          <aside className="hidden lg:block">
            <FilterPanel filters={filterData} facets={facets} />
          </aside>
          <div className="space-y-6">
            <AppliedFilters />
            <ProductGrid products={products} view={view} emptyMessage="No products found." />

            <div className="mt-10 flex items-center justify-between">
              {pagination?.previous ? (
                <Button asChild variant="ghost" size="sm">
                  <Link href={pageLink(currentPage - 1)}>Previous</Link>
                </Button>
              ) : (
                <span className="rounded-xl px-4 py-2 text-sm text-foreground/40">
                  Previous
                </span>
              )}
              <span className="text-sm text-foreground/60">
                Page {currentPage}
                {pagination?.total_pages ? ` of ${pagination.total_pages}` : ""}
              </span>
              {pagination?.next ? (
                <Button asChild variant="ghost" size="sm">
                  <Link href={pageLink(currentPage + 1)}>Next</Link>
                </Button>
              ) : (
                <span className="rounded-xl px-4 py-2 text-sm text-foreground/40">
                  Next
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="mt-12">
          <RecentlyViewedSection />
        </div>
      </div>
    </div>
  );
}
