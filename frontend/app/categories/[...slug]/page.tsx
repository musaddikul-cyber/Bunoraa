import Link from "next/link";
import { apiFetch, ApiError } from "@/lib/api";
import type { ProductListItem, ProductFilterResponse } from "@/lib/types";
import { ProductGrid } from "@/components/products/ProductGrid";
import { FilterPanel } from "@/components/products/FilterPanel";
import { FilterDrawer } from "@/components/products/FilterDrawer";
import { AppliedFilters } from "@/components/products/AppliedFilters";
import { SortMenu } from "@/components/products/SortMenu";
import { ViewToggle } from "@/components/products/ViewToggle";
import { Button } from "@/components/ui/Button";
import { notFound } from "next/navigation";
import type { CategoryFacet } from "@/components/products/FilterPanel";
import { RecentlyViewedSection } from "@/components/products/RecentlyViewedSection";

export const revalidate = 300;

type Category = {
  id: string;
  name: string;
  slug: string;
  description?: string | null;
  meta_title?: string | null;
  meta_description?: string | null;
};

type SearchParams = Record<string, string | string[] | undefined>;

async function getCategory(slug: string) {
  try {
    const response = await apiFetch<Category>(`/catalog/categories/${slug}/`, {
      next: { revalidate },
    });
    return response.data;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }
}

async function getCategoryProducts(slug: string, searchParams: SearchParams) {
  const params: Record<string, string | number | boolean | Array<string | number | boolean> | undefined> = {};
  Object.entries(searchParams).forEach(([key, value]) => {
    if (key === "view") return;
    if (value === undefined) return;
    if (Array.isArray(value)) {
      params[key] = value;
      return;
    }
    if (value !== "") {
      params[key] = key === "page" ? Number(value) || 1 : value;
    }
  });

  const response = await apiFetch<ProductListItem[]>(
    `/catalog/categories/${slug}/products/`,
    {
      params,
      next: { revalidate },
    }
  );
  return response;
}

async function getFilters(slug: string, searchParams: SearchParams) {
  const params: Record<string, string> = { category: slug };
  if (searchParams.q && typeof searchParams.q === "string") {
    params.q = searchParams.q;
  }
  const response = await apiFetch<ProductFilterResponse>("/catalog/products/filters/", {
    params,
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

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  const slugPath = slug.join("/");
  const category = await getCategory(slugPath);
  return {
    title: category.meta_title || category.name,
    description: category.meta_description || category.description || "",
  };
}

export default async function CategoryPage({
  params,
  searchParams,
}: {
  params: Promise<{ slug: string[] }>;
  searchParams: Promise<SearchParams>;
}) {
  const [{ slug }, resolvedSearchParams] = await Promise.all([params, searchParams]);
  const slugPath = slug.join("/");
  const page = Number(resolvedSearchParams.page || 1) || 1;
  const view = resolvedSearchParams.view === "list" ? "list" : "grid";

  const [category, productsResponse, filterData, facets] = await Promise.all([
    getCategory(slugPath),
    getCategoryProducts(slugPath, resolvedSearchParams),
    getFilters(slugPath, resolvedSearchParams).catch(() => null),
    getCategoryFacets(slugPath).catch(() => []),
  ]);

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
          page,
          page_size: products.length,
          total_pages: rawData.count
            ? Math.max(1, Math.ceil(rawData.count / Math.max(products.length, 1)))
            : 1,
        }
      : undefined);

  const baseParams = new URLSearchParams();
  Object.entries(resolvedSearchParams).forEach(([key, value]) => {
    if (key === "page" || value === undefined) return;
    if (Array.isArray(value)) {
      value.forEach((item) => baseParams.append(key, item));
    } else if (value !== "") {
      baseParams.set(key, value);
    }
  });

  const pageLink = (pageNumber: number) => {
    const params = new URLSearchParams(baseParams.toString());
    params.set("page", String(pageNumber));
    return `?${params.toString()}`;
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-7xl px-6 py-12">
        <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Category
            </p>
            <h1 className="text-3xl font-semibold">{category.name}</h1>
            <p className="mt-2 text-foreground/70">
              {category.meta_description || "Explore products in this category."}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <FilterDrawer filters={filterData} facets={facets} className="lg:hidden" />
            <SortMenu />
            <ViewToggle />
          </div>
        </div>

        <div className="grid gap-8 lg:grid-cols-[260px_1fr]">
          <aside className="hidden lg:block">
            <FilterPanel filters={filterData} facets={facets} />
          </aside>
          <div className="space-y-6">
            <AppliedFilters />
            <ProductGrid products={products} view={view} />

            <div className="mt-10 flex items-center justify-between">
              {pagination?.previous ? (
                <Button asChild variant="ghost" size="sm">
                  <Link href={pageLink(page - 1)}>Previous</Link>
                </Button>
              ) : (
                <span className="rounded-xl px-4 py-2 text-sm text-foreground/40">
                  Previous
                </span>
              )}
              <span className="text-sm text-foreground/60">
                Page {page}
                {pagination?.total_pages ? ` of ${pagination.total_pages}` : ""}
              </span>
              {pagination?.next ? (
                <Button asChild variant="ghost" size="sm">
                  <Link href={pageLink(page + 1)}>Next</Link>
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
