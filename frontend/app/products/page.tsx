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
import { getServerLocaleHeaders } from "@/lib/serverLocale";
import { JsonLd } from "@/components/seo/JsonLd";
import { buildItemList } from "@/lib/seo";

export const revalidate = 300;

type SearchParams = Record<string, string | string[] | undefined>;

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
      params[key] = key === "page" ? Number(value) || 1 : value;
    }
  });

  return apiFetch<ProductListItem[]>("/catalog/products/", {
    params,
    headers: await getServerLocaleHeaders(),
    next: { revalidate },
  });
}

async function getFilters(searchParams: SearchParams) {
  const params: Record<string, string> = {};
  if (searchParams.q && typeof searchParams.q === "string") {
    params.q = searchParams.q;
  }
  const response = await apiFetch<ProductFilterResponse>("/catalog/products/filters/", {
    params,
    headers: await getServerLocaleHeaders(),
    cache: "no-store",
  });
  return response.data;
}

export default async function ProductsPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const resolved = await searchParams;
  const currentPage = Number(resolved.page || 1) || 1;
  const view = resolved.view === "list" ? "list" : "grid";
  const filterParams =
    resolved.q && typeof resolved.q === "string" && resolved.q.trim()
      ? { q: resolved.q }
      : undefined;

  const [productsResponse, filterData] = await Promise.all([
    getProducts(resolved),
    getFilters(resolved).catch(() => null),
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

  const productList = buildItemList(
    products.slice(0, 50).map((product) => ({
      name: product.name,
      url: `/products/${product.slug}/`,
      image: (product.primary_image as string | undefined) || undefined,
      description: product.short_description || undefined,
    })),
    "Products"
  );

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-7xl px-6 py-12">
        <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Products
            </p>
            <h1 className="text-3xl font-semibold">Shop the catalog</h1>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <FilterDrawer filters={filterData} className="lg:hidden" filterParams={filterParams} />
            <SortMenu />
            <ViewToggle />
          </div>
        </div>

        <div className="grid gap-8 lg:grid-cols-[260px_1fr]">
          <aside className="hidden lg:block">
            <FilterPanel filters={filterData} filterParams={filterParams} />
          </aside>
          <div className="space-y-6">
            <AppliedFilters />
            <ProductGrid products={products} view={view} />

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
      {products.length ? <JsonLd data={productList} /> : null}
    </div>
  );
}
