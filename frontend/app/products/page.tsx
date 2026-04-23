import type { Metadata } from "next";
import dynamic from "next/dynamic";
import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { ProductListItem, ProductFilterResponse } from "@/lib/types";
import { getServerLocaleHeaders } from "@/lib/serverLocale";
import { JsonLd } from "@/components/seo/JsonLd";
import { buildCollectionPage, buildItemList, buildPageMetadata } from "@/lib/seo";
import { buildProductPath } from "@/lib/productPaths";
import { cn } from "@/lib/utils";

const ProductGrid = dynamic(
  () => import("@/components/products/ProductGrid").then((mod) => mod.ProductGrid)
);
const FilterPanel = dynamic(
  () => import("@/components/products/FilterPanel").then((mod) => mod.FilterPanel)
);
const FilterDrawer = dynamic(
  () => import("@/components/products/FilterDrawer").then((mod) => mod.FilterDrawer)
);
const AppliedFilters = dynamic(
  () => import("@/components/products/AppliedFilters").then((mod) => mod.AppliedFilters)
);
const SortMenu = dynamic(
  () => import("@/components/products/SortMenu").then((mod) => mod.SortMenu)
);
const RecentlyViewedSection = dynamic(
  () => import("@/components/products/RecentlyViewedSection").then((mod) => mod.RecentlyViewedSection)
);

export const revalidate = 300;

type SearchParams = Record<string, string | string[] | undefined>;

function firstValue(value: string | string[] | undefined): string | undefined {
  if (Array.isArray(value)) return value[0];
  return value;
}

function hasIndexBustingFilters(searchParams: SearchParams): boolean {
  return Object.entries(searchParams).some(([key, value]) => {
    if (key === "page" || key === "view") return false;
    if (Array.isArray(value)) return value.some((entry) => entry.trim() !== "");
    return Boolean(value && value.trim() !== "");
  });
}

function parsePageNumber(searchParams: SearchParams): number {
  const rawPage = firstValue(searchParams.page);
  const page = Number(rawPage || 1);
  return Number.isFinite(page) && page > 1 ? Math.floor(page) : 1;
}

export async function generateMetadata({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}): Promise<Metadata> {
  const resolved = await searchParams;
  const page = parsePageNumber(resolved);
  const hasFilters = hasIndexBustingFilters(resolved);
  const base = buildPageMetadata({
    title: "Shop Products",
    description: "Browse all Bunoraa products, new arrivals, and best-value picks.",
    path: page > 1 && !hasFilters ? `/products/?page=${page}` : "/products/",
  });

  if (!hasFilters) {
    return base;
  }

  return {
    ...base,
    alternates: {
      canonical: "/products/",
    },
    robots: {
      index: false,
      follow: true,
      googleBot: {
        index: false,
        follow: true,
        "max-snippet": -1,
        "max-image-preview": "large",
        "max-video-preview": -1,
      },
    },
  };
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
  const totalCount = pagination?.count ?? products.length;
  const showFilters = totalCount > 1;
  const showPagination =
    (pagination?.total_pages ? pagination.total_pages > 1 : totalCount > products.length) &&
    products.length > 0;
  const showRecentlyViewed = totalCount > 1;

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
  const totalPages = pagination?.total_pages || 1;
  const windowSize = 5;
  const startPage = Math.max(1, Math.min(currentPage - 2, totalPages - windowSize + 1));
  const endPage = Math.min(totalPages, startPage + windowSize - 1);
  const pageNumbers = Array.from({ length: Math.max(0, endPage - startPage + 1) }, (_, i) => startPage + i);
  const showFirst = startPage > 1;
  const showLast = endPage < totalPages;

  const listId = "/products/#itemlist";
  const productList = buildItemList(
    products.slice(0, 50).map((product) => ({
      name: product.name,
      url: buildProductPath(product),
      image: (product.primary_image as string | undefined) || undefined,
      description: product.short_description || undefined,
    })),
    "Products",
    listId
  );
  const collectionPage = buildCollectionPage({
    name: "Products",
    description: "Shop the Bunoraa catalog.",
    url: "/products/",
    itemListId: listId,
  });

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-7xl px-3 sm:px-5 py-10">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h1 className="text-2xl font-semibold uppercase tracking-[0.12em] sm:text-3xl">
            Products
          </h1>
          <div className="flex w-full flex-wrap items-center gap-3 sm:w-auto">
            {showFilters ? (
              <FilterDrawer
                filters={filterData}
                productCount={totalCount}
                className="lg:hidden"
                filterParams={filterParams}
                variant="minimal"
              />
            ) : null}
            <SortMenu variant="minimal" />
          </div>
        </div>

        <div className={showFilters ? "mt-8 grid gap-8 lg:grid-cols-[220px_1fr]" : "mt-8 grid gap-8"}>
          {showFilters ? (
            <aside className="hidden lg:block">
              <FilterPanel
                filters={filterData}
                productCount={totalCount}
                filterParams={filterParams}
                variant="minimal"
              />
            </aside>
          ) : null}
          <div className="space-y-6">
            <AppliedFilters variant="minimal" />
            <ProductGrid products={products} view={view} cardStyle="minimal" />

            {showPagination ? (
              <div className="mt-10 flex flex-wrap items-center justify-center gap-2 text-sm">
                {pagination?.previous ? (
                  <Link
                    href={pageLink(currentPage - 1)}
                    className="rounded-full border border-border px-3 py-1.5 hover:border-foreground"
                  >
                    Prev
                  </Link>
                ) : null}
                {showFirst ? (
                  <>
                    <Link
                      href={pageLink(1)}
                      className="rounded-full border border-border px-3 py-1.5 hover:border-foreground"
                    >
                      1
                    </Link>
                    <span className="px-2 text-foreground/50">...</span>
                  </>
                ) : null}
                {pageNumbers.map((page) => (
                  <Link
                    key={page}
                    href={pageLink(page)}
                    className={cn(
                      "rounded-full border px-3 py-1.5",
                      page === currentPage
                        ? "border-foreground bg-foreground text-background"
                        : "border-border hover:border-foreground"
                    )}
                  >
                    {page}
                  </Link>
                ))}
                {showLast ? (
                  <>
                    <span className="px-2 text-foreground/50">...</span>
                    <Link
                      href={pageLink(totalPages)}
                      className="rounded-full border border-border px-3 py-1.5 hover:border-foreground"
                    >
                      {totalPages}
                    </Link>
                  </>
                ) : null}
                {pagination?.next ? (
                  <Link
                    href={pageLink(currentPage + 1)}
                    className="rounded-full border border-border px-3 py-1.5 hover:border-foreground"
                  >
                    Next
                  </Link>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>

        {showRecentlyViewed ? (
          <div className="mt-12">
            <RecentlyViewedSection />
          </div>
        ) : null}
      </div>
      {products.length ? <JsonLd data={[collectionPage, productList]} /> : null}
    </div>
  );
}
