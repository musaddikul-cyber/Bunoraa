import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { ProductListItem } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { ProductFilterBar } from "@/components/products/ProductFilterBar";
import { WishlistIconButton } from "@/components/wishlist/WishlistIconButton";

export const revalidate = 300;

type SearchParams = {
  page?: string;
  ordering?: string;
  q?: string;
};

async function getProducts(searchParams: SearchParams) {
  const page = Number(searchParams.page || 1) || 1;
  const ordering = searchParams.ordering || "";
  const search = searchParams.q || "";

  return apiFetch<ProductListItem[]>("/catalog/products/", {
    params: {
      page,
      ordering: ordering || undefined,
      search: search || undefined,
    },
    next: { revalidate },
  });
}

export default async function ProductsPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const resolvedSearchParams = await searchParams;
  const currentPage = Number(resolvedSearchParams.page || 1) || 1;
  const response = await getProducts(resolvedSearchParams);
  const rawData = response.data as
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
    response.meta?.pagination ||
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

  const hasPrevious = Boolean(pagination?.previous);
  const hasNext = Boolean(pagination?.next);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-6xl px-6 py-12">
        <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Products
            </p>
            <h1 className="text-3xl font-semibold">Shop the catalog</h1>
          </div>
          <ProductFilterBar />
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {products.map((product) => (
            <Card key={product.id} variant="bordered" className="flex flex-col gap-4">
              <div className="relative aspect-[4/5] overflow-hidden rounded-xl bg-muted">
                <WishlistIconButton
                  productId={product.id}
                  className="absolute right-3 top-3"
                />
                {product.primary_image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={product.primary_image}
                    alt={product.name}
                    className="h-full w-full object-cover"
                  />
                ) : null}
              </div>
              <div className="flex flex-1 flex-col gap-2">
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                  {product.primary_category_name || "Featured"}
                </p>
                <h2 className="text-lg font-semibold">{product.name}</h2>
                <p className="text-sm text-foreground/70">
                  {product.short_description}
                </p>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-lg font-semibold">
                  {product.current_price} {product.currency}
                </span>
                <Button asChild size="sm" variant="secondary">
                  <Link href={`/products/${product.slug}/`}>View</Link>
                </Button>
              </div>
            </Card>
          ))}
        </div>

        <div className="mt-10 flex items-center justify-between">
          {hasPrevious ? (
            <Button asChild variant="ghost" size="sm">
              <Link href={`/products/?page=${currentPage - 1}`}>Previous</Link>
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
          {hasNext ? (
            <Button asChild variant="ghost" size="sm">
              <Link href={`/products/?page=${currentPage + 1}`}>Next</Link>
            </Button>
          ) : (
            <span className="rounded-xl px-4 py-2 text-sm text-foreground/40">
              Next
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
