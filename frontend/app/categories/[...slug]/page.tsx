import { apiFetch, ApiError } from "@/lib/api";
import type { ProductListItem } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import Link from "next/link";
import { notFound } from "next/navigation";
import { WishlistIconButton } from "@/components/wishlist/WishlistIconButton";

export const revalidate = 300;

type Category = {
  id: string;
  name: string;
  slug: string;
  description?: string | null;
  meta_title?: string | null;
  meta_description?: string | null;
};

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

async function getCategoryProducts(slug: string, page: number) {
  const response = await apiFetch<ProductListItem[]>(
    `/catalog/categories/${slug}/products/`,
    {
      params: { page },
      next: { revalidate },
    }
  );
  return response;
}

export default async function CategoryPage({
  params,
  searchParams,
}: {
  params: Promise<{ slug: string[] }>;
  searchParams: Promise<{ page?: string }>;
}) {
  const [{ slug }, resolvedSearchParams] = await Promise.all([
    params,
    searchParams,
  ]);
  const slugPath = slug.join("/");
  const page = Number(resolvedSearchParams.page || 1) || 1;
  const [category, productsResponse] = await Promise.all([
    getCategory(slugPath),
    getCategoryProducts(slugPath, page),
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

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-6xl px-6 py-12">
        <div className="mb-8">
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Category
          </p>
          <h1 className="text-3xl font-semibold">{category.name}</h1>
          <p className="mt-2 text-foreground/70">
            {category.meta_description || "Explore products in this category."}
          </p>
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
          {pagination?.previous ? (
            <Button asChild variant="ghost" size="sm">
              <Link href={`/categories/${slugPath}/?page=${page - 1}`}>
                Previous
              </Link>
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
              <Link href={`/categories/${slugPath}/?page=${page + 1}`}>
                Next
              </Link>
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
