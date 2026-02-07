import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { ProductListItem } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { WishlistIconButton } from "@/components/wishlist/WishlistIconButton";

export const revalidate = 60;

type SearchResponse = {
  products: ProductListItem[];
  categories: Array<{ id: string; name: string; slug: string }>;
  query: string;
};

async function runSearch(query: string) {
  const response = await apiFetch<SearchResponse>("/catalog/search/", {
    params: { q: query },
    next: { revalidate },
  });
  return response.data;
}

export default async function SearchPage({
  searchParams,
}: {
  searchParams: Promise<{ q?: string }>;
}) {
  const resolvedSearchParams = await searchParams;
  const query = resolvedSearchParams.q || "";
  const data = query ? await runSearch(query) : null;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-6xl px-6 py-12">
        <div className="mb-8">
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Search
          </p>
          <h1 className="text-3xl font-semibold">
            {query ? `Results for "${query}"` : "Search the catalog"}
          </h1>
        </div>

        {!query ? (
          <Card variant="bordered">
            <p className="text-sm text-foreground/60">
              Add a query using ?q=your-search.
            </p>
          </Card>
        ) : (
          <div className="space-y-10">
            <div>
              <h2 className="mb-4 text-lg font-semibold">Products</h2>
              <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                {data?.products?.length ? (
                  data.products.map((product) => (
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
                      <div>
                        <h3 className="text-lg font-semibold">{product.name}</h3>
                        <p className="text-sm text-foreground/70">
                          {product.short_description}
                        </p>
                      </div>
                      <Link
                        className="text-sm font-medium text-primary"
                        href={`/products/${product.slug}/`}
                      >
                        View product
                      </Link>
                    </Card>
                  ))
                ) : (
                  <p className="text-sm text-foreground/60">No products found.</p>
                )}
              </div>
            </div>
            <div>
              <h2 className="mb-4 text-lg font-semibold">Categories</h2>
              <div className="flex flex-wrap gap-3">
                {data?.categories?.length ? (
                  data.categories.map((category) => (
                    <Link
                      key={category.id}
                      className="rounded-full border border-border px-4 py-2 text-sm"
                      href={`/categories/${category.slug}/`}
                    >
                      {category.name}
                    </Link>
                  ))
                ) : (
                  <p className="text-sm text-foreground/60">No categories found.</p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
