import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { ProductListItem, Collection } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { WishlistIconButton } from "@/components/wishlist/WishlistIconButton";
import { getServerLocaleHeaders } from "@/lib/serverLocale";

export const revalidate = 300;

type HomepageData = {
  featured_products: ProductListItem[];
  new_arrivals: ProductListItem[];
  bestsellers: ProductListItem[];
  on_sale: ProductListItem[];
  featured_categories: Array<{ id: string; name: string; slug: string; image?: string | null }>;
  collections: Collection[];
};

async function getHomepageData() {
  const response = await apiFetch<HomepageData>("/catalog/homepage/", {
    headers: await getServerLocaleHeaders(),
    next: { revalidate },
  });
  return response.data;
}

export default async function Home() {
  const data = await getHomepageData();
  const featuredProducts = data.featured_products || [];
  const newArrivals = data.new_arrivals || [];
  const onSale = data.on_sale || [];
  const bestsellers = data.bestsellers || [];

  const getImage = (product: ProductListItem) => {
    const primary = product.primary_image as unknown as
      | string
      | { image?: string | null }
      | null;
    if (!primary) return null;
    if (typeof primary === "string") return primary;
    return primary.image || null;
  };

  const getPrice = (product: ProductListItem) => {
    return (
      product.current_price ||
      product.sale_price ||
      product.price ||
      ""
    );
  };

  const getCurrency = (product: ProductListItem) => {
    return (
      product.currency ||
      (product as unknown as { currency_code?: string }).currency_code ||
      ""
    );
  };

  return (
    <div className="bg-background text-foreground">
      <section className="mx-auto w-full max-w-7xl px-6 py-14">
        <div className="grid gap-8 lg:grid-cols-[1.2fr_1fr]">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Bunoraa
            </p>
            <h1 className="mt-3 text-4xl font-semibold sm:text-5xl">
              Handcrafted collections for modern living
            </h1>
            <p className="mt-4 max-w-2xl text-foreground/70">
              Shop curated pieces from artisanal makers. Fast SSR, ISR, and live
              cart updates keep the experience seamless.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Button asChild variant="primary-gradient">
                <Link href="/products/">Shop products</Link>
              </Button>
              <Button asChild variant="secondary">
                <Link href="/collections/">Browse collections</Link>
              </Button>
            </div>
          </div>
          <Card variant="modern-gradient" className="flex flex-col gap-4">
            <h2 className="text-xl font-semibold">Featured categories</h2>
            <div className="grid gap-3 sm:grid-cols-2">
              {data.featured_categories.map((category) => (
                <Link
                  key={category.id}
                  href={`/categories/${category.slug}/`}
                  className="rounded-xl border border-border bg-card/60 p-4 text-sm"
                >
                  <p className="font-medium">{category.name}</p>
                  <p className="text-foreground/60">Explore now</p>
                </Link>
              ))}
            </div>
          </Card>
        </div>
      </section>

      <section className="mx-auto w-full max-w-7xl px-6 pb-10">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-semibold">Featured products</h2>
          <Button asChild variant="ghost">
            <Link href="/products/">View all</Link>
          </Button>
        </div>
        <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {featuredProducts.map((product) => {
            const image = getImage(product);
            return (
            <Card key={product.id} variant="bordered" className="flex flex-col gap-3">
              <div className="relative aspect-[4/5] overflow-hidden rounded-xl bg-muted">
                <WishlistIconButton
                  productId={product.id}
                  className="absolute right-3 top-3"
                />
                {image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={image}
                    alt={product.name}
                    className="h-full w-full object-cover"
                  />
                ) : null}
              </div>
              <div className="flex flex-1 flex-col gap-1">
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                  {product.primary_category_name || "Featured"}
                </p>
                <p className="font-medium">{product.name}</p>
                <p className="text-sm text-foreground/70">
                  {getPrice(product)} {getCurrency(product)}
                </p>
              </div>
              <Button asChild size="sm" variant="secondary">
                <Link href={`/products/${product.slug}/`}>View</Link>
              </Button>
            </Card>
          )})}
        </div>
      </section>

      <section className="mx-auto w-full max-w-7xl px-6 pb-12">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-semibold">New arrivals</h2>
          <Button asChild variant="ghost">
            <Link href="/products/?ordering=-created_at">View all</Link>
          </Button>
        </div>
        <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {newArrivals.slice(0, 4).map((product) => {
            const image = getImage(product);
            return (
              <Card key={product.id} variant="bordered" className="flex flex-col gap-3">
                <div className="relative aspect-[4/5] overflow-hidden rounded-xl bg-muted">
                  <WishlistIconButton
                    productId={product.id}
                    className="absolute right-3 top-3"
                  />
                  {image ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={image}
                      alt={product.name}
                      className="h-full w-full object-cover"
                    />
                  ) : null}
                </div>
                <div className="flex flex-1 flex-col gap-1">
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                    {product.primary_category_name || "New arrival"}
                  </p>
                  <p className="font-medium">{product.name}</p>
                  <p className="text-sm text-foreground/70">
                    {getPrice(product)} {getCurrency(product)}
                  </p>
                </div>
                <Button asChild size="sm" variant="secondary">
                  <Link href={`/products/${product.slug}/`}>View</Link>
                </Button>
              </Card>
            );
          })}
        </div>

        <div className="mt-12 flex items-center justify-between">
          <h2 className="text-2xl font-semibold">On sale</h2>
          <Button asChild variant="ghost">
            <Link href="/products/?ordering=on_sale">View all</Link>
          </Button>
        </div>
        <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {onSale.slice(0, 4).map((product) => {
            const image = getImage(product);
            return (
              <Card key={product.id} variant="bordered" className="flex flex-col gap-3">
                <div className="relative aspect-[4/5] overflow-hidden rounded-xl bg-muted">
                  <WishlistIconButton
                    productId={product.id}
                    className="absolute right-3 top-3"
                  />
                  {image ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={image}
                      alt={product.name}
                      className="h-full w-full object-cover"
                    />
                  ) : null}
                </div>
                <div className="flex flex-1 flex-col gap-1">
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                    {product.primary_category_name || "On sale"}
                  </p>
                  <p className="font-medium">{product.name}</p>
                  <p className="text-sm text-foreground/70">
                    {getPrice(product)} {getCurrency(product)}
                  </p>
                </div>
                <Button asChild size="sm" variant="secondary">
                  <Link href={`/products/${product.slug}/`}>View</Link>
                </Button>
              </Card>
            );
          })}
        </div>

        {bestsellers.length ? (
          <>
            <div className="mt-12 flex items-center justify-between">
              <h2 className="text-2xl font-semibold">Bestsellers</h2>
              <Button asChild variant="ghost">
                <Link href="/products/?ordering=-sales_count">View all</Link>
              </Button>
            </div>
            <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
              {bestsellers.slice(0, 4).map((product) => {
                const image = getImage(product);
                return (
                  <Card
                    key={product.id}
                    variant="bordered"
                    className="flex flex-col gap-3"
                  >
                    <div className="relative aspect-[4/5] overflow-hidden rounded-xl bg-muted">
                      <WishlistIconButton
                        productId={product.id}
                        className="absolute right-3 top-3"
                      />
                      {image ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={image}
                          alt={product.name}
                          className="h-full w-full object-cover"
                        />
                      ) : null}
                    </div>
                    <div className="flex flex-1 flex-col gap-1">
                      <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                        {product.primary_category_name || "Bestseller"}
                      </p>
                      <p className="font-medium">{product.name}</p>
                      <p className="text-sm text-foreground/70">
                        {getPrice(product)} {getCurrency(product)}
                      </p>
                    </div>
                    <Button asChild size="sm" variant="secondary">
                      <Link href={`/products/${product.slug}/`}>View</Link>
                    </Button>
                  </Card>
                );
              })}
            </div>
          </>
        ) : null}
      </section>
    </div>
  );
}
