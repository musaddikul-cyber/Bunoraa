import { apiFetch } from "@/lib/api";
import type { Artisan, ProductListItem } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import Link from "next/link";
import { WishlistIconButton } from "@/components/wishlist/WishlistIconButton";
import { JsonLd } from "@/components/seo/JsonLd";
import { absoluteUrl, buildBreadcrumbList, buildItemList, cleanObject } from "@/lib/seo";

export const revalidate = 600;

async function tryGetArtisan(slug: string) {
  try {
    const response = await apiFetch<Artisan>(`/artisans/${slug}/`, {
      next: { revalidate },
    });
    return response.data;
  } catch {
    return null;
  }
}

async function tryGetArtisanProducts(slug: string) {
  try {
    const response = await apiFetch<ProductListItem[]>("/catalog/products/", {
      params: { artisan: slug },
      next: { revalidate },
    });
    return response.data;
  } catch {
    return [] as ProductListItem[];
  }
}

export default async function ArtisanDetailPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const [artisan, products] = await Promise.all([
    tryGetArtisan(slug),
    tryGetArtisanProducts(slug),
  ]);
  const artisanUrl = `/artisans/${slug}/`;
  const breadcrumbs = buildBreadcrumbList([
    { name: "Home", url: "/" },
    { name: "Artisans", url: "/artisans/" },
    { name: artisan?.name || "Artisan", url: artisanUrl },
  ]);
  const personSchema = artisan
    ? cleanObject({
        "@context": "https://schema.org",
        "@type": "Person",
        name: artisan.name,
        description: artisan.bio || undefined,
        image: artisan.avatar ? absoluteUrl(artisan.avatar) : undefined,
        url: absoluteUrl(artisanUrl),
      })
    : null;
  const productList = buildItemList(
    products.slice(0, 50).map((product) => ({
      name: product.name,
      url: `/products/${product.slug}/`,
      image: (product.primary_image as string | undefined) || undefined,
      description: product.short_description || undefined,
    })),
    artisan?.name ? `${artisan.name} products` : "Artisan products"
  );

  return (
    <div className="mx-auto w-full max-w-6xl px-4 sm:px-6 py-12">
      <div className="mb-8">
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Artisan
        </p>
        <h1 className="text-3xl font-semibold">{artisan?.name || "Artisan profile"}</h1>
        <p className="mt-2 text-foreground/70">
          {artisan?.bio || "Artisan details are not available via API yet."}
        </p>
      </div>
      {products.length > 0 ? (
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
                <p className="text-sm text-foreground/70">{product.short_description}</p>
              </div>
              <Button asChild size="sm" variant="secondary">
                <Link href={`/products/${product.slug}/`}>View product</Link>
              </Button>
            </Card>
          ))}
        </div>
      ) : (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          Products for this artisan are not available via the API yet.
        </Card>
      )}
      <JsonLd
        data={[
          breadcrumbs,
          ...(personSchema ? [personSchema] : []),
          ...(products.length ? [productList] : []),
        ]}
      />
    </div>
  );
}
