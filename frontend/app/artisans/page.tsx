import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { apiFetch, ApiError } from "@/lib/api";
import type { Artisan } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { JsonLd } from "@/components/seo/JsonLd";
import { buildItemList, buildPageMetadata } from "@/lib/seo";
import { asArray } from "@/lib/array";
import { getLazyImageProps } from "@/lib/lazyImage";

export const revalidate = 600;
export const metadata: Metadata = buildPageMetadata({
  title: "Artisans",
  description: "Meet Bunoraa artisans and explore products from each maker.",
  path: "/artisans/",
});

async function getArtisans() {
  try {
    const response = await apiFetch<Artisan[] | { results?: Artisan[]; count?: number }>(
      "/artisans/",
      {
        next: { revalidate },
      }
    );
    return asArray<Artisan>(response.data);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return [];
    }
    throw error;
  }
}

export default async function ArtisansPage() {
  const artisans = await getArtisans();
  if (!artisans.length) {
    notFound();
  }

  const list = buildItemList(
    artisans.map((artisan) => ({
      name: artisan.name,
      url: `/artisans/${artisan.slug}/`,
      image: artisan.avatar || undefined,
      description: artisan.bio || undefined,
    })),
    "Artisans"
  );

  return (
    <div className="mx-auto w-full max-w-6xl px-3 sm:px-5 py-12">
      <div className="mb-8">
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">Artisans</p>
        <h1 className="text-3xl font-semibold">Meet the makers</h1>
      </div>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {artisans.map((artisan) => (
          <Card key={artisan.id} variant="bordered" className="flex flex-col gap-4">
            <div className="aspect-[4/3] overflow-hidden rounded-xl bg-muted">
              {artisan.avatar ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  {...getLazyImageProps(artisan.avatar, artisan.name)}
                  className="h-full w-full object-cover"
                />
              ) : null}
            </div>
            <div className="flex flex-1 flex-col gap-2">
              <h2 className="text-lg font-semibold">{artisan.name}</h2>
              <p className="text-sm text-foreground/70">{artisan.bio}</p>
            </div>
            <Button asChild variant="primary-gradient">
              <Link href={`/artisans/${artisan.slug}/`}>View artisan</Link>
            </Button>
          </Card>
        ))}
      </div>
      <JsonLd data={list} />
    </div>
  );
}
