import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { Artisan } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { JsonLd } from "@/components/seo/JsonLd";
import { buildItemList } from "@/lib/seo";

export const revalidate = 600;

async function tryGetArtisans() {
  try {
    const response = await apiFetch<Artisan[]>("/artisans/", {
      next: { revalidate },
    });
    return response.data;
  } catch {
    return [] as Artisan[];
  }
}

export default async function ArtisansPage() {
  const artisans = await tryGetArtisans();
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
    <div className="mx-auto w-full max-w-6xl px-4 sm:px-6 py-12">
      <div className="mb-8">
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Artisans
        </p>
        <h1 className="text-3xl font-semibold">Meet the makers</h1>
        <p className="mt-2 text-foreground/70">
          Artisan profiles will appear here once the API is enabled.
        </p>
      </div>
      {artisans.length === 0 ? (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          Artisan data is not available via the API yet. Enable an artisans API
          endpoint in Django to populate this page.
        </Card>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {artisans.map((artisan) => (
            <Card key={artisan.id} variant="bordered" className="p-4">
              <h2 className="text-lg font-semibold">{artisan.name}</h2>
              <p className="text-sm text-foreground/70">{artisan.bio}</p>
              <Link className="text-sm text-primary" href={`/artisans/${artisan.slug}/`}>
                View profile
              </Link>
            </Card>
          ))}
        </div>
      )}
      {artisans.length ? <JsonLd data={list} /> : null}
    </div>
  );
}
