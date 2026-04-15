import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { apiFetch, ApiError } from "@/lib/api";
import type { Bundle } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { JsonLd } from "@/components/seo/JsonLd";
import { buildItemList, buildPageMetadata } from "@/lib/seo";
import { asArray } from "@/lib/array";
import { getLazyImageProps } from "@/lib/lazyImage";

export const revalidate = 600;
export const metadata: Metadata = buildPageMetadata({
  title: "Bundles",
  description: "Shop ready-made Bunoraa bundles with complementary products.",
  path: "/bundles/",
});

async function getBundles() {
  try {
    const response = await apiFetch<Bundle[] | { results?: Bundle[]; count?: number }>(
      "/catalog/bundles/",
      {
        next: { revalidate },
      }
    );
    return asArray<Bundle>(response.data);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return [];
    }
    throw error;
  }
}

export default async function BundlesPage() {
  const bundles = await getBundles();
  if (!bundles.length) {
    notFound();
  }
  const list = buildItemList(
    bundles.map((bundle) => ({
      name: bundle.name,
      url: `/bundles/${bundle.slug}/`,
      image: bundle.image || undefined,
      description: bundle.description || undefined,
    })),
    "Bundles"
  );

  return (
    <div className="mx-auto w-full max-w-6xl px-3 sm:px-5 py-12">
      <div className="mb-8">
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Bundles
        </p>
        <h1 className="text-3xl font-semibold">Bundle deals</h1>
      </div>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {bundles.map((bundle) => (
          <Card key={bundle.id} variant="bordered" className="flex flex-col gap-4">
            <div className="aspect-[4/3] overflow-hidden rounded-xl bg-muted">
              {bundle.image ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  {...getLazyImageProps(bundle.image, bundle.name)}
                  className="h-full w-full object-cover"
                />
              ) : null}
            </div>
            <div className="flex flex-1 flex-col gap-2">
              <h2 className="text-lg font-semibold">{bundle.name}</h2>
              <p className="text-sm text-foreground/70">{bundle.description}</p>
            </div>
            <Button asChild variant="primary-gradient">
              <Link href={`/bundles/${bundle.slug}/`}>View bundle</Link>
            </Button>
          </Card>
        ))}
      </div>
      <JsonLd data={list} />
    </div>
  );
}
