import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { Bundle } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export const revalidate = 600;

async function getBundles() {
  const response = await apiFetch<Bundle[]>("/catalog/bundles/", {
    next: { revalidate },
  });
  return response.data;
}

export default async function BundlesPage() {
  const bundles = await getBundles();

  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-12">
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
                  src={bundle.image}
                  alt={bundle.name}
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
    </div>
  );
}
