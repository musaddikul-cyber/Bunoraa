import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { Collection } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { JsonLd } from "@/components/seo/JsonLd";
import { buildCollectionPage, buildItemList } from "@/lib/seo";

export const revalidate = 600;

async function getCollections() {
  const response = await apiFetch<Collection[]>("/catalog/collections/", {
    next: { revalidate },
  });
  return response.data;
}

export default async function CollectionsPage() {
  const collections = await getCollections();
  const listId = "/collections/#itemlist";
  const list = buildItemList(
    collections.map((collection) => ({
      name: collection.name,
      url: `/collections/${collection.slug}/`,
      image: collection.image || undefined,
      description: collection.description || undefined,
    })),
    "Collections",
    listId
  );
  const collectionPage = buildCollectionPage({
    name: "Collections",
    description: "Curated Bunoraa collections.",
    url: "/collections/",
    itemListId: listId,
  });

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-6xl px-4 sm:px-6 py-12">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Collections
            </p>
            <h1 className="text-3xl font-semibold">Curated sets</h1>
          </div>
          <Button asChild variant="secondary">
            <Link href="/products/">Shop all products</Link>
          </Button>
        </div>

        {collections.length ? (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {collections.map((collection) => (
              <Card
                key={collection.id}
                variant="bordered"
                className="flex flex-col gap-4"
              >
                <div className="aspect-[4/3] overflow-hidden rounded-xl bg-muted">
                  {collection.image ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={collection.image}
                      alt={collection.name}
                      className="h-full w-full object-cover"
                    />
                  ) : null}
                </div>
                <div className="flex flex-1 flex-col gap-2">
                  <h2 className="text-lg font-semibold">{collection.name}</h2>
                  <p className="text-sm text-foreground/70">{collection.description}</p>
                </div>
                <Button asChild variant="primary-gradient">
                  <Link href={`/collections/${collection.slug}/`}>View collection</Link>
                </Button>
              </Card>
            ))}
          </div>
        ) : (
          <div className="rounded-2xl border border-dashed border-border bg-card/40 px-4 sm:px-6 py-10 text-center">
            <h2 className="text-lg font-semibold">Collections are coming soon</h2>
            <p className="mt-2 text-sm text-foreground/70">
              We are curating new product collections. Check back soon or browse all products.
            </p>
            <div className="mt-6 flex justify-center">
              <Button asChild variant="primary-gradient">
                <Link href="/products/">Browse products</Link>
              </Button>
            </div>
          </div>
        )}
        {collections.length ? <JsonLd data={[collectionPage, list]} /> : null}
      </div>
    </div>
  );
}
