import Link from "next/link";
import { apiFetch } from "@/lib/api";
import { Card } from "@/components/ui/Card";
import { JsonLd } from "@/components/seo/JsonLd";
import { buildCollectionPage, buildItemList } from "@/lib/seo";

export const revalidate = 300;

type Category = {
  id: string;
  name: string;
  slug: string;
  path?: string;
  image?: string | null;
  icon?: string | null;
  product_count?: number;
};

async function getCategories() {
  const response = await apiFetch<Category[]>("/catalog/categories/", {
    params: { parent_id: "null" },
    next: { revalidate },
  });
  return response.data;
}

export default async function CategoriesPage() {
  const categories = await getCategories();
  const listId = "/categories/#itemlist";
  const list = buildItemList(
    categories.map((category) => ({
      name: category.name,
      url: `/categories/${category.slug}/`,
      image: category.image || undefined,
      description: undefined,
    })),
    "Categories",
    listId
  );
  const collectionPage = buildCollectionPage({
    name: "Categories",
    description: "Browse Bunoraa product categories.",
    url: "/categories/",
    itemListId: listId,
  });

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-5xl px-4 sm:px-6 py-12">
        <div className="mb-8">
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Categories
          </p>
          <h1 className="text-3xl font-semibold">Browse categories</h1>
        </div>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {categories.map((category) => (
            <Card key={category.id} variant="bordered" className="flex flex-col gap-4">
              <div className="aspect-[4/3] overflow-hidden rounded-xl bg-muted">
                {category.image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={category.image}
                    alt={category.name}
                    className="h-full w-full object-cover"
                  />
                ) : null}
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">{category.name}</h2>
                  {typeof category.product_count === "number" ? (
                    <p className="text-sm text-foreground/60">
                      {category.product_count} products
                    </p>
                  ) : null}
                </div>
                <Link
                  className="text-sm font-medium text-primary"
                  href={`/categories/${category.slug}/`}
                >
                  View
                </Link>
              </div>
            </Card>
          ))}
        </div>
      </div>
      {categories.length ? <JsonLd data={[collectionPage, list]} /> : null}
    </div>
  );
}
