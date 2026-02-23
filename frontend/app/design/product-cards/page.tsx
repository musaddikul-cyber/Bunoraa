import type { Metadata } from "next";
import { apiFetch } from "@/lib/api";
import type { ProductListItem } from "@/lib/types";
import { getServerLocaleHeaders } from "@/lib/serverLocale";
import { buildNoIndexMetadata } from "@/lib/seo";
import {
  ProductCardVariant,
  PRODUCT_CARD_VARIANTS,
  type ProductCardVariantName,
} from "@/components/products/ProductCardVariants";

export const revalidate = 300;

export const metadata: Metadata = buildNoIndexMetadata({
  title: "Product Card Variants",
  description: "Internal preview page for product card variants.",
  path: "/design/product-cards/",
});

async function getProducts() {
  const response = await apiFetch<ProductListItem[] | { results?: ProductListItem[] }>(
    "/catalog/products/",
    {
      params: { page_size: 24, ordering: "-created_at" },
      headers: await getServerLocaleHeaders(),
      next: { revalidate },
    }
  );
  const payload = response.data;
  if (Array.isArray(payload)) return payload;
  if (payload && Array.isArray(payload.results)) return payload.results;
  return [];
}

const FULL_WIDTH_VARIANTS = new Set<ProductCardVariantName>([
  "horizontal",
  "dense-row",
]);

export default async function ProductCardDesignPage() {
  const products = await getProducts();

  if (!products.length) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-6xl px-4 py-12 sm:px-6">
          <h1 className="text-3xl font-semibold">Product Card Variants</h1>
          <p className="mt-3 text-sm text-foreground/65">
            No products are available to preview card variants yet.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6 sm:py-12">
        <div className="mb-8 space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-foreground/55">Internal Pattern Lab</p>
          <h1 className="text-3xl font-semibold sm:text-4xl">12 Product Card Variants</h1>
          <p className="max-w-3xl text-sm text-foreground/65 sm:text-base">
            Use these cards across different surfaces such as category grids, homepage modules,
            promotional slots, and dense list pages.
          </p>
        </div>

        <div className="space-y-8">
          {PRODUCT_CARD_VARIANTS.map((entry, index) => {
            const product = products[index % products.length];
            const fullWidth = FULL_WIDTH_VARIANTS.has(entry.id);

            return (
              <section
                key={entry.id}
                className="rounded-2xl border border-border/70 bg-card/30 p-4 sm:p-5"
              >
                <div className="mb-4 flex flex-col gap-1">
                  <h2 className="text-lg font-semibold sm:text-xl">{entry.name}</h2>
                  <p className="text-sm text-foreground/65">{entry.description}</p>
                  <p className="text-xs uppercase tracking-[0.16em] text-foreground/50">
                    Best for: {entry.bestFor}
                  </p>
                </div>
                <div className={fullWidth ? "" : "max-w-sm"}>
                  <ProductCardVariant
                    product={product}
                    variant={entry.id}
                    inCart={index % 4 === 0}
                  />
                </div>
              </section>
            );
          })}
        </div>
      </div>
    </div>
  );
}

