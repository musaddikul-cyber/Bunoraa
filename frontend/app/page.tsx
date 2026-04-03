/* eslint-disable @next/next/no-img-element */
import type { Metadata } from "next";
import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type {
  Collection,
  ProductListItem,
  SiteSettings,
} from "@/lib/types";
import { ProductGrid } from "@/components/products/ProductGrid";
import { RecentlyViewedSection } from "@/components/products/RecentlyViewedSection";
import { HomeProductTabs } from "@/components/products/HomeProductTabs";
import { HeroBannerSlider, type HeroBanner } from "@/components/promotions/HeroBannerSlider";
import { getServerLocaleHeaders } from "@/lib/serverLocale";
import { asArray } from "@/lib/array";
import { JsonLd } from "@/components/seo/JsonLd";
import { absoluteUrl, buildItemList, buildPageMetadata, cleanObject } from "@/lib/seo";
import { buildCategoryPath } from "@/lib/categoryPaths";
import { buildProductPath } from "@/lib/productPaths";

export const revalidate = 300;
export const metadata: Metadata = buildPageMetadata({
  title: "Bunoraa | Curated Products and Artisan Collections",
  description:
    "Shop curated products, themed collections, bundles, and custom preorder programs at Bunoraa.",
  path: "/",
});

type FeaturedCategory = {
  id: string;
  name: string;
  slug: string;
  image?: string | null;
  icon?: string | null;
  product_count?: number | null;
};

type Spotlight = {
  id: string;
  name: string;
  placement?: string | null;
  product?: ProductListItem | null;
  category?: FeaturedCategory | null;
  start?: string | null;
  end?: string | null;
  priority?: number | null;
  is_active?: boolean | null;
};

type HomepageData = {
  featured_products: ProductListItem[];
  new_arrivals: ProductListItem[];
  bestsellers: ProductListItem[];
  on_sale: ProductListItem[];
  featured_categories: FeaturedCategory[];
  collections: Collection[];
  spotlights?: Spotlight[];
};

type Banner = HeroBanner & {
  position?: string | null;
};

type CategorySummary = {
  id: string;
  name: string;
  slug: string;
  image?: string | null;
};

const DEFAULT_HOMEPAGE_DATA: HomepageData = {
  featured_products: [],
  new_arrivals: [],
  bestsellers: [],
  on_sale: [],
  featured_categories: [],
  collections: [],
  spotlights: [],
};

const pickText = (...values: Array<string | null | undefined>) => {
  for (const value of values) {
    if (value && value.trim()) return value.trim();
  }
  return "";
};

const getImage = (product: ProductListItem | null | undefined) => {
  if (!product) return null;
  const primary = product.primary_image as unknown as
    | string
    | { image?: string | null }
    | null;
  if (!primary) return null;
  if (typeof primary === "string") return primary;
  return primary.image || null;
};


async function getHomepageData(headers: Record<string, string>) {
  try {
    const response = await apiFetch<HomepageData>("/catalog/homepage/", {
      headers,
      next: { revalidate },
    });
    const payload =
      response.data && typeof response.data === "object" && !Array.isArray(response.data)
        ? response.data
        : {};
    return {
      ...DEFAULT_HOMEPAGE_DATA,
      ...payload,
      featured_products: asArray<ProductListItem>((payload as HomepageData).featured_products),
      new_arrivals: asArray<ProductListItem>((payload as HomepageData).new_arrivals),
      bestsellers: asArray<ProductListItem>((payload as HomepageData).bestsellers),
      on_sale: asArray<ProductListItem>((payload as HomepageData).on_sale),
      featured_categories: asArray<FeaturedCategory>(
        (payload as HomepageData).featured_categories
      ),
      collections: asArray<Collection>((payload as HomepageData).collections),
      spotlights: asArray<Spotlight>((payload as HomepageData).spotlights),
    };
  } catch {
    return DEFAULT_HOMEPAGE_DATA;
  }
}

async function getSiteSettings(headers: Record<string, string>) {
  try {
    const response = await apiFetch<SiteSettings>("/pages/settings/", {
      headers,
      next: { revalidate },
    });
    return response.data;
  } catch {
    return null;
  }
}

async function getBanners(headers: Record<string, string>, position?: string) {
  try {
    const response = await apiFetch<Banner[]>("/promotions/banners/", {
      headers,
      params: position ? { position } : undefined,
      next: { revalidate },
    });
    return asArray<Banner>(response.data);
  } catch {
    return [] as Banner[];
  }
}

async function getCategoryProducts(headers: Record<string, string>, slug: string) {
  try {
    const response = await apiFetch<
      ProductListItem[] | { results?: ProductListItem[] }
    >("/catalog/products/by-category/", {
      headers,
      params: { category: slug, page_size: 8 },
      next: { revalidate },
    });
    const payload = response.data as ProductListItem[] | { results?: ProductListItem[] };
    if (Array.isArray(payload)) return payload;
    return asArray<ProductListItem>(payload.results);
  } catch {
    return [] as ProductListItem[];
  }
}

async function getShowByCategories(headers: Record<string, string>) {
  try {
    const response = await apiFetch<CategorySummary[]>("/catalog/categories/", {
      headers,
      params: { page_size: 3, has_products: true },
      next: { revalidate },
    });
    return asArray<CategorySummary>(response.data);
  } catch {
    return [] as CategorySummary[];
  }
}

export default async function Home() {
  const localeHeaders = await getServerLocaleHeaders();
  const [
    homepageData,
    heroBanners,
    siteSettings,
  ] = await Promise.all([
    getHomepageData(localeHeaders),
    getBanners(localeHeaders, "home_hero"),
    getSiteSettings(localeHeaders),
  ]);

  const featuredProducts = asArray<ProductListItem>(homepageData.featured_products);
  const newArrivals = asArray<ProductListItem>(homepageData.new_arrivals);
  const bestsellers = asArray<ProductListItem>(homepageData.bestsellers);
  const onSale = asArray<ProductListItem>(homepageData.on_sale);
  const featuredCategories = asArray<FeaturedCategory>(homepageData.featured_categories);
  const featuredCategoriesWithProducts = featuredCategories.filter((category) => {
    if (category.product_count === null || category.product_count === undefined) return true;
    return Number(category.product_count) > 0;
  });
  const homepageCategories = featuredCategoriesWithProducts.slice(0, 3);
  const categoryProducts = await Promise.all(
    homepageCategories.map((category) => getCategoryProducts(localeHeaders, category.slug))
  );
  const categoryBands = homepageCategories.map((category, index) => ({
    category,
    products: categoryProducts[index] || [],
  }));
  const categoryBandsWithProducts = categoryBands.filter(
    (band) => band.products.length > 0
  );
  const showByCategories = await getShowByCategories(localeHeaders);
  const topCategoryChips = categoryBandsWithProducts.map((band) => band.category);
  const collections = asArray<Collection>(homepageData.collections);
  const brandName = pickText(siteSettings?.site_name);
  const heroDescription = pickText(
    siteSettings?.site_tagline,
    siteSettings?.tagline,
    siteSettings?.site_description
  );

  const seasonalFavs = (onSale.length ? onSale : featuredProducts).slice(0, 8);

  const homePageSchema = cleanObject({
    "@context": "https://schema.org",
    "@type": "WebPage",
    name: brandName,
    description: heroDescription,
    url: absoluteUrl("/"),
    isPartOf: {
      "@id": absoluteUrl("/#website"),
    },
  });

  const featuredList = buildItemList(
    featuredProducts.slice(0, 10).map((product) => ({
      name: product.name,
      url: buildProductPath(product),
      image: getImage(product) || undefined,
      description: product.short_description || undefined,
    })),
    "Featured products"
  );

  const collectionsList = buildItemList(
    collections.slice(0, 10).map((collection) => ({
      name: collection.name,
      url: `/collections/${collection.slug}/`,
      image: collection.image || undefined,
      description: collection.description || undefined,
    })),
    "Collections"
  );

  const jsonLd = [
    homePageSchema,
    ...(featuredProducts.length ? [featuredList] : []),
    ...(collections.length ? [collectionsList] : []),
  ];

  const sectionWrapperClass = "mx-auto w-full max-w-7xl px-4 sm:px-6";

  return (
    <div className="bg-background text-foreground">
      <section className="border-b border-border/70">
        <div className={`${sectionWrapperClass} py-6`}>
          {heroBanners.length ? (
            <HeroBannerSlider banners={heroBanners} className="mx-auto" />
          ) : (
            <div className="aspect-[16/7] w-full bg-muted" />
          )}
          {topCategoryChips.length ? (
            <div className="mt-6 flex flex-wrap gap-3 text-xs font-semibold uppercase tracking-[0.2em] text-foreground/70">
              {topCategoryChips.map((category) => (
                <Link
                  key={category.id}
                  href={buildCategoryPath(category.slug)}
                  className="hover:text-foreground"
                >
                  {category.name}
                </Link>
              ))}
              <Link href="/products/" className="hover:text-foreground">
                View All
              </Link>
            </div>
          ) : null}
        </div>
      </section>

      {categoryBandsWithProducts.map((band) => (
        <section key={band.category.id} className={`${sectionWrapperClass} py-8`}>
          <div className="flex items-center justify-between gap-4">
            <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
              {band.category.name}
            </h2>
            <Link
              href={buildCategoryPath(band.category.slug)}
              className="text-xs font-semibold uppercase tracking-[0.2em] text-foreground/60 hover:text-foreground"
            >
              View All
            </Link>
          </div>
          <div className="mt-4">
            <ProductGrid products={band.products} cardStyle="minimal" />
          </div>
        </section>
      ))}

      {seasonalFavs.length ? (
        <section className={`${sectionWrapperClass} py-8`}>
          <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
            Seasonal Favs
          </h2>
          <div className="mt-4">
            <ProductGrid products={seasonalFavs} cardStyle="minimal" />
          </div>
        </section>
      ) : null}

      <section className={`${sectionWrapperClass} py-8`}>
        <HomeProductTabs newDrops={newArrivals} trending={bestsellers} />
      </section>

      <section className={`${sectionWrapperClass} py-8`}>
        <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
          Recently Stalked
        </h2>
        <div className="mt-4">
          <RecentlyViewedSection />
        </div>
      </section>

      {showByCategories.length ? (
        <section className={`${sectionWrapperClass} py-8`}>
          <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
            Show By Category
          </h2>
          <div className="mt-4 grid gap-4 sm:grid-cols-3">
            {showByCategories.map((category) => (
              <Link
                key={category.id}
                href={buildCategoryPath(category.slug)}
                className="group"
              >
                <div className="aspect-[4/3] overflow-hidden bg-muted">
                  {category.image ? (
                    <img
                      src={category.image}
                      alt={category.name}
                      className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                    />
                  ) : null}
                </div>
                <p className="mt-2 text-sm font-semibold uppercase tracking-[0.18em] text-foreground/70">
                  {category.name}
                </p>
              </Link>
            ))}
          </div>
        </section>
      ) : null}

      <JsonLd data={jsonLd} />
    </div>
  );
}
