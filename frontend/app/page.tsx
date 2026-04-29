import type { Metadata } from "next";
import dynamic from "next/dynamic";
import Link from "next/link";
import Image from "next/image";
import { apiFetch } from "@/lib/api";
import type {
  Collection,
  ProductListItem,
} from "@/lib/types";
import { asArray } from "@/lib/array";
import { JsonLd } from "@/components/seo/JsonLd";
import { absoluteUrl, buildItemList, buildPageMetadata, cleanObject } from "@/lib/seo";
import { buildCategoryPath } from "@/lib/categoryPaths";
import { buildProductPath } from "@/lib/productPaths";
import { getSiteSettings } from "@/lib/siteSettings";

const ProductGrid = dynamic(
  () => import("@/components/products/ProductGrid").then((mod) => mod.ProductGrid)
);
const RecentlyViewedSection = dynamic(
  () => import("@/components/products/RecentlyViewedSection").then((mod) => mod.RecentlyViewedSection)
);
const HomeProductTabs = dynamic(
  () => import("@/components/products/HomeProductTabs").then((mod) => mod.HomeProductTabs)
);
const HeroBannerSlider = dynamic(
  () => import("@/components/promotions/HeroBannerSlider").then((mod) => mod.HeroBannerSlider)
);
import type { HeroBanner } from "@/components/promotions/HeroBannerSlider";

export const revalidate = 300;
export const metadata: Metadata = buildPageMetadata({
  title: "Curated Products and Artisan Collections",
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
  is_featured?: boolean | null;
};

type Spotlight = {
  id: string;
  name?: string;
  placement?: string;
  product?: ProductListItem | null;
  category?: FeaturedCategory | null;
};

type HomepageData = {
  featured_products: ProductListItem[];
  new_arrivals: ProductListItem[];
  bestsellers: ProductListItem[];
  on_sale: ProductListItem[];
  featured_categories: FeaturedCategory[];
  collections: Collection[];
  spotlights?: Spotlight[];
  show_by_categories?: FeaturedCategory[];
};

type Banner = HeroBanner & {
  position?: string | null;
};

const DEFAULT_HOMEPAGE_DATA: HomepageData = {
  featured_products: [],
  new_arrivals: [],
  bestsellers: [],
  on_sale: [],
  featured_categories: [],
  collections: [],
  spotlights: [],
  show_by_categories: [],
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


async function getHomepageData() {
  try {
    const response = await apiFetch<HomepageData>("/catalog/homepage/", {
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
      show_by_categories: asArray<FeaturedCategory>((payload as HomepageData).show_by_categories),
    };
  } catch (error) {
    console.error("Failed to fetch homepage data:", error instanceof Error ? error.message : error);
    return DEFAULT_HOMEPAGE_DATA;
  }
}

async function getBanners(position?: string) {
  try {
    const response = await apiFetch<Banner[]>("/promotions/banners/", {
      params: position ? { position } : undefined,
      next: { revalidate },
    });
    return asArray<Banner>(response.data);
  } catch (error) {
    console.error(`Failed to fetch banners for position ${position}:`, error instanceof Error ? error.message : error);
    return [] as Banner[];
  }
}

async function getCategoryProducts(slug: string) {
  try {
    const response = await apiFetch<
      ProductListItem[] | { results?: ProductListItem[] }
    >("/catalog/products/by-category/", {
      params: {
        category: slug,
        page_size: 8,
        include_descendants: true,
        primary_only: true,
      },
      next: { revalidate },
    });
    const payload = response.data as ProductListItem[] | { results?: ProductListItem[] };
    if (Array.isArray(payload)) return payload;
    return asArray<ProductListItem>(payload.results);
  } catch {
    return [] as ProductListItem[];
  }
}

export default async function Home() {
  const [
    homepageData,
    heroBanners,
    siteSettings,
  ] = await Promise.all([
    getHomepageData(),
    getBanners("home_hero"),
    getSiteSettings(),
  ]);

  const featuredProducts = asArray<ProductListItem>(homepageData.featured_products);
  const newArrivals = asArray<ProductListItem>(homepageData.new_arrivals);
  const bestsellers = asArray<ProductListItem>(homepageData.bestsellers);
  const onSale = asArray<ProductListItem>(homepageData.on_sale);
  const featuredCategories = asArray<FeaturedCategory>(homepageData.featured_categories);
  const spotlights = asArray<Spotlight>(homepageData.spotlights);
  const showByCategoriesRaw = asArray<FeaturedCategory>(homepageData.show_by_categories);
  const featuredCategorySlugs = new Set(
    featuredCategories.map((category) => category.slug)
  );
  const featuredCategoryIds = new Set(
    featuredCategories.map((category) => category.id)
  );
  const resolveHomepageFeaturedCategoryId = (product: ProductListItem) => {
    if (!product.primary_category_id) return null;
    if (featuredCategoryIds.has(product.primary_category_id)) {
      return product.primary_category_id;
    }
    const rawPath = product.primary_category_path || "";
    const pathIds = rawPath.split("/").map((part) => part.trim()).filter(Boolean);
    if (!pathIds.length) return null;
    for (let index = pathIds.length - 2; index >= 0; index -= 1) {
      const ancestorId = pathIds[index];
      if (featuredCategoryIds.has(ancestorId)) {
        return ancestorId;
      }
    }
    return null;
  };
  const filterFeaturedScopeProducts = (products: ProductListItem[]) =>
    products.filter((product) => Boolean(resolveHomepageFeaturedCategoryId(product)));
  const filteredFeaturedProducts = filterFeaturedScopeProducts(featuredProducts);
  const filteredNewArrivals = filterFeaturedScopeProducts(newArrivals);
  const filteredBestsellers = filterFeaturedScopeProducts(bestsellers);
  const filteredOnSale = filterFeaturedScopeProducts(onSale);
  const showByCategories = showByCategoriesRaw.filter(
    (category) => Boolean(category.is_featured) || featuredCategorySlugs.has(category.slug)
  );
  const homepageCategories = featuredCategories.slice(0, 3);
  const categoryProducts = await Promise.all(
    homepageCategories.map((category) => getCategoryProducts(category.slug))
  );
  const categoryBands = homepageCategories.map((category, index) => ({
    category,
    products: categoryProducts[index] || [],
  }));
  const seenHomepageProductIds = new Set<string>();
  const categoryBandsWithProducts = categoryBands
    .map((band) => ({
      ...band,
      products: band.products.filter((product) => {
        if (!product?.id) return false;
        if (resolveHomepageFeaturedCategoryId(product) !== band.category.id) return false;
        if (seenHomepageProductIds.has(product.id)) return false;
        seenHomepageProductIds.add(product.id);
        return true;
      }),
    }))
    .filter((band) => band.products.length > 0);
  const collections = asArray<Collection>(homepageData.collections);
  const brandName = pickText(siteSettings?.site_name);
  const heroDescription = pickText(
    siteSettings?.site_tagline,
    siteSettings?.tagline,
    siteSettings?.site_description
  );

  const seasonalFavs = filteredOnSale.slice(0, 8);

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
    filteredFeaturedProducts.slice(0, 10).map((product) => ({
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
    ...(filteredFeaturedProducts.length ? [featuredList] : []),
    ...(collections.length ? [collectionsList] : []),
  ];

  const sectionWrapperClass = "mx-auto w-full max-w-7xl px-3 sm:px-5";

  return (
    <div className="bg-background text-foreground">
      <section>
        <div className={`${sectionWrapperClass} pb-6`}>
          {heroBanners.length ? (
            <HeroBannerSlider banners={heroBanners} className="mx-auto" autoAdvance={false} />
          ) : (
            <div className="aspect-[16/7] w-full bg-muted" />
          )}
        </div>
      </section>

      {spotlights.length ? (
        <section className={`${sectionWrapperClass} py-8`}>
          <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
            Spotlights
          </h2>
          <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {spotlights.map((spotlight: Spotlight) => {
              const spotlightProduct = spotlight.product || null;
              const spotlightCategory = spotlight.category || null;
              const spotlightTitle =
                spotlight.name ||
                spotlightProduct?.name ||
                spotlightCategory?.name ||
                "Spotlight";
              const spotlightSubtitle =
                spotlightProduct?.short_description ||
                (spotlightCategory?.product_count
                  ? `${spotlightCategory.product_count} products`
                  : "Curated highlight");
              const spotlightImage =
                getImage(spotlightProduct) ||
                spotlightCategory?.image ||
                null;
              const href = spotlightProduct
                ? buildProductPath(spotlightProduct)
                : spotlightCategory
                ? buildCategoryPath(spotlightCategory.slug)
                : "/products/";
              const isProductSpotlight = Boolean(spotlightProduct);

              return (
                <Link
                  key={spotlight.id}
                  href={href}
                  prefetch={false}
                  className="group overflow-hidden rounded-2xl border border-border bg-card"
                  target={isProductSpotlight ? "_blank" : undefined}
                  rel={isProductSpotlight ? "noopener noreferrer" : undefined}
                >
                  <div className="relative aspect-[16/10] overflow-hidden bg-muted">
                    {spotlightImage ? (
                      <Image
                        src={spotlightImage}
                        alt={spotlightTitle}
                        fill
                        sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                        quality={70}
                        loading="lazy"
                        decoding="async"
                        className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                      />
                    ) : null}
                  </div>
                  <div className="space-y-1 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                      {spotlight.placement || "home"}
                    </p>
                    <p className="line-clamp-1 text-lg font-semibold">{spotlightTitle}</p>
                    <p className="line-clamp-2 text-sm text-foreground/70">
                      {spotlightSubtitle || "Explore this featured recommendation."}
                    </p>
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      ) : null}

      {categoryBandsWithProducts.map((band) => (
        <section key={band.category.id} className={`${sectionWrapperClass} py-8`}>
          <div className="flex items-center justify-between gap-4">
            <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
              {band.category.name}
            </h2>
            <Link
              href={buildCategoryPath(band.category.slug)}
              prefetch={false}
              className="text-xs font-semibold uppercase tracking-[0.2em] text-foreground/60 hover:text-foreground"
            >
              View All
            </Link>
          </div>
          <div className="mt-4">
            <ProductGrid
              products={band.products}
              cardStyle="minimal"
              allowQuickView={true}
              showWishlist={true}
            />
          </div>
        </section>
      ))}

      {seasonalFavs.length ? (
        <section className={`${sectionWrapperClass} py-8`}>
          <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
            Seasonal Favs
          </h2>
          <div className="mt-4">
            <ProductGrid
              products={seasonalFavs}
              cardStyle="minimal"
              allowQuickView={true}
              showWishlist={true}
            />
          </div>
        </section>
      ) : null}

      <section className={`${sectionWrapperClass} py-8`}>
        <HomeProductTabs
          newDrops={filteredNewArrivals}
          trending={filteredBestsellers}
          allowQuickView={true}
          showWishlist={true}
        />
      </section>

      <section className={`${sectionWrapperClass} py-8`}>
        <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/70">
          Recently viewed
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
                prefetch={false}
                className="group"
              >
                <div className="relative aspect-[4/3] overflow-hidden bg-muted">
                  {category.image ? (
                    <Image
                      src={category.image}
                      alt={category.name}
                      fill
                      sizes="(max-width: 640px) 100vw, 33vw"
                      quality={68}
                      loading="lazy"
                      decoding="async"
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
