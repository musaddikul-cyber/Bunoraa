/* eslint-disable @next/next/no-img-element */
import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type {
  Collection,
  ContactSettings,
  PreorderCategory,
  ProductListItem,
  SiteSettings,
  SubscriptionPlan,
  Bundle,
} from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { ProductGrid } from "@/components/products/ProductGrid";
import { SearchBar } from "@/components/search/SearchBar";
import { getServerLocaleHeaders } from "@/lib/serverLocale";
import { asArray } from "@/lib/array";
import { JsonLd } from "@/components/seo/JsonLd";
import { SITE_URL, absoluteUrl, buildItemList, cleanObject } from "@/lib/seo";

export const revalidate = 300;

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

type FaqItem = {
  id: string;
  question: string;
  answer: string;
  category?: string | null;
};

type BundleSummary = Bundle & {
  price?: string | null;
  item_count?: number | null;
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

const formatNumber = (value?: number | string | null) => {
  if (value === null || value === undefined) return "0";
  const numeric = typeof value === "string" ? Number(value) : value;
  if (Number.isNaN(numeric)) return String(value);
  return new Intl.NumberFormat("en-US").format(numeric);
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

const getPrice = (product: ProductListItem) => {
  return product.current_price || product.sale_price || product.price || "";
};

const getCurrency = (product: ProductListItem) => {
  return product.currency || (product as unknown as { currency_code?: string }).currency_code || "";
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

async function getContactSettings(headers: Record<string, string>) {
  try {
    const response = await apiFetch<ContactSettings>("/contacts/settings/", {
      headers,
      next: { revalidate },
    });
    return response.data;
  } catch {
    return null;
  }
}

async function getPreorderCategories(headers: Record<string, string>) {
  try {
    const response = await apiFetch<PreorderCategory[]>("/preorders/categories/", {
      headers,
      params: { page_size: 4 },
      next: { revalidate },
    });
    return asArray<PreorderCategory>(response.data);
  } catch {
    return [] as PreorderCategory[];
  }
}

async function getSubscriptionPlans(headers: Record<string, string>) {
  try {
    const response = await apiFetch<SubscriptionPlan[]>("/subscriptions/plans/", {
      headers,
      next: { revalidate },
    });
    return asArray<SubscriptionPlan>(response.data);
  } catch {
    return [] as SubscriptionPlan[];
  }
}

async function getFaqs(headers: Record<string, string>) {
  try {
    const response = await apiFetch<FaqItem[]>("/pages/faqs/", {
      headers,
      next: { revalidate },
    });
    return asArray<FaqItem>(response.data);
  } catch {
    return [] as FaqItem[];
  }
}

async function getBundles(headers: Record<string, string>) {
  try {
    const response = await apiFetch<BundleSummary[]>("/catalog/bundles/", {
      headers,
      params: { page_size: 4 },
      next: { revalidate },
    });
    return asArray<BundleSummary>(response.data);
  } catch {
    return [] as BundleSummary[];
  }
}

export default async function Home() {
  const localeHeaders = await getServerLocaleHeaders();
  const [
    homepageData,
    siteSettings,
    contactSettings,
    preorderCategories,
    subscriptionPlans,
    faqs,
    bundles,
  ] = await Promise.all([
    getHomepageData(localeHeaders),
    getSiteSettings(localeHeaders),
    getContactSettings(localeHeaders),
    getPreorderCategories(localeHeaders),
    getSubscriptionPlans(localeHeaders),
    getFaqs(localeHeaders),
    getBundles(localeHeaders),
  ]);

  const brandName = pickText(siteSettings?.site_name) || "Bunoraa";
  const heroTitle =
    pickText(siteSettings?.tagline, siteSettings?.site_tagline) ||
    "Handcrafted collections for modern living";
  const heroDescription =
    pickText(siteSettings?.site_description) ||
    "Discover curated craftsmanship, artisan bundles, and bespoke orders delivered with clarity and care.";

  const supportEmail = pickText(
    contactSettings?.support_email,
    siteSettings?.support_email,
    siteSettings?.contact_email,
    contactSettings?.general_email
  );
  const supportPhone = pickText(contactSettings?.phone, siteSettings?.contact_phone);
  const supportHours = pickText(contactSettings?.business_hours_note);

  const featuredProducts = asArray<ProductListItem>(homepageData.featured_products);
  const newArrivals = asArray<ProductListItem>(homepageData.new_arrivals);
  const bestsellers = asArray<ProductListItem>(homepageData.bestsellers);
  const onSale = asArray<ProductListItem>(homepageData.on_sale);
  const featuredCategories = asArray<FeaturedCategory>(homepageData.featured_categories);
  const collections = asArray<Collection>(homepageData.collections);

  const heroStats = [
    {
      label: "Featured products",
      value: formatNumber(featuredProducts.length),
      description: "Handpicked right now",
      href: "/products/",
    },
    {
      label: "New arrivals",
      value: formatNumber(newArrivals.length),
      description: "Fresh inventory drops",
      href: "/products/?ordering=-created_at",
    },
    {
      label: "Collections",
      value: formatNumber(collections.length),
      description: "Curated themes",
      href: "/collections/",
    },
    {
      label: "On sale",
      value: formatNumber(onSale.length),
      description: "Limited offers",
      href: "/products/?ordering=on_sale",
    },
  ];

  const productSections = [
    {
      id: "featured",
      title: "Featured picks",
      description: "Best-in-class selections across the catalog.",
      href: "/products/",
      products: featuredProducts.slice(0, 6),
    },
    {
      id: "new",
      title: "New arrivals",
      description: "Recently released pieces from our artisan partners.",
      href: "/products/?ordering=-created_at",
      products: newArrivals.slice(0, 6),
    },
  ];

  const compactShowcase = [
    {
      title: "On sale",
      href: "/products/?ordering=on_sale",
      items: onSale.slice(0, 4),
    },
    {
      title: "Bestsellers",
      href: "/products/?ordering=-sales_count",
      items: bestsellers.slice(0, 4),
    },
  ];

  const homePageSchema = cleanObject({
    "@context": "https://schema.org",
    "@type": "WebPage",
    name: brandName,
    description: heroDescription,
    url: absoluteUrl("/"),
    isPartOf: {
      "@type": "WebSite",
      name: "Bunoraa",
      url: SITE_URL,
    },
  });

  const featuredList = buildItemList(
    featuredProducts.slice(0, 10).map((product) => ({
      name: product.name,
      url: `/products/${product.slug}/`,
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

  return (
    <div className="bg-background text-foreground">
      <section className="relative overflow-hidden motion-safe:animate-fade-in">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-24 right-0 h-72 w-72 rounded-full bg-primary/10 blur-3xl"
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute bottom-0 left-0 h-72 w-72 rounded-full bg-accent/15 blur-3xl"
        />
        <div className="mx-auto w-full max-w-7xl px-6 py-16 lg:py-24">
          <div className="grid gap-10 lg:grid-cols-[1.15fr_0.85fr]">
            <div className="space-y-8">
              <div className="space-y-4">
                <p className="text-sm uppercase tracking-[0.2em] text-foreground/50">
                  {brandName}
                </p>
                <h1 className="text-4xl font-semibold sm:text-5xl lg:text-6xl">
                  <span className="text-gradient text-gradient-primary font-display">
                    {heroTitle}
                  </span>
                </h1>
                <p className="max-w-2xl text-lg text-foreground/60">
                  {heroDescription}
                </p>
              </div>
              <div className="flex flex-wrap gap-3">
                <Button asChild variant="primary">
                  <Link href="/products/">Shop products</Link>
                </Button>
                <Button asChild variant="secondary">
                  <Link href="/preorders/">Start a preorder</Link>
                </Button>
                <Button asChild variant="ghost">
                  <Link href="/collections/">Browse collections</Link>
                </Button>
              </div>
              <Card variant="glass" className="flex flex-col gap-4 p-4">
                <div>
                  <p className="text-sm font-semibold">Find your next piece</p>
                  <p className="text-xs text-foreground/55">
                    Search across products, categories, and artisan bundles.
                  </p>
                </div>
                <SearchBar />
              </Card>
              <div className="flex flex-wrap gap-2 text-xs text-foreground/60">
                {featuredCategories.slice(0, 5).map((category) => (
                  <Link
                    key={category.id}
                    href={`/categories/${category.slug}/`}
                    className="rounded-full border border-border bg-card px-3 py-1 transition hover:border-primary/50 hover:text-foreground"
                  >
                    {category.name}
                  </Link>
                ))}
                <Link
                  href="/categories/"
                  className="rounded-full border border-border bg-card px-3 py-1 transition hover:border-primary/50 hover:text-foreground"
                >
                  All categories
                </Link>
              </div>
            </div>

            <div className="grid gap-5">
              <Card variant="glass" className="space-y-6 p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-foreground/50">
                      Marketplace pulse
                    </p>
                    <h2 className="text-xl font-semibold">Live commerce overview</h2>
                  </div>
                  <Button asChild variant="ghost" size="sm">
                    <Link href="/products/">View catalog</Link>
                  </Button>
                </div>
                <dl className="grid gap-4 sm:grid-cols-2">
                  {heroStats.map((stat) => (
                    <div key={stat.label} className="rounded-xl border border-border/60 bg-card/70 p-4">
                      <dt className="text-xs uppercase tracking-[0.2em] text-foreground/50">
                        {stat.label}
                      </dt>
                      <dd className="mt-2 text-2xl font-semibold">{stat.value}</dd>
                      <p className="mt-1 text-xs text-foreground/55">{stat.description}</p>
                      <Link href={stat.href} className="mt-2 inline-flex text-xs font-semibold text-primary">
                        Explore
                      </Link>
                    </div>
                  ))}
                </dl>
                {(supportEmail || supportPhone) && (
                  <div className="rounded-xl border border-border/60 bg-card/70 p-4 text-sm">
                    <p className="font-semibold">Support concierge</p>
                    <div className="mt-2 space-y-1 text-foreground/60">
                      {supportEmail ? (
                        <p>
                          Email: <Link href={`mailto:${supportEmail}`}>{supportEmail}</Link>
                        </p>
                      ) : null}
                      {supportPhone ? (
                        <p>
                          Phone: <Link href={`tel:${supportPhone}`}>{supportPhone}</Link>
                        </p>
                      ) : null}
                      {supportHours ? <p>{supportHours}</p> : null}
                    </div>
                  </div>
                )}
              </Card>

            </div>
          </div>
        </div>
      </section>
      <section className="mx-auto w-full max-w-7xl space-y-12 px-6 py-12" id="highlights">
        {productSections.map((section) => (
          <div key={section.id} className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.2em] text-foreground/50">
                  {section.title}
                </p>
                <p className="mt-2 text-sm text-foreground/60">{section.description}</p>
              </div>
              <Button asChild variant="ghost">
                <Link href={section.href}>View all</Link>
              </Button>
            </div>
            <ProductGrid products={section.products} />
          </div>
        ))}
      </section>
      <section className="mx-auto w-full max-w-7xl px-6 py-12" id="categories">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/50">Categories</p>
            <h2 className="text-2xl font-semibold">Shop by category</h2>
          </div>
          <Button asChild variant="ghost">
            <Link href="/categories/">Browse all</Link>
          </Button>
        </div>
        <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {featuredCategories.length ? (
            featuredCategories.map((category) => (
              <Card key={category.id} variant="bordered" className="flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <p className="text-lg font-semibold">{category.name}</p>
                  {category.product_count ? (
                    <span className="text-xs text-foreground/55">
                      {formatNumber(category.product_count)} items
                    </span>
                  ) : null}
                </div>
                <p className="text-sm text-foreground/60">
                  Curated pieces selected for quality and delivery readiness.
                </p>
                <Button asChild size="sm" variant="secondary">
                  <Link href={`/categories/${category.slug}/`}>Shop category</Link>
                </Button>
              </Card>
            ))
          ) : (
            <Card variant="bordered" className="text-sm text-foreground/60">
              Categories will appear here once they are published in the catalog.
            </Card>
          )}
        </div>
      </section>
      <section className="mx-auto w-full max-w-7xl px-6 py-12" id="marketplace">
        <div className="grid gap-6 lg:grid-cols-2">
          {compactShowcase.map((block) => (
            <Card key={block.title} variant="bordered" className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">{block.title}</h3>
                <Button asChild size="sm" variant="ghost">
                  <Link href={block.href}>View all</Link>
                </Button>
              </div>
              <div className="space-y-3">
                {block.items.map((product) => (
                  <div key={product.id} className="flex items-center gap-4">
                    <div className="h-14 w-14 overflow-hidden rounded-xl bg-muted">
                      {getImage(product) ? (
                        <img
                          src={getImage(product) || ""}
                          alt={product.name}
                          className="h-full w-full object-cover"
                        />
                      ) : null}
                    </div>
                    <div className="flex-1">
                      <Link href={`/products/${product.slug}/`} className="text-sm font-semibold">
                        {product.name}
                      </Link>
                      <p className="text-xs text-foreground/55">
                        {getPrice(product)} {getCurrency(product)}
                      </p>
                    </div>
                    <Button asChild size="sm" variant="secondary">
                      <Link href={`/products/${product.slug}/`}>View</Link>
                    </Button>
                  </div>
                ))}
                {!block.items.length ? (
                  <p className="text-sm text-foreground/55">No items available right now.</p>
                ) : null}
              </div>
            </Card>
          ))}
        </div>
      </section>
      <section className="mx-auto w-full max-w-7xl px-6 py-12" id="collections">
        <div className="grid gap-6 lg:grid-cols-2">
          <Card variant="bordered" className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm uppercase tracking-[0.2em] text-foreground/50">Collections</p>
                <h2 className="text-2xl font-semibold">Curated themes</h2>
              </div>
              <Button asChild size="sm" variant="ghost">
                <Link href="/collections/">All collections</Link>
              </Button>
            </div>
            <div className="space-y-3">
              {collections.slice(0, 4).map((collection) => (
                <Link
                  key={collection.id}
                  href={`/collections/${collection.slug}/`}
                  className="flex items-center justify-between rounded-xl border border-border bg-background/80 px-4 py-3 text-sm transition hover:border-primary/40"
                >
                  <span className="font-semibold">{collection.name}</span>
                  <span className="text-xs text-foreground/55">Explore</span>
                </Link>
              ))}
              {!collections.length ? (
                <p className="text-sm text-foreground/55">Collections are loading soon.</p>
              ) : null}
            </div>
          </Card>

          <Card variant="bordered" className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm uppercase tracking-[0.2em] text-foreground/50">Bundles</p>
                <h2 className="text-2xl font-semibold">Ready-to-ship sets</h2>
              </div>
              <Button asChild size="sm" variant="ghost">
                <Link href="/bundles/">Browse bundles</Link>
              </Button>
            </div>
            <div className="space-y-3">
              {bundles.slice(0, 4).map((bundle) => (
                <Link
                  key={bundle.id}
                  href={`/bundles/${bundle.slug}/`}
                  className="flex items-center justify-between rounded-xl border border-border bg-background/80 px-4 py-3 text-sm transition hover:border-primary/40"
                >
                  <div>
                    <p className="font-semibold">{bundle.name}</p>
                    {bundle.item_count ? (
                      <p className="text-xs text-foreground/55">{bundle.item_count} items</p>
                    ) : null}
                  </div>
                  <span className="text-xs text-foreground/55">
                    {bundle.price ? `${bundle.price}` : "Details"}
                  </span>
                </Link>
              ))}
              {!bundles.length ? (
                <p className="text-sm text-foreground/55">Bundles are on the way.</p>
              ) : null}
            </div>
          </Card>
        </div>
      </section>
      <section className="mx-auto w-full max-w-7xl px-6 py-12" id="preorders">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/50">Preorders</p>
            <h2 className="text-2xl font-semibold">Made-to-order programs</h2>
            <p className="mt-2 text-sm text-foreground/60">
              Launch custom production runs with transparent timelines.
            </p>
          </div>
          <Button asChild variant="ghost">
            <Link href="/preorders/">Manage preorders</Link>
          </Button>
        </div>
        <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {preorderCategories.slice(0, 4).map((category) => (
            <Card key={category.id} variant="bordered" className="space-y-3">
              <p className="text-sm uppercase tracking-[0.2em] text-foreground/50">
                {category.name}
              </p>
              <p className="text-sm text-foreground/60">
                {category.description || "Configure a custom run with artisan oversight."}
              </p>
              <div className="text-xs text-foreground/55">
                {category.base_price ? `Starting at ${category.base_price}` : "Pricing by quote"}
              </div>
              <Button asChild size="sm" variant="secondary">
                <Link href={`/preorders/category/${category.slug}/`}>Explore</Link>
              </Button>
            </Card>
          ))}
          {!preorderCategories.length ? (
            <Card variant="bordered" className="text-sm text-foreground/60">
              Preorder categories are not available yet. Configure them in the admin
              to activate this section.
            </Card>
          ) : null}
        </div>
      </section>
      {subscriptionPlans.length ? (
        <section className="mx-auto w-full max-w-7xl px-6 py-12" id="subscriptions">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-sm uppercase tracking-[0.2em] text-foreground/50">Subscriptions</p>
              <h2 className="text-2xl font-semibold">Recurring delivery plans</h2>
            </div>
            <Button asChild variant="ghost">
              <Link href="/subscriptions/">See all plans</Link>
            </Button>
          </div>
          <div className="mt-6 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {subscriptionPlans.slice(0, 3).map((plan) => (
              <Card key={plan.id} variant="bordered" className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold">{plan.name}</h3>
                  <p className="text-sm text-foreground/60">{plan.description}</p>
                </div>
                <p className="text-base font-semibold">
                  {plan.price_amount} {plan.currency} / {plan.interval}
                </p>
                <Button asChild size="sm" variant="secondary">
                  <Link href={`/subscriptions/plans/${plan.id}/`}>View plan</Link>
                </Button>
              </Card>
            ))}
          </div>
        </section>
      ) : null}
      {faqs.length ? (
        <section className="mx-auto w-full max-w-7xl px-6 py-12" id="faq">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-sm uppercase tracking-[0.2em] text-foreground/50">FAQ</p>
              <h2 className="text-2xl font-semibold">Answers at a glance</h2>
            </div>
            <Button asChild variant="ghost">
              <Link href="/faq/">Visit FAQ</Link>
            </Button>
          </div>
          <div className="mt-6 grid gap-6 md:grid-cols-3">
            {faqs.slice(0, 3).map((faq) => (
              <Card key={faq.id} variant="bordered" className="space-y-2">
                <h3 className="text-lg font-semibold">{faq.question}</h3>
                <p className="text-sm text-foreground/60">{faq.answer}</p>
              </Card>
            ))}
          </div>
        </section>
      ) : null}
      <section className="mx-auto w-full max-w-7xl px-6 pb-16" id="cta">
        <Card variant="modern-gradient" className="flex flex-col gap-6 p-8 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/50">Ready to start</p>
            <h2 className="text-2xl font-semibold">Build your next collection with Bunoraa</h2>
            <p className="mt-2 text-sm text-foreground/60">
              Launch curated catalogs, manage preorders, and delight customers with a polished experience.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button asChild variant="primary">
              <Link href="/products/">Shop now</Link>
            </Button>
            <Button asChild variant="secondary">
              <Link href="/contact/">Talk to us</Link>
            </Button>
          </div>
        </Card>
      </section>
      <JsonLd data={jsonLd} />
    </div>
  );
}
