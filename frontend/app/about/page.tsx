import type { Metadata } from "next";
import Link from "next/link";
import { cache } from "react";
import { apiFetch, ApiError } from "@/lib/api";
import type {
  Collection,
  ContactSettings,
  PageDetail,
  SiteSettings,
} from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { JsonLd } from "@/components/seo/JsonLd";
import { absoluteUrl, buildPageMetadata, cleanObject } from "@/lib/seo";
import { asArray } from "@/lib/array";
import { buildCategoryPath } from "@/lib/categoryPaths";

export const dynamic = "force-dynamic";
export const revalidate = 900;

type CategoryPreview = {
  id: string;
  name: string;
  slug: string;
  product_count?: number | null;
};

type ListPayload<T> = T[] | { results?: T[]; count?: number };

const SOCIAL_SITE_FIELDS: Array<{ key: keyof SiteSettings; label: string }> = [
  { key: "facebook_url", label: "Facebook" },
  { key: "instagram_url", label: "Instagram" },
  { key: "twitter_url", label: "Twitter" },
  { key: "linkedin_url", label: "LinkedIn" },
  { key: "youtube_url", label: "YouTube" },
  { key: "tiktok_url", label: "TikTok" },
];

type QueryParams = Record<
  string,
  string | number | boolean | Array<string | number | boolean> | undefined
>;

const pickText = (...values: Array<string | null | undefined>) => {
  for (const value of values) {
    if (value && value.trim()) return value.trim();
  }
  return "";
};

const stripHtml = (value?: string | null) =>
  String(value || "")
    .replace(/<[^>]*>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const formatCount = (value: number) => new Intl.NumberFormat("en-US").format(value);

const formatDate = (value?: string | null) => {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return new Intl.DateTimeFormat("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  }).format(parsed);
};

const normalizeExternalHref = (value?: string | null) => {
  const trimmed = String(value || "").trim();
  if (!trimmed) return "";
  if (/^(https?:|mailto:|tel:)/i.test(trimmed)) return trimmed;
  return `https://${trimmed.replace(/^\/+/, "")}`;
};

const resolveCount = (payload: unknown, metaCount?: number) => {
  if (typeof metaCount === "number") return metaCount;
  if (
    payload &&
    typeof payload === "object" &&
    !Array.isArray(payload) &&
    typeof (payload as { count?: unknown }).count === "number"
  ) {
    return (payload as { count: number }).count;
  }
  return asArray<unknown>(payload).length;
};

const getAboutPage = cache(async (): Promise<PageDetail | null> => {
  try {
    const response = await apiFetch<PageDetail>("/pages/about/", {
      next: { revalidate },
    });
    return response.data;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null;
    }
    return null;
  }
});

const getSiteSettings = cache(async (): Promise<SiteSettings | null> => {
  try {
    const response = await apiFetch<SiteSettings>("/pages/settings/", {
      next: { revalidate },
    });
    return response.data;
  } catch {
    return null;
  }
});

const getContactSettings = cache(async (): Promise<ContactSettings | null> => {
  try {
    const response = await apiFetch<ContactSettings>("/contacts/settings/", {
      next: { revalidate },
    });
    return response.data;
  } catch {
    return null;
  }
});

const getCategorySnapshot = cache(async () => {
  try {
    const response = await apiFetch<ListPayload<CategoryPreview>>("/catalog/categories/", {
      params: { has_products: true, page_size: 6 },
      next: { revalidate },
    });
    const items = asArray<CategoryPreview>(response.data).slice(0, 6);
    return {
      items,
      count: resolveCount(response.data, response.meta?.pagination?.count),
    };
  } catch {
    return { items: [] as CategoryPreview[], count: 0 };
  }
});

const getCollectionSnapshot = cache(async () => {
  try {
    const response = await apiFetch<ListPayload<Collection>>("/catalog/collections/", {
      params: { page_size: 3 },
      next: { revalidate },
    });
    const items = asArray<Collection>(response.data).slice(0, 3);
    return {
      items,
      count: resolveCount(response.data, response.meta?.pagination?.count),
    };
  } catch {
    return { items: [] as Collection[], count: 0 };
  }
});

async function getEndpointCount(path: string, params?: QueryParams) {
  try {
    const response = await apiFetch<ListPayload<unknown>>(path, {
      params: { page_size: 1, ...params },
      next: { revalidate },
    });
    return resolveCount(response.data, response.meta?.pagination?.count);
  } catch {
    return 0;
  }
}

type SocialLink = { label: string; href: string };

function getSocialLinks(
  siteSettings: SiteSettings | null,
  contactSettings: ContactSettings | null
): SocialLink[] {
  const fromContactSettings = Object.entries(contactSettings?.social_links || {})
    .map(([key, value]) => ({
      label: key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase()),
      href: normalizeExternalHref(value),
    }))
    .filter((item) => item.href);

  const fromSiteSettings = SOCIAL_SITE_FIELDS.map((field) => ({
    label: field.label,
    href: normalizeExternalHref(siteSettings?.[field.key] as string | null | undefined),
  })).filter((item) => item.href);

  const candidateLinks = fromContactSettings.length ? fromContactSettings : fromSiteSettings;
  const seen = new Set<string>();
  return candidateLinks.filter((item) => {
    const normalized = item.href.toLowerCase();
    if (seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
}

export async function generateMetadata(): Promise<Metadata> {
  const [page, siteSettings] = await Promise.all([getAboutPage(), getSiteSettings()]);
  const brandName = pickText(siteSettings?.site_name) || "Bunoraa";
  const title = page?.meta_title || page?.title || `About ${brandName}`;
  const description =
    page?.meta_description ||
    page?.excerpt ||
    pickText(siteSettings?.site_description, siteSettings?.site_tagline, siteSettings?.tagline) ||
    `Learn how ${brandName} curates products, collections, and artisan stories.`;

  return buildPageMetadata({
    title,
    description,
    path: "/about/",
    images: page?.featured_image ? [page.featured_image] : undefined,
  });
}

export default async function AboutPage() {
  const [
    page,
    siteSettings,
    contactSettings,
    categorySnapshot,
    collectionSnapshot,
    productCount,
    bundleCount,
    artisanCount,
  ] = await Promise.all([
    getAboutPage(),
    getSiteSettings(),
    getContactSettings(),
    getCategorySnapshot(),
    getCollectionSnapshot(),
    getEndpointCount("/catalog/products/"),
    getEndpointCount("/catalog/bundles/"),
    getEndpointCount("/artisans/"),
  ]);

  const brandName = pickText(siteSettings?.site_name) || "Bunoraa";
  const title = page?.title || `About ${brandName}`;
  const heroSummary =
    page?.excerpt ||
    pickText(siteSettings?.site_description, siteSettings?.site_tagline, siteSettings?.tagline) ||
    `${brandName} curates thoughtful products and collections for everyday life.`;

  const supportEmail = pickText(
    contactSettings?.support_email,
    siteSettings?.support_email,
    contactSettings?.general_email,
    siteSettings?.contact_email
  );
  const salesEmail = pickText(contactSettings?.sales_email);
  const phone = pickText(contactSettings?.phone, siteSettings?.contact_phone);
  const address = pickText(siteSettings?.contact_address, siteSettings?.address);
  const businessHours = pickText(contactSettings?.business_hours_note);
  const responseTime = pickText(siteSettings?.support_reply_time_note);
  const socialLinks = getSocialLinks(siteSettings, contactSettings);
  const updatedAtLabel = formatDate(page?.updated_at || page?.created_at);

  const stats = [
    { key: "products", label: "Products", value: productCount, href: "/products/" },
    {
      key: "categories",
      label: "Categories",
      value: categorySnapshot.count,
      href: "/categories/",
    },
    {
      key: "collections",
      label: "Collections",
      value: collectionSnapshot.count,
      href: "/collections/",
    },
    {
      key: "bundles",
      label: "Bundles",
      value: bundleCount,
      href: bundleCount > 0 ? "/bundles/" : "",
    },
    { key: "artisans", label: "Artisans", value: artisanCount, href: "/artisans/" },
  ];

  const aboutPageSchema = cleanObject({
    "@context": "https://schema.org",
    "@type": "AboutPage",
    name: page?.meta_title || title,
    description: page?.meta_description || heroSummary,
    url: absoluteUrl("/about/"),
  });

  const organizationSchema = cleanObject({
    "@context": "https://schema.org",
    "@type": "Organization",
    name: brandName,
    description: pickText(siteSettings?.site_description, heroSummary),
    url: absoluteUrl("/"),
    email: supportEmail || undefined,
    telephone: phone || undefined,
    address: address
      ? {
          "@type": "PostalAddress",
          streetAddress: address,
        }
      : undefined,
    sameAs: socialLinks
      .filter((item) => /^https?:\/\//i.test(item.href))
      .map((item) => item.href),
  });

  return (
    <div className="mx-auto w-full max-w-6xl px-3 sm:px-5 py-12 space-y-10">
      <section className="overflow-hidden rounded-3xl border border-border bg-gradient-to-br from-primary/15 via-background to-accent/10 px-5 sm:px-8 py-8 sm:py-10">
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">About</p>
        <h1 className="mt-2 text-3xl sm:text-4xl font-semibold tracking-tight">{title}</h1>
        <p className="mt-4 max-w-3xl text-sm sm:text-base text-foreground/75">
          {stripHtml(heroSummary)}
        </p>
        <div className="mt-6 flex flex-wrap items-center gap-3">
          <Button asChild variant="primary-gradient">
            <Link href="/products/">Explore products</Link>
          </Button>
          <Button asChild variant="secondary">
            <Link href="/contact/">Contact us</Link>
          </Button>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {stats.map((item) => (
          <Card key={item.key} variant="bordered" className="rounded-2xl p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">{item.label}</p>
            <p className="mt-2 text-2xl font-semibold">{formatCount(item.value)}</p>
            {item.href ? (
              <Link href={item.href} className="mt-3 inline-flex text-sm text-primary">
                View {item.label.toLowerCase()}
              </Link>
            ) : (
              <p className="mt-3 text-sm text-foreground/60">Currently unavailable</p>
            )}
          </Card>
        ))}
      </section>

      <section className="grid gap-6 lg:grid-cols-3">
        <Card variant="bordered" className="lg:col-span-2 p-6 sm:p-7">
          <h2 className="text-xl font-semibold">Our story</h2>
          {page?.content ? (
            <div
              className="prose prose-slate mt-5 max-w-none"
              dangerouslySetInnerHTML={{ __html: page.content }}
            />
          ) : (
            <div className="mt-5 space-y-3 text-sm text-foreground/75">
              <p>
                {brandName} exists to make discovering quality products simple, thoughtful, and
                inspiring.
              </p>
              <p>
                We continuously refine our catalog, collaborate with makers, and design better
                shopping experiences across products, collections, and preorders.
              </p>
            </div>
          )}
        </Card>

        <Card variant="bordered" className="p-6 sm:p-7">
          <h2 className="text-xl font-semibold">Quick facts</h2>
          <dl className="mt-5 space-y-3 text-sm">
            <div>
              <dt className="text-foreground/60">Brand</dt>
              <dd className="font-medium">{brandName}</dd>
            </div>
            {updatedAtLabel ? (
              <div>
                <dt className="text-foreground/60">Page updated</dt>
                <dd className="font-medium">{updatedAtLabel}</dd>
              </div>
            ) : null}
            {supportEmail ? (
              <div>
                <dt className="text-foreground/60">Support</dt>
                <dd>
                  <Link href={`mailto:${supportEmail}`} className="font-medium text-primary">
                    {supportEmail}
                  </Link>
                </dd>
              </div>
            ) : null}
            {phone ? (
              <div>
                <dt className="text-foreground/60">Phone</dt>
                <dd>
                  <Link href={`tel:${phone}`} className="font-medium text-primary">
                    {phone}
                  </Link>
                </dd>
              </div>
            ) : null}
            {businessHours ? (
              <div>
                <dt className="text-foreground/60">Business hours</dt>
                <dd className="font-medium">{businessHours}</dd>
              </div>
            ) : null}
            {responseTime ? (
              <div>
                <dt className="text-foreground/60">Support response time</dt>
                <dd className="font-medium">{responseTime}</dd>
              </div>
            ) : null}
          </dl>
        </Card>
      </section>

      {categorySnapshot.items.length || collectionSnapshot.items.length ? (
        <section className="grid gap-6 lg:grid-cols-2">
          {categorySnapshot.items.length ? (
            <Card variant="bordered" className="p-6 sm:p-7">
              <h2 className="text-xl font-semibold">Top categories</h2>
              <div className="mt-5 flex flex-wrap gap-2.5">
                {categorySnapshot.items.map((category) => (
                  <Link
                    key={category.id}
                    href={buildCategoryPath(category.slug)}
                    className="inline-flex items-center rounded-full border border-border bg-muted/30 px-3 py-1.5 text-sm hover:bg-muted"
                  >
                    {category.name}
                    {typeof category.product_count === "number" ? (
                      <span className="ml-2 text-xs text-foreground/60">
                        {category.product_count}
                      </span>
                    ) : null}
                  </Link>
                ))}
              </div>
            </Card>
          ) : null}

          {collectionSnapshot.items.length ? (
            <Card variant="bordered" className="p-6 sm:p-7">
              <h2 className="text-xl font-semibold">Featured collections</h2>
              <ul className="mt-5 space-y-3">
                {collectionSnapshot.items.map((collection) => (
                  <li key={collection.id}>
                    <Link
                      href={`/collections/${collection.slug}/`}
                      className="flex items-start justify-between gap-3 rounded-xl border border-border/70 bg-card/40 px-3 py-3 text-sm transition hover:border-border"
                    >
                      <span className="font-medium">{collection.name}</span>
                      <span className="text-foreground/60">View</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </Card>
          ) : null}
        </section>
      ) : null}

      <section className="grid gap-6 lg:grid-cols-2">
        <Card variant="bordered" className="p-6 sm:p-7">
          <h2 className="text-xl font-semibold">Contact & support</h2>
          <div className="mt-5 space-y-3 text-sm">
            {supportEmail ? (
              <p>
                <span className="text-foreground/60">Support:</span>{" "}
                <Link href={`mailto:${supportEmail}`} className="text-primary">
                  {supportEmail}
                </Link>
              </p>
            ) : null}
            {salesEmail ? (
              <p>
                <span className="text-foreground/60">Sales:</span>{" "}
                <Link href={`mailto:${salesEmail}`} className="text-primary">
                  {salesEmail}
                </Link>
              </p>
            ) : null}
            {phone ? (
              <p>
                <span className="text-foreground/60">Phone:</span>{" "}
                <Link href={`tel:${phone}`} className="text-primary">
                  {phone}
                </Link>
              </p>
            ) : null}
            {address ? (
              <p>
                <span className="text-foreground/60">Address:</span> {address}
              </p>
            ) : null}
            {!supportEmail && !salesEmail && !phone && !address ? (
              <p className="text-foreground/70">
                Contact details are being updated. Visit our contact page for the latest options.
              </p>
            ) : null}
          </div>
          <div className="mt-6">
            <Button asChild variant="secondary">
              <Link href="/contact/">Open contact page</Link>
            </Button>
          </div>
        </Card>

        <Card variant="bordered" className="p-6 sm:p-7">
          <h2 className="text-xl font-semibold">Follow {brandName}</h2>
          {socialLinks.length ? (
            <ul className="mt-5 grid gap-2.5 sm:grid-cols-2 text-sm">
              {socialLinks.map((link) => (
                <li key={`${link.label}-${link.href}`}>
                  <Link
                    href={link.href}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center justify-between rounded-xl border border-border/70 bg-card/40 px-3 py-2.5 transition hover:border-border"
                  >
                    <span>{link.label}</span>
                    <span className="text-foreground/60">Visit</span>
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-5 text-sm text-foreground/70">
              Social profiles are not configured yet. Check back soon.
            </p>
          )}
        </Card>
      </section>

      <JsonLd data={[aboutPageSchema, organizationSchema]} />
    </div>
  );
}
