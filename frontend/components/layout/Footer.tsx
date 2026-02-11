import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type {
  Category,
  Collection,
  ContactSettings,
  MenuPage,
  SiteSettings,
  SocialLink,
  StoreLocation,
} from "@/lib/types";
import { ThemeSwitcher } from "@/components/theme/ThemeProvider";
import { LocaleSwitcher } from "@/components/locale/LocaleSwitcher";
import { FooterNewsletter } from "@/components/layout/FooterNewsletter";
import { asArray } from "@/lib/array";

async function getFooterPages() {
  try {
    const response = await apiFetch<MenuPage[]>("/pages/footer/", {
      next: { revalidate: 600 },
    });
    return asArray<MenuPage>(response.data);
  } catch {
    return [];
  }
}

async function getSiteSettings() {
  try {
    const response = await apiFetch<SiteSettings>("/pages/settings/", {
      next: { revalidate: 600 },
    });
    return response.data;
  } catch {
    return null;
  }
}

async function getContactSettings() {
  try {
    const response = await apiFetch<ContactSettings>("/contacts/settings/", {
      next: { revalidate: 600 },
    });
    return response.data;
  } catch {
    return null;
  }
}

async function getTopCategories() {
  try {
    const response = await apiFetch<Category[]>("/catalog/categories/", {
      params: { page_size: 6 },
      next: { revalidate: 600 },
    });
    return asArray<Category>(response.data);
  } catch {
    return [];
  }
}

async function getCollections() {
  try {
    const response = await apiFetch<Collection[]>("/catalog/collections/", {
      params: { page_size: 6 },
      next: { revalidate: 600 },
    });
    return asArray<Collection>(response.data);
  } catch {
    return [];
  }
}

async function getMainLocation() {
  try {
    const response = await apiFetch<StoreLocation>("/contacts/locations/main/", {
      next: { revalidate: 600 },
    });
    return response.data;
  } catch {
    return null;
  }
}

const SOCIAL_LABELS: Record<string, string> = {
  facebook: "Facebook",
  instagram: "Instagram",
  twitter: "Twitter",
  linkedin: "LinkedIn",
  youtube: "YouTube",
  tiktok: "TikTok",
  pinterest: "Pinterest",
};

const SOCIAL_SITE_FIELDS: Array<{ key: keyof SiteSettings; label: string }> = [
  { key: "facebook_url", label: "Facebook" },
  { key: "instagram_url", label: "Instagram" },
  { key: "twitter_url", label: "Twitter" },
  { key: "linkedin_url", label: "LinkedIn" },
  { key: "youtube_url", label: "YouTube" },
  { key: "tiktok_url", label: "TikTok" },
];

const pickText = (...values: Array<string | null | undefined>) => {
  for (const value of values) {
    if (value && value.trim()) return value.trim();
  }
  return "";
};

export async function Footer() {
  const [
    pagesResult,
    siteSettingsResult,
    contactSettingsResult,
    categoriesResult,
    collectionsResult,
    mainLocationResult,
  ] = await Promise.allSettled([
    getFooterPages(),
    getSiteSettings(),
    getContactSettings(),
    getTopCategories(),
    getCollections(),
    getMainLocation(),
  ]);

  const pages = pagesResult.status === "fulfilled" ? pagesResult.value : [];
  const siteSettings = siteSettingsResult.status === "fulfilled" ? siteSettingsResult.value : null;
  const contactSettings =
    contactSettingsResult.status === "fulfilled" ? contactSettingsResult.value : null;
  const categories = categoriesResult.status === "fulfilled" ? categoriesResult.value : [];
  const collections = collectionsResult.status === "fulfilled" ? collectionsResult.value : [];
  const mainLocation =
    mainLocationResult.status === "fulfilled" ? mainLocationResult.value : null;

  const brandName = pickText(siteSettings?.site_name) || "Bunoraa";
  const brandDescription =
    pickText(
      siteSettings?.footer_text,
      siteSettings?.site_description,
      siteSettings?.tagline,
      siteSettings?.site_tagline
    ) ||
    "Discover handcrafted fashion, home, and lifestyle essentials curated by Bunoraa artisans.";

  const emailItems = [
    { label: "Support", value: pickText(siteSettings?.support_email) },
    { label: "Email", value: pickText(siteSettings?.contact_email) },
  ]
    .filter((item) => item.value)
    .filter(
      (item, index, self) =>
        self.findIndex((entry) => entry.value === item.value) === index
    );

  const phone = pickText(siteSettings?.contact_phone);
  const address = pickText(siteSettings?.address, siteSettings?.contact_address);

  const contactSocialLinks: SocialLink[] = contactSettings?.social_links
    ? Object.entries(contactSettings.social_links)
        .filter(([, url]) => url && String(url).trim())
        .map(([key, url]) => ({
          label: SOCIAL_LABELS[key] || key.replace(/_/g, " "),
          url: String(url),
        }))
    : [];

  const siteSocialLinks: SocialLink[] = siteSettings
    ? SOCIAL_SITE_FIELDS.map((field) => ({
        label: field.label,
        url: pickText(siteSettings[field.key] as string | null | undefined),
      })).filter((item) => item.url)
    : [];

  const socialLinks = contactSocialLinks.length ? contactSocialLinks : siteSocialLinks;
  const copyrightText =
    pickText(siteSettings?.copyright_text) || `${brandName}. All rights reserved.`;

  return (
    <footer id="footer" className="border-t border-border bg-card">
      <div className="mx-auto w-full max-w-7xl px-6 py-12">
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-6 xl:grid-cols-7">
          <div className="space-y-4 sm:col-span-2 lg:col-span-1 xl:col-span-2">
            <div>
              <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
                {brandName}
              </p>
              <p className="mt-2 text-sm text-foreground/70">{brandDescription}</p>
            </div>
            <FooterNewsletter />
            {socialLinks.length ? (
              <div>
                <p className="text-sm font-semibold">Follow along</p>
                <ul className="mt-3 flex flex-wrap gap-3 text-sm text-foreground/70">
                  {socialLinks.map((link) => (
                    <li key={link.url}>
                      <Link className="hover:text-foreground" href={link.url}>
                        {link.label}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>

          <div>
            <p className="text-sm font-semibold">Shop</p>
            <ul className="mt-3 space-y-2 text-sm text-foreground/70">
              {categories.length ? (
                categories.map((category) => (
                  <li key={category.id}>
                    <Link href={`/categories/${category.slug}/`}>{category.name}</Link>
                  </li>
                ))
              ) : (
                <>
                  <li>
                    <Link href="/categories/">Browse categories</Link>
                  </li>
                  <li>
                    <Link href="/products/">All products</Link>
                  </li>
                </>
              )}
            </ul>
          </div>

          <div>
            <p className="text-sm font-semibold">Collections</p>
            <ul className="mt-3 space-y-2 text-sm text-foreground/70">
              {collections.length ? (
                collections.slice(0, 4).map((collection) => (
                  <li key={collection.id}>
                    <Link href={`/collections/${collection.slug}/`}>{collection.name}</Link>
                  </li>
                ))
              ) : (
                <li>
                  <Link href="/collections/">All collections</Link>
                </li>
              )}
              <li>
                <Link href="/bundles/">Bundles</Link>
              </li>
              <li>
                <Link href="/artisans/">Artisans</Link>
              </li>
              <li>
                <Link href="/preorders/">Preorders</Link>
              </li>
            </ul>
          </div>

          <div>
            <p className="text-sm font-semibold">Company</p>
            <ul className="mt-3 space-y-2 text-sm text-foreground/70">
              {pages.length ? (
                pages.map((page) => (
                  <li key={page.id}>
                    <Link href={`/pages/${page.slug}/`}>{page.title}</Link>
                  </li>
                ))
              ) : (
                <>
                  <li>
                    <Link href="/about/">About Bunoraa</Link>
                  </li>
                  <li>
                    <Link href="/faq/">FAQ</Link>
                  </li>
                </>
              )}
            </ul>
          </div>

          <div>
            <p className="text-sm font-semibold">Support</p>
            <ul className="mt-3 space-y-2 text-sm text-foreground/70">
              <li>
                <Link href="/contact/">Contact</Link>
              </li>
              <li>
                <Link href="/faq/">FAQ</Link>
              </li>
              <li>
                <Link href="/pages/shipping/">Shipping</Link>
              </li>
              <li>
                <Link href="/pages/returns/">Returns</Link>
              </li>
            </ul>
          </div>

          <div>
            <p className="text-sm font-semibold">Contact & Location</p>
            <ul className="mt-3 space-y-2 text-sm text-foreground/70">
              {emailItems.map((item) => (
                <li key={item.label}>
                  <span className="text-foreground/60">{item.label}:</span>{" "}
                  <Link href={`mailto:${item.value}`}>{item.value}</Link>
                </li>
              ))}
              {phone ? (
                <li>
                  <span className="text-foreground/60">Phone:</span>{" "}
                  <Link href={`tel:${phone}`}>{phone}</Link>
                </li>
              ) : null}
              {address ? <li>{address}</li> : null}
              {mainLocation?.full_address ? (
                <li>
                  <span className="text-foreground/60">Main store:</span>{" "}
                  {mainLocation.full_address}
                </li>
              ) : null}
            </ul>
          </div>
        </div>

        <div className="mt-10 flex flex-col gap-4 border-t border-border pt-6 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-foreground/60">
            &copy; {new Date().getFullYear()} {copyrightText}
          </p>
          <div className="flex w-full flex-col gap-3 text-sm text-foreground/70 sm:w-auto sm:flex-row sm:items-center sm:justify-end">
            <ThemeSwitcher />
            <LocaleSwitcher />
          </div>
        </div>
      </div>
    </footer>
  );
}
