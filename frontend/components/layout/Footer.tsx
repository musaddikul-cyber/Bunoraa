import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type {
  Category,
  Collection,
  ContactSettings,
  MenuPage,
  SiteSettings,
  SocialLink,
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
  ] = await Promise.allSettled([
    getFooterPages(),
    getSiteSettings(),
    getContactSettings(),
    getTopCategories(),
    getCollections(),
  ]);

  const pages = pagesResult.status === "fulfilled" ? pagesResult.value : [];
  const siteSettings = siteSettingsResult.status === "fulfilled" ? siteSettingsResult.value : null;
  const contactSettings =
    contactSettingsResult.status === "fulfilled" ? contactSettingsResult.value : null;
  const categories = categoriesResult.status === "fulfilled" ? categoriesResult.value : [];
  const collections = collectionsResult.status === "fulfilled" ? collectionsResult.value : [];

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
  const fallbackCompanyLinks = [
    { label: "About Bunoraa", href: "/about/" },
    { label: "FAQ", href: "/faq/" },
  ];
  const companyLinks = pages.length
    ? pages.map((page) => ({ label: page.title, href: `/pages/${page.slug}/` }))
    : fallbackCompanyLinks;
  const supportLinks = [
    { label: "Contact", href: "/contact/" },
    { label: "FAQ", href: "/faq/" },
    { label: "Shipping", href: "/pages/shipping/" },
    { label: "Returns", href: "/pages/returns/" },
  ];
  const mergedCompanyLinks = Array.from(
    new Map(
      [...companyLinks, ...supportLinks].map((item) => [item.href, item])
    ).values()
  );
  const shopLinks = categories.length
    ? categories.map((category) => ({
        key: category.id,
        label: category.name,
        href: `/categories/${category.slug}/`,
      }))
    : [
        { key: "browse-categories", label: "Browse categories", href: "/categories/" },
        { key: "all-products", label: "All products", href: "/products/" },
      ];

  const collectionLinks = [
    ...(collections.length
      ? collections.slice(0, 4).map((collection) => ({
          key: collection.id,
          label: collection.name,
          href: `/collections/${collection.slug}/`,
        }))
      : [{ key: "all-collections", label: "All collections", href: "/collections/" }]),
    { key: "collections-all-products", label: "All products", href: "/products/" },
    { key: "collections-bundles", label: "Bundles", href: "/bundles/" },
    { key: "collections-artisans", label: "Artisans", href: "/artisans/" },
    { key: "collections-preorders", label: "Preorders", href: "/preorders/" },
  ];

  const contactItems = [
    ...emailItems.map((item) => ({
      key: `email-${item.label}`,
      label: item.label,
      value: item.value,
      href: `mailto:${item.value}`,
    })),
    ...(phone
      ? [
          {
            key: "phone",
            label: "Phone",
            value: phone,
            href: `tel:${phone}`,
          },
        ]
      : []),
    ...(address
      ? [
          {
            key: "address",
            label: "",
            value: address,
            href: "",
          },
        ]
      : []),
  ];

  const footerAccordionClass =
    "group rounded-xl border border-border bg-background/40";
  const footerSummaryClass =
    "flex min-h-11 cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm font-semibold text-foreground/90 [&::-webkit-details-marker]:hidden";
  const footerListClass = "space-y-2 border-t border-border px-4 pb-4 pt-3 text-sm text-foreground/70";

  return (
    <footer id="footer" className="border-t border-border bg-card">
      <div className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-12">
        <div className="space-y-8 lg:hidden">
          <div className="space-y-4">
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

          <div className="space-y-3">
            <details className={footerAccordionClass} name="footer-sections" open>
              <summary className={footerSummaryClass}>
                <span>Shop</span>
                <svg
                  aria-hidden="true"
                  viewBox="0 0 20 20"
                  className="h-4 w-4 shrink-0 text-foreground/60 transition group-open:rotate-180"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M5 7.5l5 5 5-5" />
                </svg>
              </summary>
              <ul className={footerListClass}>
                {shopLinks.map((item) => (
                  <li key={item.key}>
                    <Link href={item.href}>{item.label}</Link>
                  </li>
                ))}
              </ul>
            </details>

            <details className={footerAccordionClass} name="footer-sections">
              <summary className={footerSummaryClass}>
                <span>Collections</span>
                <svg
                  aria-hidden="true"
                  viewBox="0 0 20 20"
                  className="h-4 w-4 shrink-0 text-foreground/60 transition group-open:rotate-180"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M5 7.5l5 5 5-5" />
                </svg>
              </summary>
              <ul className={footerListClass}>
                {collectionLinks.map((item) => (
                  <li key={item.key}>
                    <Link href={item.href}>{item.label}</Link>
                  </li>
                ))}
              </ul>
            </details>

            <details className={footerAccordionClass} name="footer-sections">
              <summary className={footerSummaryClass}>
                <span>Company & Support</span>
                <svg
                  aria-hidden="true"
                  viewBox="0 0 20 20"
                  className="h-4 w-4 shrink-0 text-foreground/60 transition group-open:rotate-180"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M5 7.5l5 5 5-5" />
                </svg>
              </summary>
              <ul className={footerListClass}>
                {mergedCompanyLinks.map((item) => (
                  <li key={item.href}>
                    <Link href={item.href}>{item.label}</Link>
                  </li>
                ))}
              </ul>
            </details>

            <details className={footerAccordionClass} name="footer-sections">
              <summary className={footerSummaryClass}>
                <span>Contact & Location</span>
                <svg
                  aria-hidden="true"
                  viewBox="0 0 20 20"
                  className="h-4 w-4 shrink-0 text-foreground/60 transition group-open:rotate-180"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M5 7.5l5 5 5-5" />
                </svg>
              </summary>
              <ul className={footerListClass}>
                {contactItems.map((item) => (
                  <li key={item.key}>
                    {item.label ? (
                      <>
                        <span className="text-foreground/60">{item.label}:</span>{" "}
                        {item.href ? <Link href={item.href}>{item.value}</Link> : item.value}
                      </>
                    ) : (
                      item.value
                    )}
                  </li>
                ))}
              </ul>
            </details>
          </div>
        </div>

        <div className="hidden gap-8 sm:grid-cols-2 lg:grid lg:grid-cols-6">
          <div className="space-y-4 sm:col-span-2 lg:col-span-2">
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
              {shopLinks.map((item) => (
                <li key={item.key}>
                  <Link href={item.href}>{item.label}</Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <p className="text-sm font-semibold">Collections</p>
            <ul className="mt-3 space-y-2 text-sm text-foreground/70">
              {collectionLinks.map((item) => (
                <li key={item.key}>
                  <Link href={item.href}>{item.label}</Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <p className="text-sm font-semibold">Company & Support</p>
            <ul className="mt-3 space-y-2 text-sm text-foreground/70">
              {mergedCompanyLinks.map((item) => (
                <li key={item.href}>
                  <Link href={item.href}>{item.label}</Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <p className="text-sm font-semibold">Contact & Location</p>
            <ul className="mt-3 space-y-2 text-sm text-foreground/70">
              {contactItems.map((item) => (
                <li key={item.key}>
                  {item.label ? (
                    <>
                      <span className="text-foreground/60">{item.label}:</span>{" "}
                      {item.href ? <Link href={item.href}>{item.value}</Link> : item.value}
                    </>
                  ) : (
                    item.value
                  )}
                </li>
              ))}
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
