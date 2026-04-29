import type { Metadata, Viewport } from "next";
import { unstable_noStore as noStore } from "next/cache";
import "./globals.css";
import { Providers } from "@/components/providers/Providers";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { SkipToContent } from "@/components/layout/SkipToContent";
import { JsonLd } from "@/components/seo/JsonLd";
import { DEFAULT_OG_IMAGE_PATH, SITE_NAME, SITE_URL, absoluteUrl, cleanObject } from "@/lib/seo";
import Script from "next/script";
import { DeferredClientEnhancements } from "@/components/layout/DeferredClientEnhancements";

const SITE_DESCRIPTION =
  "Discover curated products, bundles, and artisan-made collections at Bunoraa.";
const metadataBase = new URL(
  SITE_URL.startsWith("http://") || SITE_URL.startsWith("https://")
    ? SITE_URL
    : `https://${SITE_URL}`
);

export const metadata: Metadata = {
  metadataBase,
  title: {
    default: SITE_NAME,
    template: `%s | ${SITE_NAME}`,
  },
  description: SITE_DESCRIPTION,
  alternates: {
    canonical: "/",
  },
  applicationName: SITE_NAME,
  manifest: "/site.webmanifest",
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "16x16", type: "image/x-icon" },
      { url: "/favicon.ico", sizes: "32x32", type: "image/x-icon" },
      { url: "/icon.png", sizes: "512x512", type: "image/png" },
    ],
    shortcut: [{ url: "/favicon.ico" }],
    apple: [{ url: "/apple-icon.png", sizes: "180x180", type: "image/png" }],
  },
  keywords: [
    "Bunoraa",
    "ecommerce",
    "online shopping",
    "artisan products",
    "collections",
    "bundles",
  ],
  openGraph: {
    type: "website",
    url: absoluteUrl("/"),
    siteName: SITE_NAME,
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
    images: [absoluteUrl(DEFAULT_OG_IMAGE_PATH)],
  },
  twitter: {
    card: "summary_large_image",
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
    images: [absoluteUrl(DEFAULT_OG_IMAGE_PATH)],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
      "max-video-preview": -1,
    },
  },
  verification: cleanObject({
    google: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION || undefined,
    yandex: process.env.NEXT_PUBLIC_YANDEX_VERIFICATION || undefined,
    yahoo: process.env.NEXT_PUBLIC_YAHOO_SITE_VERIFICATION || undefined,
    other: process.env.NEXT_PUBLIC_BING_SITE_VERIFICATION
      ? {
          "msvalidate.01": process.env.NEXT_PUBLIC_BING_SITE_VERIFICATION,
        }
      : undefined,
  }),
  other: {
    "apple-mobile-web-app-title": SITE_NAME,
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

const disablePrerender =
  process.env.NEXT_DISABLE_PRERENDER === "true" ||
  process.env.NEXT_DISABLE_PRERENDER === "1";

const themeBootstrapScript = `
(() => {
  try {
    const key = "bunoraa-theme";
    const allowed = ["light", "dark", "moonlight", "gray", "modern", "system"];
    const stored = window.localStorage.getItem(key);
    const theme = allowed.includes(String(stored)) ? String(stored) : "system";
    const root = document.documentElement;
    root.classList.remove("light", "dark", "moonlight", "gray", "modern", "system");

    if (theme === "system") {
      root.classList.add("system");
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.classList.toggle("dark", prefersDark);
      root.style.colorScheme = "light dark";
      return;
    }

    root.classList.add(theme);
    root.classList.toggle("dark", theme === "dark");
    root.style.colorScheme = theme === "dark" ? "dark" : "light";
  } catch (_error) {
    // Ignore theme bootstrap failures and keep default system theme.
  }
})();
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const organizationSchema = cleanObject({
    "@context": "https://schema.org",
    "@type": "Organization",
    "@id": absoluteUrl("/#organization"),
    name: SITE_NAME,
    url: absoluteUrl("/"),
    logo: cleanObject({
      "@type": "ImageObject",
      url: absoluteUrl("/icon.png"),
    }),
  });

  const websiteSchema = cleanObject({
    "@context": "https://schema.org",
    "@type": "WebSite",
    "@id": absoluteUrl("/#website"),
    name: SITE_NAME,
    alternateName: "Bunoraa Marketplace",
    url: absoluteUrl("/"),
    publisher: {
      "@id": absoluteUrl("/#organization"),
    },
    potentialAction: {
      "@type": "SearchAction",
      target: `${SITE_URL}/search/?q={search_term_string}`,
      "query-input": "required name=search_term_string",
    },
  });

  if (disablePrerender) {
    noStore();
  }
  return (
    <html lang="en" className="system" suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground antialiased">
        <Script id="theme-bootstrap" strategy="beforeInteractive">
          {themeBootstrapScript}
        </Script>
        <SkipToContent />
        <Providers>
          <Header />
          <main id="main-content" className="min-h-[70vh]" role="main" aria-label="Main content">
            {children}
          </main>
          <Footer />
          <DeferredClientEnhancements />
        </Providers>
        <JsonLd data={[organizationSchema, websiteSchema]} />
      </body>
    </html>
  );
}
