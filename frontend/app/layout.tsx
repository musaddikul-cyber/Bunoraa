import type { Metadata, Viewport } from "next";
import { unstable_noStore as noStore } from "next/cache";
import { Suspense } from "react";
import "./globals.css";
import { Providers } from "@/components/providers/Providers";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { ChatWidget } from "@/components/chat/ChatWidget";
import { CompareTray } from "@/components/products/CompareTray";
import { PageViewTracker } from "@/components/analytics/PageViewTracker";
import { JsonLd } from "@/components/seo/JsonLd";
import { DEFAULT_OG_IMAGE_PATH, SITE_NAME, SITE_URL, absoluteUrl, cleanObject } from "@/lib/seo";
import Script from "next/script";

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

// export const dynamic = "force-dynamic";

const disablePrerender =
  process.env.NEXT_DISABLE_PRERENDER === "true" ||
  process.env.NEXT_DISABLE_PRERENDER === "1";
const shouldLoadCloudflareBeacon = process.env.NODE_ENV === "production";

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
    name: SITE_NAME,
    url: SITE_URL,
    logo: absoluteUrl("/favicon.ico"),
  });

  const websiteSchema = cleanObject({
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: SITE_NAME,
    alternateName: "bunoraa.com",
    url: SITE_URL,
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
        <a href="#main-content" className="skip-link">
          Skip to main content
        </a>
        <Providers>
          <Suspense fallback={null}>
            <PageViewTracker />
          </Suspense>
          <Header />
          <main id="main-content" className="min-h-[70vh]">
            {children}
          </main>
          <Footer />
          <CompareTray />
          <ChatWidget />
        </Providers>
        <JsonLd data={[organizationSchema, websiteSchema]} />
        {shouldLoadCloudflareBeacon ? (
          <Script
            src="https://static.cloudflareinsights.com/beacon.min.js"
            strategy="afterInteractive"
            data-cf-beacon='{"token": "99cd4569fd314a31bb530d46e16f26c9"}'
          />
        ) : null}
      </body>
    </html>
  );
}
