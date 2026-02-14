import type { Metadata, Viewport } from "next";
import { unstable_noStore as noStore } from "next/cache";
import "./globals.css";
import { Providers } from "@/components/providers/Providers";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { ChatWidget } from "@/components/chat/ChatWidget";
import { CompareTray } from "@/components/products/CompareTray";
import { PageViewTracker } from "@/components/analytics/PageViewTracker";
import { JsonLd } from "@/components/seo/JsonLd";
import { SITE_URL, absoluteUrl, cleanObject } from "@/lib/seo";
import Script from "next/script";

const SITE_NAME = "Bunoraa";
const SITE_DESCRIPTION = "Bunoraa e-commerce storefront";
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
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    url: SITE_URL,
    siteName: SITE_NAME,
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
  },
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
        <a href="#main-content" className="skip-link">
          Skip to main content
        </a>
        <Providers>
          <PageViewTracker />
          <Header />
          <main id="main-content" className="min-h-[70vh]">
            {children}
          </main>
          <Footer />
          <CompareTray />
          <ChatWidget />
        </Providers>
        <JsonLd data={[organizationSchema, websiteSchema]} />
        <Script
          src="https://static.cloudflareinsights.com/beacon.min.js"
          strategy="afterInteractive"
          data-cf-beacon='{"token": "99cd4569fd314a31bb530d46e16f26c9"}'
        />
      </body>
    </html>
  );
}
