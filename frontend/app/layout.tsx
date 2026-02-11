import type { Metadata } from "next";
import { unstable_noStore as noStore } from "next/cache";
import "./globals.css";
import { Providers } from "@/components/providers/Providers";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { ChatWidget } from "@/components/chat/ChatWidget";
import { CompareTray } from "@/components/products/CompareTray";
import { JsonLd } from "@/components/seo/JsonLd";
import { SITE_URL, absoluteUrl, cleanObject } from "@/lib/seo";
import Script from "next/script";

export const metadata: Metadata = {
  title: "Bunoraa",
  description: "Bunoraa e-commerce storefront",
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
    name: "Bunoraa",
    url: SITE_URL,
    logo: absoluteUrl("/favicon.ico"),
  });

  const websiteSchema = cleanObject({
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: "Bunoraa",
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
        <Providers>
          <Header />
          <main className="min-h-[70vh]">{children}</main>
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
