import type { Metadata } from "next";
import { unstable_noStore as noStore } from "next/cache";
import "./globals.css";
import { Providers } from "@/components/providers/Providers";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { ChatWidget } from "@/components/chat/ChatWidget";
import { CompareTray } from "@/components/products/CompareTray";

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
      </body>
      <script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "99cd4569fd314a31bb530d46e16f26c9"}'></script>
    </html>
  );
}
