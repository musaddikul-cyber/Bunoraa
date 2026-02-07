import type { Metadata } from "next";
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

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
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
    </html>
  );
}
