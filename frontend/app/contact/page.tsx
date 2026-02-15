import type { Metadata } from "next";
import { ContactPageClient } from "@/components/contact/ContactPageClient";
import { buildPageMetadata } from "@/lib/seo";

export const metadata: Metadata = buildPageMetadata({
  title: "Contact Bunoraa",
  description: "Contact Bunoraa for support, sales, and partnership inquiries.",
  path: "/contact/",
});

export default function ContactPage() {
  return <ContactPageClient />;
}
