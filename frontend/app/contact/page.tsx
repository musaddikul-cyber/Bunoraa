import type { Metadata } from "next";
import dynamic from "next/dynamic";
import { buildPageMetadata } from "@/lib/seo";

const ContactPageClient = dynamic(
  () => import("@/components/contact/ContactPageClient").then((mod) => mod.ContactPageClient)
);

export const metadata: Metadata = buildPageMetadata({
  title: "Contact Bunoraa",
  description: "Contact Bunoraa for support, sales, and partnership inquiries.",
  path: "/contact/",
});

export default function ContactPage() {
  return <ContactPageClient />;
}
