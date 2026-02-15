import type { Metadata } from "next";
import { buildNoIndexMetadata } from "@/lib/seo";

export const metadata: Metadata = buildNoIndexMetadata({
  title: "My Subscriptions",
  description: "Manage your active Bunoraa subscriptions.",
  path: "/subscriptions/my-subscriptions/",
});

export default function MySubscriptionsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
