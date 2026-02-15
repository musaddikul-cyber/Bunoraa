import type { Metadata } from "next";
import { buildNoIndexMetadata } from "@/lib/seo";

export const metadata: Metadata = buildNoIndexMetadata({
  title: "Subscription Management",
  description: "Update, pause, or change your Bunoraa subscription.",
  path: "/subscriptions/subscription/",
});

export default function SubscriptionManagementLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
