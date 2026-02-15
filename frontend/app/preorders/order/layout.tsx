import type { Metadata } from "next";
import { buildNoIndexMetadata } from "@/lib/seo";

export const metadata: Metadata = buildNoIndexMetadata({
  title: "Preorder Details",
  description: "View and manage your Bunoraa preorder details.",
  path: "/preorders/order/",
});

export default function PreorderDetailsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
