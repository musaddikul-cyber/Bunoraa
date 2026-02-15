import type { Metadata } from "next";
import { buildNoIndexMetadata } from "@/lib/seo";

export const metadata: Metadata = buildNoIndexMetadata({
  title: "My Preorders",
  description: "Review your Bunoraa preorder history and status.",
  path: "/preorders/my-orders/",
});

export default function MyPreordersLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
