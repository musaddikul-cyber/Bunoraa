import type { Metadata } from "next";
import { buildNoIndexMetadata } from "@/lib/seo";

export const metadata: Metadata = buildNoIndexMetadata({
  title: "Bag",
  description: "Review your Bunoraa bag items.",
  path: "/cart/",
});

export default function CartLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
