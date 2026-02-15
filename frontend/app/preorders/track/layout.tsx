import type { Metadata } from "next";
import { buildNoIndexMetadata } from "@/lib/seo";

export const metadata: Metadata = buildNoIndexMetadata({
  title: "Track Preorder",
  description: "Track your Bunoraa preorder status.",
  path: "/preorders/track/",
});

export default function PreorderTrackLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
