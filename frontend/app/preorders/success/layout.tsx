import type { Metadata } from "next";
import { buildNoIndexMetadata } from "@/lib/seo";

export const metadata: Metadata = buildNoIndexMetadata({
  title: "Preorder Submitted",
  description: "Confirmation page for your Bunoraa preorder.",
  path: "/preorders/success/",
});

export default function PreorderSuccessLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
