import type { Metadata } from "next";
import { buildNoIndexMetadata } from "@/lib/seo";

export const metadata: Metadata = buildNoIndexMetadata({
  title: "Create Preorder",
  description: "Submit custom preorder requirements on Bunoraa.",
  path: "/preorders/create/",
});

export default function CreatePreorderLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
