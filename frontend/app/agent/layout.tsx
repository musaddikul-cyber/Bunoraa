import type { Metadata } from "next";
import { buildNoIndexMetadata } from "@/lib/seo";

export const metadata: Metadata = buildNoIndexMetadata({
  title: "Agent Console",
  description: "Internal Bunoraa agent operations workspace.",
  path: "/agent/",
});

export default function AgentLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
