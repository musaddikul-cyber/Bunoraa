import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { buildCategoryPath } from "@/lib/categoryPaths";
import {
  buildCategoryMetadataForPath,
  type CategorySearchParams,
} from "./categoryPageShared";

export const revalidate = 300;

export async function generateMetadata({
  params,
  searchParams,
}: {
  params: Promise<{ slug: string[] }>;
  searchParams: Promise<CategorySearchParams>;
}): Promise<Metadata> {
  const [{ slug }, resolvedSearchParams] = await Promise.all([params, searchParams]);
  return buildCategoryMetadataForPath(slug.join("/"), resolvedSearchParams);
}

export default async function CategoryPage({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  const slugPath = slug.join("/");
  redirect(buildCategoryPath(slugPath));
}
