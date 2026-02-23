import "server-only";

import { cache } from "react";
import { apiFetch } from "@/lib/api";
import { normalizeCategorySlugPath } from "@/lib/categoryPaths";

type CategoryTreeNode = {
  slug?: string | null;
  children?: CategoryTreeNode[] | null;
};

type PageSummary = {
  slug?: string | null;
};

const revalidate = 300;

const getCategoryTree = cache(async (): Promise<CategoryTreeNode[]> => {
  try {
    const response = await apiFetch<CategoryTreeNode[]>("/catalog/categories/tree/", {
      next: { revalidate },
      suppressError: true,
      suppressErrorStatus: [404],
    });
    return Array.isArray(response.data) ? response.data : [];
  } catch {
    return [];
  }
});

function matchesCategoryPath(nodes: CategoryTreeNode[], parts: string[], index = 0): boolean {
  const segment = parts[index];
  if (!segment) return false;

  const node = nodes.find((candidate) => {
    const slug = String(candidate?.slug || "").trim();
    return slug === segment;
  });
  if (!node) return false;
  if (index === parts.length - 1) return true;

  const children = Array.isArray(node.children) ? node.children : [];
  return matchesCategoryPath(children, parts, index + 1);
}

export async function categoryPathExists(slugPath: string): Promise<boolean> {
  const normalized = normalizeCategorySlugPath(slugPath);
  if (!normalized) return false;
  const tree = await getCategoryTree();
  if (!tree.length) return false;
  return matchesCategoryPath(tree, normalized.split("/"));
}

const getPublishedPageSlugs = cache(async (): Promise<Set<string>> => {
  try {
    const response = await apiFetch<PageSummary[]>("/pages/", {
      next: { revalidate },
      suppressError: true,
      suppressErrorStatus: [404],
    });
    const pages = Array.isArray(response.data) ? response.data : [];
    return new Set(
      pages
        .map((page) => String(page?.slug || "").trim())
        .filter(Boolean)
    );
  } catch {
    return new Set<string>();
  }
});

export async function publishedPageSlugExists(slug: string): Promise<boolean> {
  const normalized = String(slug || "").trim().replace(/^\/+|\/+$/g, "");
  if (!normalized || normalized.includes("/")) return false;
  const slugs = await getPublishedPageSlugs();
  return slugs.has(normalized);
}
