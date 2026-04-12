import type { Metadata } from "next";
import { apiFetch, ApiError } from "@/lib/api";
import type { PageDetail } from "@/lib/types";
import { notFound } from "next/navigation";
import { JsonLd } from "@/components/seo/JsonLd";
import { absoluteUrl, buildBreadcrumbList, buildPageMetadata, cleanObject } from "@/lib/seo";
import {
  buildCategoryMetadataForPath,
  renderCategoryPageForPath,
  type CategorySearchParams,
} from "@/app/categories/[...slug]/categoryPageShared";
import { categoryPathExists, publishedPageSlugExists } from "@/lib/routeLookup";

export const revalidate = 300;

async function getPage(slug: string) {
  try {
    const response = await apiFetch<PageDetail>(`/pages/${slug}/`, {
      next: { revalidate },
      suppressError: true,
      suppressErrorStatus: [404],
    });
    return response.data;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null;
    }
    throw error;
  }
}

export async function generateMetadata({
  params,
  searchParams,
}: {
  params: Promise<{ slug: string }>;
  searchParams: Promise<CategorySearchParams>;
}): Promise<Metadata> {
  const [{ slug }, resolvedSearchParams] = await Promise.all([params, searchParams]);
  if (await categoryPathExists(slug)) {
    return buildCategoryMetadataForPath(slug, resolvedSearchParams);
  }

  if (!(await publishedPageSlugExists(slug))) {
    return buildPageMetadata({
      title: "Page",
      description: "Read this page on Bunoraa.",
      path: `/${slug}/`,
    });
  }

  const page = await getPage(slug);
  if (!page) {
    return buildPageMetadata({
      title: "Page",
      description: "Read this page on Bunoraa.",
      path: `/${slug}/`,
    });
  }

  return buildPageMetadata({
    title: page.meta_title || page.title,
    description: page.meta_description || page.excerpt || "Read this page on Bunoraa.",
    path: `/${page.slug}/`,
  });
}

export default async function PageDetail({
  params,
  searchParams,
}: {
  params: Promise<{ slug: string }>;
  searchParams: Promise<CategorySearchParams>;
}) {
  const [{ slug }, resolvedSearchParams] = await Promise.all([params, searchParams]);
  if (await categoryPathExists(slug)) {
    return renderCategoryPageForPath(slug, resolvedSearchParams);
  }

  if (!(await publishedPageSlugExists(slug))) {
    notFound();
  }

  const page = await getPage(slug);
  if (!page) {
    notFound();
  }

  const pageUrl = `/${page.slug}/`;
  const pageSchema = cleanObject({
    "@context": "https://schema.org",
    "@type": "WebPage",
    name: page.meta_title || page.title,
    description: page.meta_description || page.excerpt || undefined,
    url: absoluteUrl(pageUrl),
  });
  const breadcrumbs = buildBreadcrumbList([
    { name: "Home", url: "/" },
    { name: page.title, url: pageUrl },
  ]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-4xl px-3 sm:px-5 py-12">
        <h1 className="text-3xl font-semibold">{page.title}</h1>
        <div
          className="prose prose-stone mt-6 max-w-none"
          dangerouslySetInnerHTML={{ __html: page.content || "" }}
        />
      </div>
      <JsonLd data={[pageSchema, breadcrumbs]} />
    </div>
  );
}
