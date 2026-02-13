import { apiFetch, ApiError } from "@/lib/api";
import type { PageDetail } from "@/lib/types";
import { notFound } from "next/navigation";
import { JsonLd } from "@/components/seo/JsonLd";
import { absoluteUrl, buildBreadcrumbList, cleanObject } from "@/lib/seo";

export const revalidate = 300;

async function getPage(slug: string) {
  try {
    const response = await apiFetch<PageDetail>(`/pages/${slug}/`, {
      next: { revalidate },
    });
    return response.data;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const page = await getPage(slug);
  return {
    title: page.meta_title || page.title,
    description: page.meta_description || page.excerpt || "",
  };
}

export default async function PageDetail({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const page = await getPage(slug);
  const pageUrl = `/pages/${page.slug}/`;
  const pageSchema = cleanObject({
    "@context": "https://schema.org",
    "@type": "WebPage",
    name: page.meta_title || page.title,
    description: page.meta_description || page.excerpt || undefined,
    url: absoluteUrl(pageUrl),
  });
  const breadcrumbs = buildBreadcrumbList([
    { name: "Home", url: "/" },
    { name: "Pages", url: "/pages/" },
    { name: page.title, url: pageUrl },
  ]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-4xl px-6 py-12">
        <h1 className="text-3xl font-semibold text-foreground">{page.title}</h1>
        <div
          className="prose mt-6 max-w-none text-foreground prose-headings:text-foreground prose-p:text-foreground/80 prose-li:text-foreground/80 prose-strong:text-foreground prose-a:text-primary prose-a:underline prose-a:underline-offset-4 prose-hr:border-border"
          dangerouslySetInnerHTML={{ __html: page.content || "" }}
        />
      </div>
      <JsonLd data={[pageSchema, breadcrumbs]} />
    </div>
  );
}
