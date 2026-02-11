import { apiFetch, ApiError } from "@/lib/api";
import type { PageDetail } from "@/lib/types";
import { notFound } from "next/navigation";
import { JsonLd } from "@/components/seo/JsonLd";
import { absoluteUrl, cleanObject } from "@/lib/seo";

export const revalidate = 900;

async function getAboutPage() {
  try {
    const response = await apiFetch<PageDetail>("/pages/about/", {
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

export default async function AboutPage() {
  const page = await getAboutPage();
  const pageSchema = cleanObject({
    "@context": "https://schema.org",
    "@type": "AboutPage",
    name: page.meta_title || page.title,
    description: page.meta_description || page.excerpt || undefined,
    url: absoluteUrl("/about/"),
  });
  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-12">
      <h1 className="text-3xl font-semibold">{page.title}</h1>
      <div
        className="prose prose-slate mt-6 max-w-none"
        dangerouslySetInnerHTML={{ __html: page.content || "" }}
      />
      <JsonLd data={pageSchema} />
    </div>
  );
}
