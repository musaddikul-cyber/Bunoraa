import { apiFetch, ApiError } from "@/lib/api";
import type { PageDetail } from "@/lib/types";
import { notFound } from "next/navigation";

export const revalidate = 900;

async function getAboutPage() {
  try {
    const response = await apiFetch<PageDetail>("/pages/pages/about/", {
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
  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-12">
      <h1 className="text-3xl font-semibold">{page.title}</h1>
      <div
        className="prose prose-slate mt-6 max-w-none"
        dangerouslySetInnerHTML={{ __html: page.content || "" }}
      />
    </div>
  );
}
