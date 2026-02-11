import Link from "next/link";
import { apiFetch } from "@/lib/api";
import { Card } from "@/components/ui/Card";

export const revalidate = 300;

type PageSummary = {
  id: string;
  title: string;
  slug: string;
  excerpt?: string | null;
};

async function getPages() {
  const response = await apiFetch<PageSummary[]>("/pages/", {
    next: { revalidate },
  });
  return response.data;
}

export default async function PagesIndex() {
  const pages = await getPages();

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-4xl px-6 py-12">
        <h1 className="text-3xl font-semibold">Pages</h1>
        <div className="mt-6 space-y-4">
          {pages.map((page) => (
            <Card key={page.id} variant="bordered" className="space-y-2">
              <Link className="text-lg font-semibold" href={`/pages/${page.slug}/`}>
                {page.title}
              </Link>
              {page.excerpt ? (
                <p className="text-sm text-foreground/70">{page.excerpt}</p>
              ) : null}
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
