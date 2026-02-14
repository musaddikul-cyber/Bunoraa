import Link from "next/link";
import Image from "next/image";
import { apiFetch } from "@/lib/api";
import type { PreorderCategory } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { formatMoney } from "@/lib/checkout";
import { asArray } from "@/lib/array";

export const revalidate = 600;

async function getCategories() {
  const response = await apiFetch<PreorderCategory[]>("/preorders/categories/", {
    next: { revalidate },
  });
  return asArray<PreorderCategory>(response.data);
}

export default async function PreordersLandingPage() {
  const categories = await getCategories();

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 sm:py-12">
      <div className="mb-8 grid gap-4 sm:gap-6 lg:mb-10 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="space-y-4">
          <p className="text-xs uppercase tracking-[0.3em] text-foreground/60">
            Preorders
          </p>
          <h1 className="text-2xl font-semibold sm:text-4xl">
            Start a custom preorder
          </h1>
          <p className="text-sm text-foreground/60">
            Submit your custom request, get a production timeline, and receive a
            dedicated quote. We will guide you through approvals, payments, and
            delivery.
          </p>
          <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:gap-3">
            <Button asChild className="w-full sm:w-auto" variant="primary-gradient">
              <Link href="/preorders/create/1/">Start preorder</Link>
            </Button>
            <Button asChild className="w-full sm:w-auto" variant="secondary">
              <Link href="/preorders/track/">Track preorder</Link>
            </Button>
          </div>
        </div>
        <Card variant="modern-gradient" className="space-y-3 p-4 sm:p-5">
          <h2 className="text-lg font-semibold">How it works</h2>
          <ul className="space-y-2 text-sm text-foreground/70">
            <li>Share details, options, and reference files.</li>
            <li>Receive a quote with production timeline.</li>
            <li>Approve, pay deposit, and track progress.</li>
            <li>Finalize delivery with updates and messaging.</li>
          </ul>
        </Card>
      </div>
      <div className="grid gap-4 sm:gap-6 md:grid-cols-2 lg:grid-cols-3">
        {categories.map((category) => (
          <Card key={category.id} variant="bordered" className="p-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">{category.name}</h2>
                <p className="text-sm text-foreground/70">
                  {category.description}
                </p>
              </div>
              {category.image ? (
                <div className="relative h-14 w-14 overflow-hidden rounded-xl bg-muted">
                  <Image
                    src={category.image}
                    alt={category.name}
                    fill
                    className="object-cover"
                  />
                </div>
              ) : null}
            </div>
            <div className="mt-4 flex flex-wrap gap-3 text-xs text-foreground/60">
              {category.base_price ? (
                <span>From {formatMoney(category.base_price, "BDT")}</span>
              ) : null}
              {category.deposit_percentage ? (
                <span>Deposit {category.deposit_percentage}%</span>
              ) : null}
              {category.min_production_days ? (
                <span>
                  {category.min_production_days}-{category.max_production_days} days
                </span>
              ) : null}
            </div>
            <div className="mt-4">
              <Button asChild variant="secondary" size="sm" className="w-full sm:w-auto">
                <Link href={`/preorders/category/${category.slug}/`}>View details</Link>
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
