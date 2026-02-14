import Link from "next/link";
import Image from "next/image";
import { apiFetch, ApiError } from "@/lib/api";
import type { PreorderCategory, PreorderOption } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { formatMoney } from "@/lib/checkout";
import { notFound } from "next/navigation";

export const revalidate = 600;

async function getCategory(slug: string) {
  try {
    const response = await apiFetch<PreorderCategory>(`/preorders/categories/${slug}/`, {
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

export default async function PreorderCategoryPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const category = await getCategory(slug);

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 sm:py-10">
      <div className="grid gap-4 sm:gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <Card variant="bordered" className="space-y-4 p-4 sm:p-5">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-2">
              <h1 className="text-xl font-semibold sm:text-2xl">{category.name}</h1>
              <p className="text-sm text-foreground/70">
                {category.description}
              </p>
              <div className="flex flex-wrap gap-3 text-xs text-foreground/60">
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
            </div>
            {category.image ? (
              <div className="relative h-16 w-16 shrink-0 overflow-hidden rounded-2xl bg-muted sm:h-20 sm:w-20">
                <Image
                  src={category.image}
                  alt={category.name}
                  fill
                  className="object-cover"
                />
              </div>
            ) : null}
          </div>
          <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:gap-3">
            <Button asChild variant="primary-gradient" className="w-full sm:w-auto">
              <Link href={`/preorders/create/1/?category=${category.slug}`}>
                Start preorder
              </Link>
            </Button>
            <Button asChild variant="secondary" className="w-full sm:w-auto">
              <Link href="/preorders/">Back to categories</Link>
            </Button>
          </div>
        </Card>
        <Card variant="glass" className="space-y-3 p-4 sm:p-5">
          <h2 className="text-lg font-semibold">Requirements</h2>
          <ul className="space-y-2 text-sm text-foreground/70">
            <li>Minimum quantity: {category.min_quantity || 1}</li>
            <li>Maximum quantity: {category.max_quantity || "Open"}</li>
            <li>
              Requires design files: {category.requires_design ? "Yes" : "No"}
            </li>
            <li>
              Approval required: {category.requires_approval ? "Yes" : "No"}
            </li>
            {category.allow_rush_order ? (
              <li>Rush fee: {category.rush_order_fee_percentage}%</li>
            ) : null}
          </ul>
        </Card>
      </div>

      {category.options?.length ? (
        <div className="mt-8 space-y-4 sm:mt-10">
          <h2 className="text-xl font-semibold">Available options</h2>
          <div className="grid gap-3 sm:gap-4 md:grid-cols-2">
            {category.options.map((option: PreorderOption) => (
              <Card key={option.id} variant="bordered" className="p-4">
                <p className="text-sm font-semibold">{option.name}</p>
                <p className="text-xs text-foreground/60">
                  {option.description || option.help_text}
                </p>
                <p className="mt-2 text-xs text-foreground/60">
                  Type: {option.option_type}
                  {option.is_required ? " - Required" : ""}
                </p>
              </Card>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
