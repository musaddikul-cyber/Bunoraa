import * as React from "react";
import type { ProductBadge, ProductListItem } from "@/lib/types";
import { cn } from "@/lib/utils";

export function getDefaultBadges(
  product: ProductListItem,
  options?: { omitOnSale?: boolean }
) {
  const badges: Array<{ label: string; tone: "primary" | "accent" | "secondary" }> = [];
  if (product.is_on_sale && !options?.omitOnSale) {
    badges.push({ label: "On sale", tone: "accent" });
  }
  if (product.is_new_arrival) badges.push({ label: "New", tone: "primary" });
  if (product.is_bestseller) badges.push({ label: "Bestseller", tone: "secondary" });
  if (!product.is_in_stock) badges.push({ label: "Out of stock", tone: "secondary" });
  return badges;
}

export function ProductBadges({
  product,
  badges,
  className,
  omitOnSale = false,
}: {
  product?: ProductListItem;
  badges?: ProductBadge[] | null;
  className?: string;
  omitOnSale?: boolean;
}) {
  const defaultBadges = product ? getDefaultBadges(product, { omitOnSale }) : [];
  const extraBadges = (badges || []).map((badge) => ({
    label: badge.name,
    tone: "primary" as const,
  }));
  const all = [...extraBadges, ...defaultBadges];

  if (!all.length) return null;

  return (
    <div className={cn("flex flex-wrap gap-2", className)}>
      {all.map((badge, index) => (
        <span
          key={`${badge.label}-${index}`}
          className={cn(
            "rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em]",
            badge.tone === "accent"
              ? "bg-accent-500/15 text-accent-600"
              : badge.tone === "secondary"
              ? "bg-secondary-200 text-secondary-700"
              : "bg-primary-500/15 text-primary-700"
          )}
        >
          {badge.label}
        </span>
      ))}
    </div>
  );
}
