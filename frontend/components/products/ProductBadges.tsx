import * as React from "react";
import type { ProductBadge, ProductListItem } from "@/lib/types";
import { cn } from "@/lib/utils";

type BadgeTone = "primary" | "accent" | "secondary";

type RenderBadge = {
  key: string;
  label: string;
  tone: BadgeTone;
  className?: string | null;
};

const normalizeBadgeKey = (value?: string | null) =>
  (value || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-");

const inferTone = (slug?: string | null, label?: string | null): BadgeTone => {
  const source = `${slug || ""} ${label || ""}`.toLowerCase();
  if (
    source.includes("sale") ||
    source.includes("discount") ||
    source.includes("deal") ||
    source.includes("offer")
  ) {
    return "accent";
  }
  if (
    source.includes("stock") ||
    source.includes("sold-out") ||
    source.includes("soldout") ||
    source.includes("bestseller") ||
    source.includes("best-seller")
  ) {
    return "secondary";
  }
  return "primary";
};

export function getDefaultBadges(
  product: ProductListItem,
  options?: { omitOnSale?: boolean }
) {
  const badges: RenderBadge[] = [];
  if (product.is_on_sale && !options?.omitOnSale) {
    badges.push({ key: "on-sale", label: "On sale", tone: "accent" });
  }
  if (product.is_new_arrival) badges.push({ key: "new", label: "New", tone: "primary" });
  if (product.is_bestseller) {
    badges.push({ key: "bestseller", label: "Bestseller", tone: "secondary" });
  }
  if (!product.is_in_stock) {
    badges.push({ key: "out-of-stock", label: "Out of stock", tone: "secondary" });
  }
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
  const extraBadges: RenderBadge[] = (badges || [])
    .map((badge) => ({
      key: normalizeBadgeKey(badge.slug || badge.name),
      label: badge.name,
      tone: inferTone(badge.slug, badge.name),
      className: badge.css_class,
    }))
    .filter((badge) => Boolean(badge.key && badge.label));
  const byKey = new Map<string, RenderBadge>();
  for (const badge of extraBadges) byKey.set(badge.key, badge);
  for (const badge of defaultBadges) {
    if (!byKey.has(badge.key)) byKey.set(badge.key, badge);
  }
  const all = Array.from(byKey.values());

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
              : "bg-primary-500/15 text-primary-700",
            badge.className
          )}
        >
          {badge.label}
        </span>
      ))}
    </div>
  );
}
