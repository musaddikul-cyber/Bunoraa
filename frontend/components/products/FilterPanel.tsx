"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import type { ProductFilterResponse } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import {
  parseFilters,
  toggleMultiValue,
  updateParamValue,
} from "@/lib/productFilters";
import { cn } from "@/lib/utils";

export type CategoryFacet = {
  id: string;
  name: string;
  slug: string;
  type?: string;
  values?: Array<{ value: string; display_value?: string }>;
  value_counts?: Array<{ value: string; count: number }>;
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold">{title}</h3>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

export function FilterPanel({
  filters,
  facets,
  className,
}: {
  filters: ProductFilterResponse | null;
  facets?: CategoryFacet[];
  className?: string;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const current = parseFilters(searchParams);
  const [priceMin, setPriceMin] = React.useState(current.priceMin || "");
  const [priceMax, setPriceMax] = React.useState(current.priceMax || "");

  React.useEffect(() => {
    setPriceMin(current.priceMin || "");
    setPriceMax(current.priceMax || "");
  }, [current.priceMin, current.priceMax]);

  const applyPrice = () => {
    let params = updateParamValue(searchParams, "price_min", priceMin || null);
    params = updateParamValue(params, "price_max", priceMax || null);
    router.push(`?${params.toString()}`);
  };

  const attributeGroups = React.useMemo(() => {
    const groups: Array<{ name: string; slug: string; values: Array<{ value: string; count?: number }> }> = [];
    if (filters?.attributes) {
      Object.entries(filters.attributes).forEach(([name, info]) => {
        groups.push({
          name,
          slug: info.slug,
          values: info.values.map((value) => ({ value })),
        });
      });
    }
    if (facets && facets.length) {
      facets.forEach((facet) => {
        const values = facet.value_counts
          ? facet.value_counts.map((item) => ({ value: item.value, count: item.count }))
          : (facet.values || []).map((item) => ({
              value: typeof item === "string" ? item : item.value,
            }));
        groups.push({ name: facet.name, slug: facet.slug, values });
      });
    }
    const bySlug: Record<string, { name: string; slug: string; values: Array<{ value: string; count?: number }> }> = {};
    groups.forEach((group) => {
      if (!bySlug[group.slug]) {
        bySlug[group.slug] = { ...group };
      } else {
        const merged = new Map(bySlug[group.slug].values.map((item) => [item.value, item]));
        group.values.forEach((item) => merged.set(item.value, item));
        bySlug[group.slug].values = Array.from(merged.values());
      }
    });
    return Object.values(bySlug);
  }, [filters, facets]);

  return (
    <div className={cn("space-y-6", className)}>
      <Section title="Price range">
        <div className="flex items-center gap-2">
          <input
            type="number"
            placeholder="Min"
            value={priceMin}
            onChange={(event) => setPriceMin(event.target.value)}
            className="h-10 w-full rounded-lg border border-border bg-card px-3 text-sm"
          />
          <input
            type="number"
            placeholder="Max"
            value={priceMax}
            onChange={(event) => setPriceMax(event.target.value)}
            className="h-10 w-full rounded-lg border border-border bg-card px-3 text-sm"
          />
        </div>
        <Button size="sm" variant="secondary" onClick={applyPrice}>
          Apply
        </Button>
      </Section>

      <Section title="Availability">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            className="h-4 w-4 rounded border-border text-primary"
            checked={current.inStock}
            onChange={(event) => {
              const params = updateParamValue(
                searchParams,
                "in_stock",
                event.target.checked ? "true" : null
              );
              router.push(`?${params.toString()}`);
            }}
          />
          In stock only
        </label>
        {filters?.has_on_sale ? (
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="h-4 w-4 rounded border-border text-primary"
              checked={current.onSale}
              onChange={(event) => {
                const params = updateParamValue(
                  searchParams,
                  "on_sale",
                  event.target.checked ? "true" : null
                );
                router.push(`?${params.toString()}`);
              }}
            />
            On sale
          </label>
        ) : null}
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            className="h-4 w-4 rounded border-border text-primary"
            checked={current.newArrivals}
            onChange={(event) => {
              const params = updateParamValue(
                searchParams,
                "new_arrivals",
                event.target.checked ? "true" : null
              );
              router.push(`?${params.toString()}`);
            }}
          />
          New arrivals
        </label>
      </Section>

      <Section title="Rating">
        {[4, 3, 2].map((rating) => (
          <label key={rating} className="flex items-center gap-2 text-sm">
            <input
              type="radio"
              name="min_rating"
              className="h-4 w-4 rounded border-border text-primary"
              checked={current.minRating === String(rating)}
              onChange={() => {
                const params = updateParamValue(searchParams, "min_rating", String(rating));
                router.push(`?${params.toString()}`);
              }}
            />
            {rating}+ stars
          </label>
        ))}
        {current.minRating ? (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              const params = updateParamValue(searchParams, "min_rating", null);
              router.push(`?${params.toString()}`);
            }}
          >
            Clear rating
          </Button>
        ) : null}
      </Section>

      {filters?.tags?.length ? (
        <Section title="Tags">
          <div className="flex flex-wrap gap-2">
            {filters.tags.map((tag) => {
              const isSelected = current.tags.includes(tag.name);
              return (
                <button
                  key={tag.slug}
                  type="button"
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs",
                    isSelected
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border text-foreground/70"
                  )}
                  onClick={() => {
                    const params = toggleMultiValue(searchParams, "tags", tag.name);
                    router.push(`?${params.toString()}`);
                  }}
                >
                  {tag.name}
                </button>
              );
            })}
          </div>
        </Section>
      ) : null}

      {attributeGroups.map((group) => (
        <Section key={group.slug} title={group.name}>
          <div className="flex flex-wrap gap-2">
            {group.values.map((item) => {
              const currentValues = current.attrs[group.slug] || [];
              const isSelected = currentValues.includes(item.value);
              return (
                <button
                  key={item.value}
                  type="button"
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs",
                    isSelected
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border text-foreground/70"
                  )}
                  onClick={() => {
                    const params = toggleMultiValue(
                      searchParams,
                      `attr_${group.slug}`,
                      item.value
                    );
                    router.push(`?${params.toString()}`);
                  }}
                >
                  {item.value}
                  {typeof item.count === "number" ? ` (${item.count})` : ""}
                </button>
              );
            })}
          </div>
        </Section>
      ))}
    </div>
  );
}
