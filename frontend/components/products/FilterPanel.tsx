"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import type { ProductFilterResponse } from "@/lib/types";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import {
  parseFilters,
  toggleMultiValue,
  updateParamValue,
} from "@/lib/productFilters";
import { cn } from "@/lib/utils";
import { formatMoney } from "@/lib/money";
import { getStoredLocale } from "@/lib/locale";

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
  filterParams,
}: {
  filters: ProductFilterResponse | null;
  facets?: CategoryFacet[];
  className?: string;
  filterParams?: Record<string, string>;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [activeFilters, setActiveFilters] = React.useState<ProductFilterResponse | null>(filters);
  const [preferredCurrency, setPreferredCurrency] = React.useState<string | undefined>();
  const [activeHandle, setActiveHandle] = React.useState<"min" | "max" | null>(null);

  React.useEffect(() => {
    setActiveFilters(filters);
  }, [filters]);

  React.useEffect(() => {
    setPreferredCurrency(getStoredLocale().currency);
  }, []);

  const paramsKey = React.useMemo(() => JSON.stringify(filterParams || {}), [filterParams]);

  React.useEffect(() => {
    let cancelled = false;
    const params = JSON.parse(paramsKey) as Record<string, string>;
    apiFetch<ProductFilterResponse>("/catalog/products/filters/", {
      params,
      suppressError: true,
    })
      .then((response) => {
        if (!cancelled) {
          setActiveFilters(response.data);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [paramsKey, preferredCurrency]);
  const parseNumber = (value: string | number | null | undefined, fallback: number) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };
  const current = parseFilters(searchParams);
  const minRange = Math.max(0, parseNumber(activeFilters?.price_range?.min, 0));
  const maxRange = Math.max(
    minRange,
    parseNumber(activeFilters?.price_range?.max, minRange)
  );
  const sliderMax = maxRange <= minRange ? minRange + 1 : maxRange;
  const currencyCode = activeFilters?.price_range?.currency || "USD";
  const rangeSpan = Math.max(0, maxRange - minRange);
  const clampValue = (value: number, min: number, max: number) =>
    Math.min(Math.max(value, min), max);
  const clampPercent = (value: number) => Math.min(100, Math.max(0, value));
  const percentFromValue = React.useCallback(
    (value: number) => {
      if (rangeSpan <= 0) return 0;
      return ((value - minRange) / rangeSpan) * 100;
    },
    [minRange, rangeSpan]
  );
  const valueFromPercent = (percent: number) => {
    if (rangeSpan <= 0) return minRange;
    return minRange + (rangeSpan * percent) / 100;
  };
  const [minPercentValue, setMinPercentValue] = React.useState(0);
  const [maxPercentValue, setMaxPercentValue] = React.useState(100);

  React.useEffect(() => {
    const nextMin = clampValue(parseNumber(current.priceMin, minRange), minRange, maxRange);
    const nextMax = clampValue(parseNumber(current.priceMax, maxRange), minRange, maxRange);
    const safeMin = Math.min(nextMin, nextMax);
    const safeMax = Math.max(nextMin, nextMax);
    setMinPercentValue(clampPercent(Math.round(percentFromValue(safeMin))));
    setMaxPercentValue(clampPercent(Math.round(percentFromValue(safeMax))));
  }, [current.priceMin, current.priceMax, minRange, maxRange, rangeSpan, percentFromValue]);

  const applyPrice = () => {
    if (rangeSpan <= 0) return;
    const rawMin = valueFromPercent(Math.min(minPercentValue, maxPercentValue));
    const rawMax = valueFromPercent(Math.max(minPercentValue, maxPercentValue));
    const safeMin = clampValue(Number(rawMin.toFixed(2)), minRange, sliderMax);
    const safeMax = clampValue(Number(rawMax.toFixed(2)), minRange, sliderMax);
    let params = updateParamValue(searchParams, "price_min", String(safeMin));
    params = updateParamValue(params, "price_max", String(safeMax));
    router.push(`?${params.toString()}`);
  };
  const minPercent = minPercentValue;
  const maxPercent = maxPercentValue;
  const rangeDisabled = !Number.isFinite(minRange) || !Number.isFinite(maxRange) || rangeSpan <= 0;
  const minOnTop =
    minPercentValue > maxPercentValue - 5;
  const minZ = activeHandle === "min" || minOnTop ? "z-30" : "z-10";
  const maxZ = activeHandle === "max" ? "z-30" : "z-20";

  const attributeGroups = React.useMemo(() => {
    const groups: Array<{ name: string; slug: string; values: Array<{ value: string; count?: number }> }> = [];
    if (activeFilters?.attributes) {
      Object.entries(activeFilters.attributes).forEach(([name, info]) => {
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
  }, [activeFilters, facets]);

  return (
    <div className={cn("space-y-6", className)}>
      <Section title="Price range">
        <div className="space-y-3">
          <div className="relative h-4">
            <div className="pointer-events-none absolute inset-x-0 top-1/2 h-2 -translate-y-1/2 rounded-full bg-muted" />
            <div
              className="pointer-events-none absolute inset-x-0 top-1/2 h-2 -translate-y-1/2 rounded-full bg-primary/30"
              style={{ left: `${minPercent}%`, right: `${100 - maxPercent}%` }}
            />
            <input
              type="range"
              min={0}
              max={100}
              step={1}
              value={minPercentValue}
              disabled={rangeDisabled}
              onPointerDown={() => setActiveHandle("min")}
              onPointerUp={() => setActiveHandle(null)}
              onChange={(event) => {
                const nextPercent = clampPercent(Number(event.target.value));
                setMinPercentValue(Math.min(nextPercent, maxPercentValue));
              }}
              onMouseUp={applyPrice}
              onTouchEnd={applyPrice}
              aria-label="Minimum price"
              className={cn(
                "range-slider range-slider-min absolute inset-0 h-4 w-full cursor-pointer bg-transparent",
                minZ
              )}
            />
            <input
              type="range"
              min={0}
              max={100}
              step={1}
              value={maxPercentValue}
              disabled={rangeDisabled}
              onPointerDown={() => setActiveHandle("max")}
              onPointerUp={() => setActiveHandle(null)}
              onChange={(event) => {
                const nextPercent = clampPercent(Number(event.target.value));
                setMaxPercentValue(Math.max(nextPercent, minPercentValue));
              }}
              onMouseUp={applyPrice}
              onTouchEnd={applyPrice}
              aria-label="Maximum price"
              className={cn(
                "range-slider range-slider-max absolute inset-0 h-4 w-full cursor-pointer bg-transparent",
                maxZ
              )}
            />
          </div>
          <div className="flex items-center justify-between text-xs text-foreground/60">
            <span>Min {formatMoney(minRange, currencyCode)}</span>
            <span>Max {formatMoney(maxRange, currencyCode)}</span>
          </div>
        </div>
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
        {activeFilters?.has_on_sale ? (
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

      {activeFilters?.tags?.length ? (
        <Section title="Tags">
          <div className="flex flex-wrap gap-2">
            {activeFilters.tags.map((tag) => {
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
