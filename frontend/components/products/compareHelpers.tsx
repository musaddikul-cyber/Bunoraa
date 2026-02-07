"use client";

import * as React from "react";
import type { ProductListItem } from "@/lib/types";
import type { CompareItem } from "@/lib/compare";
import { toggleCompareItem, isInCompare } from "@/lib/compare";

export function compareItemFromProduct(product: ProductListItem): CompareItem {
  const image =
    typeof product.primary_image === "string"
      ? product.primary_image
      : (product.primary_image as unknown as { image?: string | null })?.image || null;
  const fallbackImage =
    (product as { images?: Array<{ image?: string | null }> }).images?.[0]?.image || null;
  return {
    id: product.id,
    slug: product.slug,
    name: product.name,
    primary_image: image || fallbackImage,
    current_price: product.current_price,
    currency: product.currency,
    average_rating: product.average_rating,
    reviews_count: product.reviews_count,
    is_in_stock: product.is_in_stock,
    primary_category_name: product.primary_category_name,
  };
}

export function useCompareToggle(product: ProductListItem) {
  const [inCompare, setInCompare] = React.useState(false);

  const refresh = React.useCallback(() => {
    setInCompare(isInCompare(product.id));
  }, [product.id]);

  React.useEffect(() => {
    refresh();
    const handler = () => refresh();
    window.addEventListener("compare-updated", handler);
    return () => window.removeEventListener("compare-updated", handler);
  }, [refresh]);

  const toggleCompare = React.useCallback(
    (item: CompareItem) => {
      const next = toggleCompareItem(item);
      setInCompare(next);
    },
    []
  );

  return { isInCompare: inCompare, toggleCompare };
}
