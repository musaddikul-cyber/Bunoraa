import { cache } from "react";
import { apiFetch } from "@/lib/api";
import type { Bundle } from "@/lib/types";
import { asArray } from "@/lib/array";

const BUNDLE_REVALIDATE_SECONDS = 600;

export const hasPublishedBundles = cache(async (): Promise<boolean> => {
  try {
    const response = await apiFetch<Bundle[] | { results?: Bundle[]; count?: number }>(
      "/catalog/bundles/",
      {
        params: { page_size: 1 },
        next: { revalidate: BUNDLE_REVALIDATE_SECONDS },
      }
    );
    const metaCount = response.meta?.pagination?.count;
    if (typeof metaCount === "number") {
      return metaCount > 0;
    }
    const payload = response.data;
    if (
      payload &&
      typeof payload === "object" &&
      !Array.isArray(payload) &&
      typeof (payload as { count?: unknown }).count === "number"
    ) {
      return ((payload as { count: number }).count || 0) > 0;
    }
    return asArray<Bundle>(payload).length > 0;
  } catch {
    return false;
  }
});
