import { cache } from "react";
import { apiFetch } from "@/lib/api";
import type { SiteSettings } from "@/lib/types";

const SITE_SETTINGS_REVALIDATE_SECONDS = 600;

export async function fetchSiteSettings(): Promise<SiteSettings | null> {
  const response = await apiFetch<SiteSettings>("/pages/settings/");
  return response.data;
}

export const getSiteSettings = cache(async (): Promise<SiteSettings | null> => {
  try {
    const response = await apiFetch<SiteSettings>("/pages/settings/", {
      next: { revalidate: SITE_SETTINGS_REVALIDATE_SECONDS },
    });
    return response.data;
  } catch {
    return null;
  }
});
