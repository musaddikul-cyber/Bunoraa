import { apiFetch } from "@/lib/api";
import { getServerLocaleHeaders } from "@/lib/serverLocale";
import type { SiteSettings } from "@/lib/types";
import { LoadingScreen } from "@/components/ui/LoadingScreen";

const pickText = (...values: Array<string | null | undefined>) => {
  for (const value of values) {
    if (value && value.trim()) return value.trim();
  }
  return "";
};

const getTimeGreeting = (date: Date) => {
  const hour = date.getHours();
  if (hour < 5) return "Good night";
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  if (hour < 21) return "Good evening";
  return "Good night";
};

async function getSiteSettings(headers: Record<string, string>) {
  try {
    const response = await apiFetch<SiteSettings>("/pages/settings/", {
      headers,
      next: { revalidate: 300 },
    });
    return response.data;
  } catch {
    return null;
  }
}

export default async function Loading() {
  const localeHeaders = await getServerLocaleHeaders();
  const siteSettings = await getSiteSettings(localeHeaders);
  const brandName = pickText(siteSettings?.site_name) || "Bunoraa";
  const directSubtitle = pickText(
    siteSettings?.site_tagline,
    siteSettings?.tagline,
    siteSettings?.site_description
  );

  const now = new Date();
  const greeting = getTimeGreeting(now);
  const dynamicSubtitles = [
    `${greeting}. Preparing ${brandName} for you.`,
    `Curating artisan finds for ${brandName}.`,
    "Loading fresh collections and stories.",
    "Assembling the latest handcrafted pieces.",
    `Setting up ${brandName} with care.`,
  ];
  const subtitle =
    directSubtitle || dynamicSubtitles[now.getMinutes() % dynamicSubtitles.length];

  return <LoadingScreen fullScreen title={brandName} subtitle={subtitle} />;
}
