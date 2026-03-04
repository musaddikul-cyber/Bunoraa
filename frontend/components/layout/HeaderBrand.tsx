"use client";

import * as React from "react";
import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { SiteSettings } from "@/lib/types";

const SCROLL_SWAP_OFFSET = 24;
const DESKTOP_BREAKPOINT_QUERY = "(min-width: 1024px)";

type HeaderBrandProps = {
  defaultBrandName: string;
  defaultFaviconUrl?: string | null;
};

function pickText(...values: Array<string | null | undefined>) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

export function HeaderBrand({ defaultBrandName, defaultFaviconUrl }: HeaderBrandProps) {
  const staticBrandName = pickText(defaultBrandName);
  const staticFaviconUrl = pickText(defaultFaviconUrl);
  const hasStaticBrandName = Boolean(staticBrandName);
  const hasStaticFaviconUrl = Boolean(staticFaviconUrl);

  const [isDesktop, setIsDesktop] = React.useState(false);
  const [isScrolled, setIsScrolled] = React.useState(false);
  const [brandName, setBrandName] = React.useState(staticBrandName || "Bunoraa");
  const [faviconUrl, setFaviconUrl] = React.useState(staticFaviconUrl);
  const [settingsFaviconUrl, setSettingsFaviconUrl] = React.useState("");
  const [usingSettingsFavicon, setUsingSettingsFavicon] = React.useState(!hasStaticFaviconUrl);
  const [staticFaviconFailed, setStaticFaviconFailed] = React.useState(false);

  React.useEffect(() => {
    let active = true;

    const loadSiteSettings = async () => {
      try {
        const response = await apiFetch<SiteSettings>("/pages/settings/");
        if (!active || !response?.data) return;

        const nextBrandName = pickText(response.data.site_name);
        const nextFavicon = pickText(response.data.favicon);

        if (!hasStaticBrandName && nextBrandName) {
          setBrandName(nextBrandName);
        }

        if (nextFavicon) {
          setSettingsFaviconUrl(nextFavicon);
        }

        if (nextFavicon && (!hasStaticFaviconUrl || staticFaviconFailed)) {
          setFaviconUrl(nextFavicon);
          setUsingSettingsFavicon(true);
        }
      } catch {
        // Keep static defaults on API failure.
      }
    };

    void loadSiteSettings();

    return () => {
      active = false;
    };
  }, [hasStaticBrandName, hasStaticFaviconUrl, staticFaviconFailed]);

  React.useEffect(() => {
    const media = window.matchMedia(DESKTOP_BREAKPOINT_QUERY);

    const syncDesktopState = () => {
      setIsDesktop(media.matches);
    };

    const syncScrollState = () => {
      setIsScrolled(window.scrollY > SCROLL_SWAP_OFFSET);
    };

    syncDesktopState();
    syncScrollState();

    media.addEventListener("change", syncDesktopState);
    window.addEventListener("scroll", syncScrollState, { passive: true });

    return () => {
      media.removeEventListener("change", syncDesktopState);
      window.removeEventListener("scroll", syncScrollState);
    };
  }, []);

  const showFavicon = isDesktop && isScrolled && Boolean(faviconUrl);

  const handleFaviconError = () => {
    if (!usingSettingsFavicon) {
      setStaticFaviconFailed(true);
      if (settingsFaviconUrl) {
        setFaviconUrl(settingsFaviconUrl);
        setUsingSettingsFavicon(true);
        return;
      }
    }
    setFaviconUrl("");
  };

  return (
    <Link
      href="/"
      aria-label={brandName}
      className="inline-flex h-10 items-center text-xl font-bold sm:text-2xl lg:text-3xl"
    >
      {showFavicon ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={faviconUrl}
          alt={brandName}
          className="h-8 w-8 rounded-sm object-contain lg:h-9 lg:w-9"
          onError={handleFaviconError}
        />
      ) : (
        <span>{brandName}</span>
      )}
    </Link>
  );
}
