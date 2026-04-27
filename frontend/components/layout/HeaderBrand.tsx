"use client";

import * as React from "react";
import Link from "next/link";

const SCROLL_SWAP_OFFSET = 24;
const DESKTOP_BREAKPOINT_QUERY = "(min-width: 1024px)";

type HeaderBrandProps = {
  defaultBrandName: string;
  defaultFaviconUrl?: string | null;
  fallbackStaticFaviconUrl?: string | null;
};

function pickText(...values: Array<string | null | undefined>) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

export function HeaderBrand({
  defaultBrandName,
  defaultFaviconUrl,
  fallbackStaticFaviconUrl,
}: HeaderBrandProps) {
  const brandName = pickText(defaultBrandName) || "Bunoraa";
  const staticFaviconUrl = pickText(defaultFaviconUrl);
  const staticFallbackFaviconUrl = pickText(fallbackStaticFaviconUrl);

  const [isDesktop, setIsDesktop] = React.useState(false);
  const [isScrolled, setIsScrolled] = React.useState(false);
  const [faviconUrl, setFaviconUrl] = React.useState(
    staticFaviconUrl || staticFallbackFaviconUrl
  );
  const [isUsingFallback, setIsUsingFallback] = React.useState(
    !staticFaviconUrl && Boolean(staticFallbackFaviconUrl)
  );

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
    if (!isUsingFallback && staticFallbackFaviconUrl) {
      setFaviconUrl(staticFallbackFaviconUrl);
      setIsUsingFallback(true);
      return;
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
