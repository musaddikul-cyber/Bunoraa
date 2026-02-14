"use client";

import { useEffect, useRef } from "react";
import { usePathname, useSearchParams } from "next/navigation";

import { apiFetch } from "@/lib/api";

export function PageViewTracker() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const lastTrackedRef = useRef<string>("");

  useEffect(() => {
    const pagePath = pathname || "/";
    const queryString = searchParams?.toString() || "";
    const pageKey = `${pagePath}?${queryString}`;

    if (lastTrackedRef.current === pageKey) {
      return;
    }
    lastTrackedRef.current = pageKey;

    void apiFetch("/analytics/track/", {
      method: "POST",
      skipAuth: true,
      suppressError: true,
      body: {
        event_type: "page_view",
        metadata: {
          page_path: pagePath,
          query_string: queryString,
          referrer: typeof document !== "undefined" ? document.referrer : "",
        },
      },
    });
  }, [pathname, searchParams]);

  return null;
}
