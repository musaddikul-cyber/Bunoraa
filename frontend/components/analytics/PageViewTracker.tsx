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

    const track = () => {
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
    };

    if (typeof window === "undefined") {
      track();
      return;
    }

    let idleId: number | null = null;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    const idleApi = globalThis as typeof globalThis & {
      requestIdleCallback?: (callback: IdleRequestCallback, options?: IdleRequestOptions) => number;
      cancelIdleCallback?: (handle: number) => void;
    };
    if (typeof idleApi.requestIdleCallback === "function") {
      idleId = idleApi.requestIdleCallback(track, { timeout: 3000 });
    } else {
      timeoutId = setTimeout(track, 1800);
    }

    return () => {
      if (idleId !== null && typeof idleApi.cancelIdleCallback === "function") {
        idleApi.cancelIdleCallback(idleId);
      }
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
    };
  }, [pathname, searchParams]);

  return null;
}
