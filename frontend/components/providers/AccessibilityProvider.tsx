"use client";

import * as React from "react";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { UserPreferences } from "@/lib/types";

/**
 * Applies user accessibility preferences (reduce_motion, high_contrast, large_text)
 * as data attributes on the <html> element so CSS can respond globally.
 *
 * - data-reduce-motion="true"  → disables animations/transitions
 * - data-high-contrast="true"  → increases contrast ratios
 * - data-large-text="true"     → increases base font size
 */
export function AccessibilityProvider({ children }: { children: React.ReactNode }) {
  const { hasToken } = useAuthContext();

  const { data: prefs } = useQuery({
    queryKey: ["account", "preferences"],
    queryFn: async () => {
      const res = await apiFetch<UserPreferences>("/accounts/preferences/");
      return res.data;
    },
    enabled: hasToken,
    staleTime: 5 * 60 * 1000,
  });

  React.useEffect(() => {
    const root = document.documentElement;

    // Read from localStorage for instant load (before API responds)
    const cached = (() => {
      try {
        const raw = localStorage.getItem("bunoraa-accessibility");
        return raw ? JSON.parse(raw) : null;
      } catch {
        return null;
      }
    })();

    const reduceMotion = prefs?.reduce_motion ?? cached?.reduce_motion ?? false;
    const highContrast = prefs?.high_contrast ?? cached?.high_contrast ?? false;
    const largeText = prefs?.large_text ?? cached?.large_text ?? false;

    root.setAttribute("data-reduce-motion", String(reduceMotion));
    root.setAttribute("data-high-contrast", String(highContrast));
    root.setAttribute("data-large-text", String(largeText));

    // Persist to localStorage for next page load
    if (prefs) {
      try {
        localStorage.setItem(
          "bunoraa-accessibility",
          JSON.stringify({
            reduce_motion: reduceMotion,
            high_contrast: highContrast,
            large_text: largeText,
          })
        );
      } catch {
        // Ignore storage errors
      }
    }

    return () => {
      root.removeAttribute("data-reduce-motion");
      root.removeAttribute("data-high-contrast");
      root.removeAttribute("data-large-text");
    };
  }, [prefs]);

  return <>{children}</>;
}
