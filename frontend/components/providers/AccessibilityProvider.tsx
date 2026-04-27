"use client";

import * as React from "react";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { useQueryClient } from "@tanstack/react-query";
import type { UserPreferences } from "@/lib/types";

/**
 * Applies user accessibility preferences (reduce_motion, high_contrast, large_text)
 * as data attributes on the <html> element so CSS can respond globally.
 *
 * - data-reduce-motion="true"  → disables animations/transitions
 * - data-high-contrast="true"  → increases contrast ratios
 * - data-large-text="true"     → increases base font size
 *
 * This provider reads from the React Query cache instead of fetching,
 * avoiding duplicate API calls when usePreferences is already fetching.
 */
export function AccessibilityProvider({ children }: { children: React.ReactNode }) {
  const { hasToken } = useAuthContext();
  const queryClient = useQueryClient();

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

    // Read from React Query cache (shared with usePreferences)
    const prefs = queryClient.getQueryData<UserPreferences>(
      ["account", "preferences"]
    );

    const reduceMotion = prefs?.reduce_motion ?? cached?.reduce_motion ?? false;
    const highContrast = prefs?.high_contrast ?? cached?.high_contrast ?? false;
    const largeText = prefs?.large_text ?? cached?.large_text ?? false;

    // Set data attributes as boolean strings ("true" or "false")
    root.setAttribute("data-reduce-motion", reduceMotion ? "true" : "false");
    root.setAttribute("data-high-contrast", highContrast ? "true" : "false");
    root.setAttribute("data-large-text", largeText ? "true" : "false");

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
  }, [hasToken, queryClient]);

  return <>{children}</>;
}
