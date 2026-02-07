"use client";

import * as React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { getStoredLocale, setStoredLocale, type LocaleState } from "@/lib/locale";

type LocaleContextValue = {
  locale: LocaleState;
  setLocale: (next: Partial<LocaleState>) => void;
  isLoading: boolean;
};

const LocaleContext = React.createContext<LocaleContextValue | undefined>(
  undefined
);

async function fetchPreferences() {
  const response = await apiFetch<Record<string, unknown>>("/i18n/preferences/", {
    method: "GET",
  });
  return response.data;
}

export function LocaleProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = React.useState<LocaleState>(() =>
    typeof window !== "undefined" ? getStoredLocale() : {}
  );

  const prefsQuery = useQuery({
    queryKey: ["locale", "preferences"],
    queryFn: fetchPreferences,
    staleTime: 10 * 60 * 1000,
  });

  React.useEffect(() => {
    if (!prefsQuery.data) return;
    const data = prefsQuery.data as Record<string, unknown>;
    const next: LocaleState = {
      language: (data.language as string) || (data.language_code as string),
      currency: (data.currency_code as string) || (data.currency as string),
      timezone: (data.timezone as string) || (data.timezone_name as string),
      country: (data.country as string) || (data.country_code as string),
    };
    setLocaleState((prev) => {
      const merged = { ...prev, ...next };
      setStoredLocale(merged);
      return merged;
    });
  }, [prefsQuery.data]);

  const updatePrefs = useMutation({
    mutationFn: async (payload: Partial<LocaleState>) => {
      const body = {
        language: payload.language,
        currency_code: payload.currency,
        timezone: payload.timezone,
        country: payload.country,
      };
      const response = await apiFetch<Record<string, unknown>>(
        "/i18n/preferences/",
        {
          method: "PUT",
          body,
        }
      );
      return response.data;
    },
  });

  const setLocale = React.useCallback(
    (next: Partial<LocaleState>) => {
      setLocaleState((prev) => {
        const merged = { ...prev, ...next };
        setStoredLocale(merged);
        return merged;
      });
      updatePrefs.mutate(next);
    },
    [updatePrefs]
  );

  return (
    <LocaleContext.Provider
      value={{ locale, setLocale, isLoading: prefsQuery.isLoading }}
    >
      {children}
    </LocaleContext.Provider>
  );
}

export function useLocale() {
  const ctx = React.useContext(LocaleContext);
  if (!ctx) {
    throw new Error("useLocale must be used within LocaleProvider");
  }
  return ctx;
}
