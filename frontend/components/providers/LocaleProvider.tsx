"use client";

import * as React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { apiFetch, ApiError } from "@/lib/api";
import { getStoredLocale, setStoredLocale, type LocaleState } from "@/lib/locale";
import { AUTH_EVENT_NAME, getAccessToken } from "@/lib/auth";

type LocaleApiPayload = Record<string, unknown>;

type LocaleContextValue = {
  locale: LocaleState;
  setLocale: (next: Partial<LocaleState>) => void;
  isLoading: boolean;
};

const LocaleContext = React.createContext<LocaleContextValue | undefined>(
  undefined
);

async function fetchPreferences() {
  const response = await apiFetch<LocaleApiPayload>("/i18n/preferences/", {
    method: "GET",
  });
  return response.data;
}

function normalizeText(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim();
  return normalized || undefined;
}

function normalizeCode(value: string, length: number): string {
  const normalized = value.trim();
  const upper = normalized.toUpperCase();
  if (new RegExp(`^[A-Z]{${length}}$`).test(upper)) {
    return upper;
  }
  return normalized;
}

function readLocaleValue(
  value: unknown,
  nestedKeys: string[] = ["code", "name"]
): string | undefined {
  const direct = normalizeText(value);
  if (direct) return direct;
  if (!value || typeof value !== "object") return undefined;

  const record = value as Record<string, unknown>;
  for (const key of nestedKeys) {
    const nested = normalizeText(record[key]);
    if (nested) return nested;
  }
  return undefined;
}

function parseLocalePreferences(data: LocaleApiPayload): Partial<LocaleState> {
  const language =
    readLocaleValue(data.language_code, ["code"]) ||
    readLocaleValue(data.language, ["code"]);
  const currencyRaw =
    readLocaleValue(data.currency_code, ["code"]) ||
    readLocaleValue(data.currency, ["code"]);
  const timezone =
    readLocaleValue(data.timezone_name, ["name"]) ||
    readLocaleValue(data.timezone, ["name"]);
  const countryRaw =
    readLocaleValue(data.country_code, ["code"]) ||
    readLocaleValue(data.country, ["code", "name"]);

  const next: Partial<LocaleState> = {};
  if (language) next.language = language;
  if (currencyRaw) next.currency = normalizeCode(currencyRaw, 3);
  if (timezone) next.timezone = timezone;
  if (countryRaw) next.country = normalizeCode(countryRaw, 2);
  return next;
}

function buildPreferenceUpdateBody(payload: Partial<LocaleState>): Record<string, string> {
  const body: Record<string, string> = {};

  const language = normalizeText(payload.language);
  const currency = normalizeText(payload.currency);
  const timezone = normalizeText(payload.timezone);
  const country = normalizeText(payload.country);

  if (language) body.language_code = language;
  if (currency) body.currency_code = normalizeCode(currency, 3);
  if (timezone) body.timezone_name = timezone;
  if (country) body.country_code = normalizeCode(country, 2);

  return body;
}

export function LocaleProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [, startTransition] = React.useTransition();
  const [locale, setLocaleState] = React.useState<LocaleState>({});
  const [hasAuthToken, setHasAuthToken] = React.useState(false);
  const [shouldFetchPreferences, setShouldFetchPreferences] = React.useState(false);

  React.useEffect(() => {
    const syncAuthState = () => {
      setHasAuthToken(Boolean(getAccessToken()));
    };

    syncAuthState();
    if (typeof window === "undefined") return;

    window.addEventListener(AUTH_EVENT_NAME, syncAuthState);
    window.addEventListener("storage", syncAuthState);
    return () => {
      window.removeEventListener(AUTH_EVENT_NAME, syncAuthState);
      window.removeEventListener("storage", syncAuthState);
    };
  }, []);

  React.useEffect(() => {
    const stored = getStoredLocale();
    if (Object.keys(stored).length === 0) return;
    setLocaleState((prev) => {
      const merged = { ...stored, ...prev };
      setStoredLocale(merged);
      return merged;
    });
  }, []);

  React.useEffect(() => {
    if (!hasAuthToken) {
      setShouldFetchPreferences(false);
      return;
    }
    if (typeof window === "undefined") return;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let idleId: number | null = null;

    const markReady = () => setShouldFetchPreferences(true);
    const idleApi = globalThis as typeof globalThis & {
      requestIdleCallback?: (callback: IdleRequestCallback, options?: IdleRequestOptions) => number;
      cancelIdleCallback?: (handle: number) => void;
    };
    if (typeof idleApi.requestIdleCallback === "function") {
      idleId = idleApi.requestIdleCallback(markReady, { timeout: 2000 });
      return () => {
        if (idleId !== null && typeof idleApi.cancelIdleCallback === "function") {
          idleApi.cancelIdleCallback(idleId);
        }
      };
    }

    timeoutId = setTimeout(markReady, 1200);
    return () => {
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
    };
  }, [hasAuthToken]);

  const prefsQuery = useQuery({
    queryKey: ["locale", "preferences"],
    queryFn: fetchPreferences,
    enabled: hasAuthToken && shouldFetchPreferences,
    staleTime: 12 * 60 * 60 * 1000,
    gcTime: 12 * 60 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchOnMount: false,
    retry: (failureCount, error) => {
      if (error instanceof ApiError && error.status === 429) return false;
      return failureCount < 2;
    },
  });

  React.useEffect(() => {
    if (!prefsQuery.data) return;
    const next = parseLocalePreferences(prefsQuery.data as LocaleApiPayload);
    if (Object.keys(next).length === 0) return;
    setLocaleState((prev) => {
      const merged = { ...prev, ...next };
      setStoredLocale(merged);
      return merged;
    });
  }, [prefsQuery.data]);

  const updatePrefs = useMutation({
    mutationFn: async (payload: Partial<LocaleState>) => {
      const body = buildPreferenceUpdateBody(payload);
      if (Object.keys(body).length === 0) return {};
      const response = await apiFetch<LocaleApiPayload>(
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
      const sanitized = buildPreferenceUpdateBody(next);
      const normalizedNext: Partial<LocaleState> = {};
      if (sanitized.language_code) normalizedNext.language = sanitized.language_code;
      if (sanitized.currency_code) normalizedNext.currency = sanitized.currency_code;
      if (sanitized.timezone_name) normalizedNext.timezone = sanitized.timezone_name;
      if (sanitized.country_code) normalizedNext.country = sanitized.country_code;
      if (Object.keys(normalizedNext).length === 0) return;

      setLocaleState((prev) => {
        const merged = { ...prev, ...normalizedNext };
        setStoredLocale(merged);
        return merged;
      });
      updatePrefs.mutate(normalizedNext);
      startTransition(() => {
        router.refresh();
      });
    },
    [router, startTransition, updatePrefs]
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
