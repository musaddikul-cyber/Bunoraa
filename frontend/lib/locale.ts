export type LocaleState = {
  language?: string;
  currency?: string;
  timezone?: string;
  country?: string;
};

const LOCALE_KEY = "bunoraa-locale";

export function getStoredLocale(): LocaleState {
  if (typeof window === "undefined") return {};
  const raw = window.localStorage.getItem(LOCALE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw) as LocaleState;
    return parsed || {};
  } catch {
    return {};
  }
}

export function setStoredLocale(next: LocaleState) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LOCALE_KEY, JSON.stringify(next));

  const setCookie = (name: string, value?: string) => {
    if (!value) return;
    const encoded = encodeURIComponent(value);
    document.cookie = `${name}=${encoded}; path=/; samesite=Lax; max-age=${60 * 60 * 24 * 365}`;
  };

  setCookie("language", next.language);
  setCookie("currency", next.currency);
  setCookie("timezone", next.timezone);
  setCookie("country", next.country);
}

export function getLocaleHeaders(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const stored = getStoredLocale();
  const headers: Record<string, string> = {};
  if (stored.language) {
    headers["Accept-Language"] = stored.language;
  }
  if (stored.currency) {
    headers["X-User-Currency"] = stored.currency;
  }
  if (stored.timezone) {
    headers["X-User-Timezone"] = stored.timezone;
  }
  if (stored.country) {
    headers["X-User-Country"] = stored.country;
  }
  return headers;
}
