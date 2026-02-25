"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocale } from "@/components/providers/LocaleProvider";
import { apiFetch, ApiError } from "@/lib/api";
import type { Country } from "@/lib/types";
import { cn } from "@/lib/utils";

type LanguageOption = {
  id: string;
  code: string;
  name?: string | null;
  native_name?: string | null;
  flag_emoji?: string | null;
};

type CurrencyOption = {
  id: string;
  code: string;
  name?: string | null;
  symbol?: string | null;
  native_symbol?: string | null;
};

type CountryOption = Country;
type TimezoneOption = {
  id: string;
  name: string;
  display_name?: string | null;
  formatted_offset?: string | null;
};

async function fetchLanguages() {
  const response = await apiFetch<LanguageOption[]>("/i18n/languages/");
  return response.data;
}

async function fetchCurrencies() {
  const response = await apiFetch<CurrencyOption[]>("/i18n/currencies/");
  return response.data;
}

async function fetchCountries() {
  const response = await apiFetch<CountryOption[]>("/i18n/countries/");
  return response.data;
}

async function fetchTimezones() {
  const response = await apiFetch<TimezoneOption[]>("/i18n/timezones/common/");
  return response.data;
}

export function LocaleSwitcher({
  className,
  includeCountry = false,
  includeTimezone = false,
  stacked = false,
  stackedInlineOnMobile = false,
  selectClassName,
}: {
  className?: string;
  includeCountry?: boolean;
  includeTimezone?: boolean;
  stacked?: boolean;
  stackedInlineOnMobile?: boolean;
  selectClassName?: string;
}) {
  const { locale, setLocale, isLoading } = useLocale();
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => {
    setMounted(true);
  }, []);
  const normalizeCurrencyCode = React.useCallback((value?: string | null) => {
    if (!value) return "";
    const code = String(value).trim().toUpperCase();
    return /^[A-Z]{3}$/.test(code) ? code : "";
  }, []);
  const normalizeCountryCode = React.useCallback((value?: string | null) => {
    if (!value) return "";
    const code = String(value).trim().toUpperCase();
    return /^[A-Z]{2}$/.test(code) ? code : "";
  }, []);
  const languagesQuery = useQuery({
    queryKey: ["i18n", "languages"],
    queryFn: fetchLanguages,
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
  const currenciesQuery = useQuery({
    queryKey: ["i18n", "currencies"],
    queryFn: fetchCurrencies,
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
  const countriesQuery = useQuery({
    queryKey: ["i18n", "countries"],
    queryFn: fetchCountries,
    enabled: includeCountry,
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
  const timezonesQuery = useQuery({
    queryKey: ["i18n", "timezones", "common"],
    queryFn: fetchTimezones,
    enabled: includeTimezone,
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

  const languages = mounted ? languagesQuery.data ?? [] : [];
  const currencies = mounted ? currenciesQuery.data ?? [] : [];
  const countries = mounted ? countriesQuery.data ?? [] : [];
  const timezones = mounted ? timezonesQuery.data ?? [] : [];
  const language = mounted ? locale.language || languages[0]?.code || "" : "";
  const normalizedCurrency = mounted ? normalizeCurrencyCode(locale.currency) : "";
  const normalizedCountry = mounted ? normalizeCountryCode(locale.country) : "";
  const resolvedCountryCode =
    normalizedCountry ||
    countries.find(
      (country) =>
        String(country.name || "").trim().toLowerCase() ===
        String(locale.country || "").trim().toLowerCase()
    )?.code ||
    "";
  const isBusy =
    !mounted ||
    isLoading ||
    languagesQuery.isLoading ||
    currenciesQuery.isLoading ||
    (includeCountry && countriesQuery.isLoading) ||
    (includeTimezone && timezonesQuery.isLoading);

  const languageOptions = languages.map((option) => ({
    value: option.code,
    label: `${option.flag_emoji ? `${option.flag_emoji} ` : ""}${
      option.native_name || option.name || option.code
    }`,
  }));

  const currencySymbolsByCode = currencies.reduce<Record<string, string>>((acc, option) => {
    const code = normalizeCurrencyCode(option.code);
    if (!code) return acc;
    const symbol = String(option.native_symbol || option.symbol || "").trim();
    if (symbol) acc[code] = symbol;
    return acc;
  }, {});

  const currencyOptions = currencies
    .map((option) => {
      const code = normalizeCurrencyCode(option.code);
      const symbol = code ? currencySymbolsByCode[code] || "" : "";
      return code
        ? {
            value: code,
            label: symbol ? `${code}\u00A0\u00A0\u00A0${symbol}` : code,
          }
        : null;
    })
    .filter(
      (option): option is { value: string; label: string } => Boolean(option?.value)
    );
  const resolvedCurrencyOptions =
    normalizedCurrency &&
    !currencyOptions.some((option) => option.value === normalizedCurrency)
      ? [{ value: normalizedCurrency, label: normalizedCurrency }, ...currencyOptions]
      : currencyOptions;
  const currency = normalizedCurrency || resolvedCurrencyOptions[0]?.value || "";
  const countryOptions = countries.map((option) => ({
    value: option.code,
    label: `${option.flag_emoji ? `${option.flag_emoji} ` : ""}${option.name}`,
  }));
  const resolvedCountryOptions =
    resolvedCountryCode &&
    !countryOptions.some((option) => option.value === resolvedCountryCode)
      ? [{ value: resolvedCountryCode, label: resolvedCountryCode }, ...countryOptions]
      : countryOptions;
  const country = resolvedCountryCode || resolvedCountryOptions[0]?.value || "";
  const timezoneValue = mounted ? String(locale.timezone || "").trim() : "";
  const timezoneMatch = timezones.find((option) => {
    const current = timezoneValue.toLowerCase();
    if (!current) return false;
    const name = String(option.name || "").trim().toLowerCase();
    const displayName = String(option.display_name || "").trim().toLowerCase();
    return name === current || displayName === current;
  });
  const timezone = timezoneMatch?.name || timezoneValue || timezones[0]?.name || "";
  const timezoneOptions = timezones.map((option) => {
    const label = option.display_name || option.name;
    const offset = String(option.formatted_offset || "").trim();
    return {
      value: option.name,
      label: offset ? `${label} (${offset})` : label,
    };
  });
  const resolvedTimezoneOptions =
    timezone && !timezoneOptions.some((option) => option.value === timezone)
      ? [{ value: timezone, label: timezone }, ...timezoneOptions]
      : timezoneOptions;

  React.useEffect(() => {
    if (!mounted || isBusy) return;
    const next: {
      language?: string;
      currency?: string;
      country?: string;
      timezone?: string;
    } = {};

    if (!locale.language && language) next.language = language;
    if (!locale.currency && currency) next.currency = currency;
    if (includeCountry && !locale.country && country) next.country = country;
    if (includeTimezone && !locale.timezone && timezone) next.timezone = timezone;

    if (Object.keys(next).length > 0) {
      setLocale(next);
    }
  }, [
    mounted,
    isBusy,
    includeCountry,
    includeTimezone,
    locale.language,
    locale.currency,
    locale.country,
    locale.timezone,
    language,
    currency,
    country,
    timezone,
    setLocale,
  ]);

  const wrapperClass = stacked
    ? "flex w-full flex-col gap-4"
    : "flex w-auto flex-col gap-2 sm:flex-row sm:items-center";
  const rowClass = stacked
    ? stackedInlineOnMobile
      ? "grid w-full grid-cols-[minmax(6.5rem,auto)_minmax(0,1fr)] items-center gap-3 text-sm font-medium text-foreground/80 sm:gap-4"
      : "grid w-full gap-2 text-sm font-medium text-foreground/80 sm:grid-cols-[minmax(8rem,auto)_1fr] sm:items-center sm:gap-4"
    : "flex w-auto items-center gap-2 text-sm font-medium text-foreground/80";
  const selectClass = stacked
    ? "h-11 min-h-11 w-full rounded-lg border border-border bg-card px-3 text-sm leading-tight text-foreground disabled:cursor-not-allowed disabled:opacity-60"
    : "h-10 min-h-10 w-[8.5rem] rounded-lg border border-border bg-card px-2 text-sm leading-tight text-foreground disabled:cursor-not-allowed disabled:opacity-60 sm:w-32 sm:text-sm";

  return (
    <div className={cn(wrapperClass, className)}>
      <label className={rowClass}>
        <span className="whitespace-nowrap">Language</span>
        <select
          value={language}
          onChange={(event) => setLocale({ language: event.target.value })}
          disabled={isBusy || languageOptions.length === 0}
          className={cn(selectClass, selectClassName)}
        >
          {languageOptions.length === 0 ? (
            <option value="">
              {isBusy ? "Loading..." : "No languages"}
            </option>
          ) : (
            languageOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))
          )}
        </select>
      </label>
      <label className={rowClass}>
        <span className="whitespace-nowrap">Currency</span>
        <select
          value={currency}
          onChange={(event) => setLocale({ currency: event.target.value })}
          disabled={isBusy || resolvedCurrencyOptions.length === 0}
          className={cn(selectClass, selectClassName)}
        >
          {resolvedCurrencyOptions.length === 0 ? (
            <option value="">
              {isBusy ? "Loading..." : "No currencies"}
            </option>
          ) : (
            resolvedCurrencyOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))
          )}
        </select>
      </label>
      {includeCountry ? (
        <label className={rowClass}>
          <span className="whitespace-nowrap">Country</span>
          <select
            value={country}
            onChange={(event) => setLocale({ country: event.target.value })}
            disabled={isBusy || resolvedCountryOptions.length === 0}
            className={cn(selectClass, selectClassName)}
          >
            {resolvedCountryOptions.length === 0 ? (
              <option value="">
                {isBusy ? "Loading..." : "No countries"}
              </option>
            ) : (
              resolvedCountryOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))
            )}
          </select>
        </label>
      ) : null}
      {includeTimezone ? (
        <label className={rowClass}>
          <span className="whitespace-nowrap">Timezone</span>
          <select
            value={timezone}
            onChange={(event) => setLocale({ timezone: event.target.value })}
            disabled={isBusy || resolvedTimezoneOptions.length === 0}
            className={cn(selectClass, selectClassName)}
          >
            {resolvedTimezoneOptions.length === 0 ? (
              <option value="">
                {isBusy ? "Loading..." : "No timezones"}
              </option>
            ) : (
              resolvedTimezoneOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))
            )}
          </select>
        </label>
      ) : null}
    </div>
  );
}
