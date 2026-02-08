"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocale } from "@/components/providers/LocaleProvider";
import { apiFetch, ApiError } from "@/lib/api";
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

async function fetchLanguages() {
  const response = await apiFetch<LanguageOption[]>("/i18n/languages/");
  return response.data;
}

async function fetchCurrencies() {
  const response = await apiFetch<CurrencyOption[]>("/i18n/currencies/");
  return response.data;
}

export function LocaleSwitcher({ className }: { className?: string }) {
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

  const languages = mounted ? languagesQuery.data ?? [] : [];
  const currencies = mounted ? currenciesQuery.data ?? [] : [];
  const language = mounted ? locale.language || languages[0]?.code || "" : "";
  const normalizedCurrency = mounted ? normalizeCurrencyCode(locale.currency) : "";
  const isBusy =
    !mounted || isLoading || languagesQuery.isLoading || currenciesQuery.isLoading;

  const languageOptions = languages.map((option) => ({
    value: option.code,
    label: `${option.flag_emoji ? `${option.flag_emoji} ` : ""}${
      option.native_name || option.name || option.code
    }`,
  }));

  const currencyOptions = currencies
    .map((option) => {
      const code = normalizeCurrencyCode(option.code);
      return code
        ? {
            value: code,
            label: `${code}${option.symbol ? ` ${option.symbol}` : ""}`,
          }
        : null;
    })
    .filter(
      (option): option is { value: string; label: string } =>
        Boolean(option?.value)
    );
  const resolvedCurrencyOptions =
    normalizedCurrency &&
    !currencyOptions.some((option) => option.value === normalizedCurrency)
      ? [{ value: normalizedCurrency, label: normalizedCurrency }, ...currencyOptions]
      : currencyOptions;
  const currency = normalizedCurrency || resolvedCurrencyOptions[0]?.value || "";

  return (
    <div
      className={cn(
        "flex w-full flex-col gap-3 sm:w-auto sm:flex-row sm:items-center",
        className
      )}
    >
      <label className="flex w-full items-center gap-2 text-xs text-foreground/70 sm:w-auto">
        <span className="whitespace-nowrap">Language</span>
        <select
          value={language}
          onChange={(event) => setLocale({ language: event.target.value })}
          disabled={isBusy || languageOptions.length === 0}
          className="h-8 min-h-0 w-full rounded-lg border border-border bg-card px-2 text-xs leading-tight text-foreground disabled:cursor-not-allowed disabled:opacity-60 sm:h-9 sm:w-36 sm:text-sm"
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
      <label className="flex w-full items-center gap-2 text-xs text-foreground/70 sm:w-auto">
        <span className="whitespace-nowrap">Currency</span>
        <select
          value={currency}
          onChange={(event) => setLocale({ currency: event.target.value })}
          disabled={isBusy || resolvedCurrencyOptions.length === 0}
          className="h-8 min-h-0 w-full rounded-lg border border-border bg-card px-2 text-xs leading-tight text-foreground disabled:cursor-not-allowed disabled:opacity-60 sm:h-9 sm:w-36 sm:text-sm"
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
    </div>
  );
}
