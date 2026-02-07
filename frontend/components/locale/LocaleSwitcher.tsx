"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocale } from "@/components/providers/LocaleProvider";
import { apiFetch } from "@/lib/api";
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
  const languagesQuery = useQuery({
    queryKey: ["i18n", "languages"],
    queryFn: fetchLanguages,
    staleTime: 30 * 60 * 1000,
  });
  const currenciesQuery = useQuery({
    queryKey: ["i18n", "currencies"],
    queryFn: fetchCurrencies,
    staleTime: 30 * 60 * 1000,
  });

  const languages = languagesQuery.data ?? [];
  const currencies = currenciesQuery.data ?? [];
  const language = locale.language || languages[0]?.code || "";
  const currency = locale.currency || currencies[0]?.code || "";
  const isBusy = isLoading || languagesQuery.isLoading || currenciesQuery.isLoading;

  const languageOptions = languages.map((option) => ({
    value: option.code,
    label: `${option.flag_emoji ? `${option.flag_emoji} ` : ""}${
      option.native_name || option.name || option.code
    }`,
  }));

  const currencyOptions = currencies.map((option) => ({
    value: option.code,
    label: `${option.code}${option.symbol ? ` ${option.symbol}` : ""}`,
  }));

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
          disabled={isBusy || currencyOptions.length === 0}
          className="h-8 min-h-0 w-full rounded-lg border border-border bg-card px-2 text-xs leading-tight text-foreground disabled:cursor-not-allowed disabled:opacity-60 sm:h-9 sm:w-36 sm:text-sm"
        >
          {currencyOptions.length === 0 ? (
            <option value="">
              {isBusy ? "Loading..." : "No currencies"}
            </option>
          ) : (
            currencyOptions.map((option) => (
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
