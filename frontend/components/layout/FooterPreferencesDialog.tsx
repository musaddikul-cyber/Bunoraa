"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/Card";
import { LocaleSwitcher } from "@/components/locale/LocaleSwitcher";
import { ThemeSwitcher, useTheme } from "@/components/theme/ThemeProvider";
import { useLocale } from "@/components/providers/LocaleProvider";
import { Button } from "@/components/ui/Button";
import { apiFetch, ApiError } from "@/lib/api";
import type { Country } from "@/lib/types";

type LanguageOption = {
  code: string;
  name?: string | null;
  native_name?: string | null;
};

type CurrencyOption = {
  code: string;
  symbol?: string | null;
  native_symbol?: string | null;
};

type CountryOption = Country;

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

function formatTheme(theme: string): string {
  if (!theme) return "System";
  return `${theme.charAt(0).toUpperCase()}${theme.slice(1)}`;
}

function formatValue(value?: string | null): string {
  const normalized = String(value || "").trim();
  if (!normalized) return "--";
  return normalized;
}

function normalizeCode(value?: string | null, length = 2): string {
  if (!value) return "";
  const normalized = String(value).trim().toUpperCase();
  return new RegExp(`^[A-Z]{${length}}$`).test(normalized) ? normalized : "";
}

function normalizeText(value?: string | null): string {
  return String(value || "").trim();
}

export function FooterPreferencesDialog({ className }: { className?: string }) {
  const { theme } = useTheme();
  const { locale } = useLocale();
  const [isOpen, setIsOpen] = React.useState(false);
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
    if (!isOpen) return;

    const previousOverflow = document.body.style.overflow;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsOpen(false);
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isOpen]);

  const languages = languagesQuery.data ?? [];
  const currencies = currenciesQuery.data ?? [];
  const countries = countriesQuery.data ?? [];

  const languageRaw = normalizeText(locale.language);
  const languageOption =
    languages.find((language) => language.code.toLowerCase() === languageRaw.toLowerCase()) ||
    languages.find((language) => normalizeText(language.name).toLowerCase() === languageRaw.toLowerCase()) ||
    languages.find(
      (language) => normalizeText(language.native_name).toLowerCase() === languageRaw.toLowerCase()
    ) ||
    languages[0];
  const languageLabel =
    normalizeText(languageOption?.native_name) ||
    normalizeText(languageOption?.name) ||
    normalizeText(languageOption?.code) ||
    languageRaw;

  const normalizedCurrency = normalizeCode(locale.currency, 3);
  const currencyOption =
    currencies.find((currency) => normalizeCode(currency.code, 3) === normalizedCurrency) ||
    currencies.find(
      (currency) =>
        normalizeCode(currency.code, 3) === normalizeCode(locale.currency, 3)
    ) ||
    currencies[0];
  const currencyCode =
    normalizedCurrency ||
    normalizeCode(currencyOption?.code, 3) ||
    normalizeText(locale.currency).toUpperCase();
  const currencySymbol =
    normalizeText(currencyOption?.native_symbol) || normalizeText(currencyOption?.symbol);
  const currencyLabel = currencyCode
    ? currencySymbol
      ? `${currencyCode} (${currencySymbol})`
      : currencyCode
    : "";

  const normalizedCountry = normalizeCode(locale.country, 2);
  const countryOption =
    countries.find((country) => country.code.toUpperCase() === normalizedCountry) ||
    countries.find(
      (country) =>
        String(country.name || "").trim().toLowerCase() ===
        String(locale.country || "").trim().toLowerCase()
    ) ||
    countries[0];
  const countryLabel = normalizeText(countryOption?.name) || normalizeText(locale.country);

  const summaryParts = [
    formatTheme(theme),
    formatValue(languageLabel),
    formatValue(currencyLabel),
    formatValue(countryLabel),
  ];

  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className={cn(
          "group inline-flex w-full flex-col items-center justify-center rounded-full border border-border/70 bg-background/60 px-3 py-1.5 text-center shadow-soft transition-all duration-200 hover:-translate-y-0.5 hover:border-foreground/35 hover:bg-background/80 hover:shadow-md lg:w-auto lg:items-start lg:text-left",
          className
        )}
        aria-haspopup="dialog"
        aria-expanded={isOpen}
      >
        <span className="flex flex-wrap items-center justify-center text-xs font-semibold leading-tight text-foreground/85 transition-colors duration-200 group-hover:text-foreground lg:justify-start">
          {summaryParts.map((part, index) => (
            <React.Fragment key={`${part}-${index}`}>
              {index > 0 ? (
                <span
                  aria-hidden="true"
                  className="px-2 text-foreground/55 transition-all duration-200 group-hover:scale-110 group-hover:text-foreground/80"
                >
                  |
                </span>
              ) : null}
              <span className="transition-colors duration-200 group-hover:text-foreground">{part}</span>
            </React.Fragment>
          ))}
        </span>
      </button>

      {isOpen ? (
        <div
          className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 p-3 sm:items-center sm:p-4"
          role="dialog"
          aria-modal="true"
          aria-label="Update preferences"
        >
          <button
            type="button"
            className="absolute inset-0"
            aria-label="Close preferences"
            onClick={() => setIsOpen(false)}
          />
          <Card
            variant="bordered"
            className="relative z-10 w-full max-w-xl bg-background p-4 sm:p-6"
          >
            <div className="mb-4 flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Display preferences</h2>
              <Button variant="ghost" size="sm" onClick={() => setIsOpen(false)}>
                Close
              </Button>
            </div>
            <div className="space-y-4">
              <ThemeSwitcher
                className="w-full justify-between"
                selectClassName="w-52 sm:w-52"
              />
              <LocaleSwitcher
                includeCountry
                stacked
                className="w-full"
                selectClassName="w-52 sm:w-52"
              />
            </div>
          </Card>
        </div>
      ) : null}
    </>
  );
}
