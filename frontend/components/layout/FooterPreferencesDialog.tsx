"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/Card";
import { LocaleSwitcher } from "@/components/locale/LocaleSwitcher";
import { ThemeSwitcher, useTheme } from "@/components/theme/ThemeProvider";
import { useLocale } from "@/components/providers/LocaleProvider";
import { Button } from "@/components/ui/Button";

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

  const languageRaw = normalizeText(locale.language);
  const languageLabel = languageRaw || "";

  const normalizedCurrency = normalizeCode(locale.currency, 3);
  const currencyLabel = normalizedCurrency || normalizeText(locale.currency).toUpperCase();

  const countryLabel = normalizeText(locale.country).toUpperCase();

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
          className="fixed inset-0 z-[80] flex items-end justify-center overflow-y-auto bg-black/50 p-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))] sm:items-center sm:p-4"
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
            className="relative z-10 w-full max-w-xl max-h-[min(92dvh,44rem)] overflow-y-auto bg-background p-4 sm:p-6"
          >
            <div className="mb-4 flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Display preferences</h2>
              <Button variant="ghost" size="sm" onClick={() => setIsOpen(false)}>
                Close
              </Button>
            </div>
            <div className="space-y-4">
              <ThemeSwitcher
                className="grid w-full grid-cols-[minmax(6.5rem,auto)_minmax(0,1fr)] items-center gap-3 text-sm font-medium text-foreground/80 sm:gap-4"
                selectClassName="w-full sm:w-52 sm:justify-self-end"
              />
              <LocaleSwitcher
                includeCountry
                stacked
                stackedInlineOnMobile
                className="w-full"
                selectClassName="w-full sm:w-52 sm:justify-self-end"
              />
            </div>
          </Card>
        </div>
      ) : null}
    </>
  );
}
