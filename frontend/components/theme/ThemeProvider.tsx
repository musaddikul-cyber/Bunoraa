"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

export type ThemeName =
  | "light"
  | "dark"
  | "moonlight"
  | "gray"
  | "modern"
  | "system";

type ThemeContextValue = {
  theme: ThemeName;
  setTheme: (theme: ThemeName) => void;
};

const THEME_KEY = "bunoraa-theme";
const THEME_CLASSES: ThemeName[] = [
  "light",
  "dark",
  "moonlight",
  "gray",
  "modern",
  "system",
];

const ThemeContext = React.createContext<ThemeContextValue | undefined>(
  undefined
);

function applyTheme(theme: ThemeName) {
  const root = document.documentElement;
  root.classList.remove(...THEME_CLASSES);

  if (theme === "system") {
    root.classList.add("system");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.classList.toggle("dark", prefersDark);
    root.style.colorScheme = "light dark";
    return;
  }

  root.classList.add(theme);
  root.classList.toggle("dark", theme === "dark");
  if (theme === "dark") {
    root.style.colorScheme = "dark";
  } else {
    root.style.colorScheme = "light";
  }
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = React.useState<ThemeName>("system");

  React.useEffect(() => {
    const stored = window.localStorage.getItem(THEME_KEY) as ThemeName | null;
    const initial = stored && THEME_CLASSES.includes(stored) ? stored : "system";
    setThemeState(initial);
    applyTheme(initial);
  }, []);

  React.useEffect(() => {
    window.localStorage.setItem(THEME_KEY, theme);
    applyTheme(theme);

    if (theme !== "system") return;

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = () => applyTheme("system");
    media.addEventListener("change", listener);
    return () => media.removeEventListener("change", listener);
  }, [theme]);

  const setTheme = React.useCallback((next: ThemeName) => {
    setThemeState(next);
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = React.useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme must be used within ThemeProvider");
  }
  return ctx;
}

export function ThemeSwitcher({ className }: { className?: string }) {
  const { theme, setTheme } = useTheme();

  return (
    <label
      className={cn(
        "flex w-auto items-center gap-2 text-sm font-medium text-foreground/80",
        className
      )}
    >
      <span className="whitespace-nowrap">Theme</span>
      <select
        value={theme}
        onChange={(event) => setTheme(event.target.value as ThemeName)}
        className="h-10 min-h-0 w-[8.5rem] rounded-lg border border-border bg-card px-2 text-sm leading-tight text-foreground sm:h-9 sm:w-32 sm:text-sm"
      >
        {THEME_CLASSES.map((option) => (
          <option key={option} value={option}>
            {option.charAt(0).toUpperCase() + option.slice(1)}
          </option>
        ))}
      </select>
    </label>
  );
}
