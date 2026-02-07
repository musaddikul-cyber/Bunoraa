"use client";

import { usePreferences } from "@/components/account/usePreferences";
import type { UserPreferences } from "@/lib/types";
import { useTheme } from "@/components/theme/ThemeProvider";
import { useLocale } from "@/components/providers/LocaleProvider";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export default function PreferencesPage() {
  const { preferencesQuery, updatePreferences } = usePreferences();
  const { theme, setTheme } = useTheme();
  const { locale, setLocale } = useLocale();
  const prefs: UserPreferences = preferencesQuery.data ?? {};
  type PreferenceToggleKey =
    | "reduce_motion"
    | "high_contrast"
    | "large_text"
    | "allow_tracking"
    | "share_data_for_ads";

  if (preferencesQuery.isLoading) {
    return (
      <Card variant="bordered" className="p-6 text-sm text-foreground/70">
        Loading preferences...
      </Card>
    );
  }

  const toggle = (field: PreferenceToggleKey) => {
    updatePreferences.mutate({ [field]: !prefs[field] });
  };

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Account
        </p>
        <h1 className="text-3xl font-semibold">Preferences</h1>
        <p className="mt-2 text-sm text-foreground/70">
          Personalize your storefront, language, and accessibility settings.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card variant="bordered" className="space-y-4">
          <h2 className="text-lg font-semibold">Theme</h2>
          <div className="flex flex-wrap gap-2">
            {["system", "light", "dark", "moonlight", "gray", "modern"].map(
              (option) => (
                <Button
                  key={option}
                  size="sm"
                  variant={theme === option ? "primary" : "secondary"}
                  onClick={() => {
                    setTheme(option as typeof theme);
                    updatePreferences.mutate({ theme: option });
                  }}
                >
                  {option}
                </Button>
              )
            )}
          </div>
        </Card>

        <Card variant="bordered" className="space-y-4">
          <h2 className="text-lg font-semibold">Locale</h2>
          <label className="block text-sm">
            Language code
            <input
              className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
              value={locale.language || ""}
              onChange={(event) => setLocale({ language: event.target.value })}
              placeholder="en"
            />
          </label>
          <label className="block text-sm">
            Currency
            <input
              className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
              value={locale.currency || ""}
              onChange={(event) => setLocale({ currency: event.target.value })}
              placeholder="USD"
            />
          </label>
          <label className="block text-sm">
            Timezone
            <input
              className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
              value={locale.timezone || ""}
              onChange={(event) => setLocale({ timezone: event.target.value })}
              placeholder="Asia/Dhaka"
            />
          </label>
        </Card>
      </div>

      <Card variant="bordered" className="space-y-4">
        <h2 className="text-lg font-semibold">Accessibility</h2>
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant={prefs.reduce_motion ? "primary" : "secondary"}
            onClick={() => toggle("reduce_motion")}
          >
            Reduce motion
          </Button>
          <Button
            size="sm"
            variant={prefs.high_contrast ? "primary" : "secondary"}
            onClick={() => toggle("high_contrast")}
          >
            High contrast
          </Button>
          <Button
            size="sm"
            variant={prefs.large_text ? "primary" : "secondary"}
            onClick={() => toggle("large_text")}
          >
            Large text
          </Button>
        </div>
      </Card>

      <Card variant="bordered" className="space-y-4">
        <h2 className="text-lg font-semibold">Privacy</h2>
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant={prefs.allow_tracking ? "primary" : "secondary"}
            onClick={() => toggle("allow_tracking")}
          >
            Allow personalization
          </Button>
          <Button
            size="sm"
            variant={prefs.share_data_for_ads ? "primary" : "secondary"}
            onClick={() => toggle("share_data_for_ads")}
          >
            Share data for ads
          </Button>
        </div>
      </Card>
    </div>
  );
}
