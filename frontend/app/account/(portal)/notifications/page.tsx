"use client";

import * as React from "react";
import { useNotificationPreferences } from "@/components/notifications/useNotificationPreferences";
import type { NotificationPreference } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { subscribeToBrowserPush } from "@/lib/push";

const HAS_VAPID_PUBLIC_KEY = Boolean(process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY);

export default function NotificationPreferencesPage() {
  const { preferencesQuery, updatePreferences } = useNotificationPreferences();
  const prefs: NotificationPreference = preferencesQuery.data ?? {};
  const [pushStatus, setPushStatus] = React.useState<
    "idle" | "enabled" | "denied" | "unsupported" | "loading" | "error"
  >("idle");
  const [pushError, setPushError] = React.useState<string | null>(null);
  const isUpdating = updatePreferences.isPending;

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    if (
      !("serviceWorker" in navigator) ||
      !("PushManager" in window) ||
      !("Notification" in window)
    ) {
      setPushStatus("unsupported");
      setPushError("Browser notifications are unavailable on this device.");
      return;
    }
    if (!HAS_VAPID_PUBLIC_KEY) {
      setPushStatus("unsupported");
      setPushError("Browser notifications are not configured for this environment.");
      return;
    }
    if (Notification.permission === "denied") {
      setPushStatus("denied");
      setPushError(null);
    } else if (Notification.permission === "granted") {
      setPushStatus("enabled");
      setPushError(null);
    }
  }, []);

  if (preferencesQuery.isLoading) {
    return (
      <Card variant="bordered" className="p-6 text-sm text-foreground/70">
        Loading notification preferences...
      </Card>
    );
  }

  if (preferencesQuery.isError) {
    const message =
      preferencesQuery.error instanceof Error
        ? preferencesQuery.error.message
        : "Failed to load notification preferences.";
    return (
      <Card variant="bordered" className="space-y-4 p-6">
        <p className="text-sm text-red-600">{message}</p>
        <Button size="sm" variant="secondary" onClick={() => preferencesQuery.refetch()}>
          Retry
        </Button>
      </Card>
    );
  }

  const toggle = (field: keyof NotificationPreference) => {
    updatePreferences.mutate({ [field]: !prefs[field] });
  };

  const updateField = (field: keyof NotificationPreference, value: unknown) => {
    updatePreferences.mutate({ [field]: value });
  };

  const enableBrowserPush = async () => {
    setPushError(null);
    if (!HAS_VAPID_PUBLIC_KEY) {
      setPushStatus("unsupported");
      setPushError("Browser notifications are not configured for this environment.");
      return;
    }
    setPushStatus("loading");
    try {
      const result = await subscribeToBrowserPush();
      if (result.status === "enabled") {
        setPushStatus("enabled");
        updatePreferences.mutate({ push_enabled: true });
      } else if (result.status === "denied") {
        setPushStatus("denied");
      } else if (result.status === "unsupported") {
        setPushStatus("unsupported");
        if (result.error) setPushError(result.error);
      } else {
        setPushStatus("error");
        setPushError(result.error || "Failed to enable browser notifications.");
      }
    } catch {
      setPushStatus("error");
      setPushError("Failed to enable browser notifications.");
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Account
        </p>
        <h1 className="text-3xl font-semibold">Notifications</h1>
        <p className="mt-2 text-sm text-foreground/70">
          Control how you receive alerts, digests, and promotions.
        </p>
      </div>

      <Card variant="bordered" className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold">Email notifications</p>
            <p className="text-sm text-foreground/70">Enable or disable all email alerts.</p>
          </div>
          <Button
            variant={prefs.email_enabled ? "primary" : "secondary"}
            size="sm"
            disabled={isUpdating}
            onClick={() => toggle("email_enabled")}
          >
            {prefs.email_enabled ? "On" : "Off"}
          </Button>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold">SMS notifications</p>
            <p className="text-sm text-foreground/70">Order updates via SMS.</p>
          </div>
          <Button
            variant={prefs.sms_enabled ? "primary" : "secondary"}
            size="sm"
            disabled={isUpdating}
            onClick={() => toggle("sms_enabled")}
          >
            {prefs.sms_enabled ? "On" : "Off"}
          </Button>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold">Push notifications</p>
            <p className="text-sm text-foreground/70">Real-time alerts on supported devices.</p>
          </div>
          <Button
            variant={prefs.push_enabled ? "primary" : "secondary"}
            size="sm"
            disabled={isUpdating}
            onClick={() => toggle("push_enabled")}
          >
            {prefs.push_enabled ? "On" : "Off"}
          </Button>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-background/60 p-4">
          <div>
            <p className="font-semibold">Browser notifications</p>
            <p className="text-sm text-foreground/70">
              Enable web push for this device.
            </p>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={enableBrowserPush}
            disabled={isUpdating || pushStatus === "loading" || pushStatus === "unsupported"}
          >
            {pushStatus === "enabled"
              ? "Enabled"
              : pushStatus === "denied"
              ? "Blocked"
              : pushStatus === "unsupported"
              ? "Unavailable"
              : pushStatus === "loading"
              ? "Enabling..."
              : pushStatus === "error"
              ? "Try again"
              : "Enable"}
          </Button>
        </div>
        {pushError ? <p className="text-sm text-red-600">{pushError}</p> : null}
      </Card>

      <Card variant="bordered" className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold">Digest frequency</p>
            <p className="text-sm text-foreground/70">Batch non-critical updates.</p>
          </div>
          <select
            className="rounded-lg border border-border bg-card px-3 py-2 text-sm"
            value={prefs.digest_frequency || "immediate"}
            disabled={isUpdating}
            onChange={(event) => updateField("digest_frequency", event.target.value)}
          >
            <option value="immediate">Immediate</option>
            <option value="hourly">Hourly</option>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="never">Never</option>
          </select>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-2 text-sm">
            <span className="font-semibold">Quiet hours start</span>
            <input
              type="time"
              value={prefs.quiet_hours_start || ""}
              disabled={isUpdating}
              onChange={(event) => updateField("quiet_hours_start", event.target.value || null)}
              className="rounded-lg border border-border bg-card px-3 py-2 text-sm"
            />
          </label>
          <label className="flex flex-col gap-2 text-sm">
            <span className="font-semibold">Quiet hours end</span>
            <input
              type="time"
              value={prefs.quiet_hours_end || ""}
              disabled={isUpdating}
              onChange={(event) => updateField("quiet_hours_end", event.target.value || null)}
              className="rounded-lg border border-border bg-card px-3 py-2 text-sm"
            />
          </label>
        </div>
      </Card>

      <Card variant="bordered" className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold">Order updates</p>
            <p className="text-sm text-foreground/70">Shipping, delivery, and status alerts.</p>
          </div>
          <Button
            variant={prefs.email_order_updates ? "primary" : "secondary"}
            size="sm"
            disabled={isUpdating}
            onClick={() => toggle("email_order_updates")}
          >
            {prefs.email_order_updates ? "On" : "Off"}
          </Button>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold">Promotions</p>
            <p className="text-sm text-foreground/70">Sales, new drops, and curated picks.</p>
          </div>
          <Button
            variant={prefs.email_promotions ? "primary" : "secondary"}
            size="sm"
            disabled={isUpdating}
            onClick={() => toggle("email_promotions")}
          >
            {prefs.email_promotions ? "On" : "Off"}
          </Button>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold">Price drops</p>
            <p className="text-sm text-foreground/70">Alerts when wishlist items drop in price.</p>
          </div>
          <Button
            variant={prefs.email_price_drops ? "primary" : "secondary"}
            size="sm"
            disabled={isUpdating}
            onClick={() => toggle("email_price_drops")}
          >
            {prefs.email_price_drops ? "On" : "Off"}
          </Button>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold">Back in stock</p>
            <p className="text-sm text-foreground/70">Know when items return.</p>
          </div>
          <Button
            variant={prefs.email_back_in_stock ? "primary" : "secondary"}
            size="sm"
            disabled={isUpdating}
            onClick={() => toggle("email_back_in_stock")}
          >
            {prefs.email_back_in_stock ? "On" : "Off"}
          </Button>
        </div>
        <div className="grid gap-3">
          <div className="flex items-start justify-between gap-3">
            <div className="space-y-1">
              <p className="font-semibold">Marketing opt-in</p>
              <p className="text-sm text-foreground/70">Allow marketing notifications.</p>
            </div>
            <Button
              variant={prefs.marketing_opt_in ? "primary" : "secondary"}
              size="sm"
              disabled={isUpdating}
              onClick={() => toggle("marketing_opt_in")}
            >
              {prefs.marketing_opt_in ? "On" : "Off"}
            </Button>
          </div>
          <div className="flex items-start justify-between gap-3">
            <div className="space-y-1">
              <p className="font-semibold">Transactional opt-in</p>
              <p className="text-sm text-foreground/70">Allow order and system alerts.</p>
            </div>
            <Button
              variant={prefs.transactional_opt_in ? "primary" : "secondary"}
              size="sm"
              disabled={isUpdating}
              onClick={() => toggle("transactional_opt_in")}
            >
              {prefs.transactional_opt_in ? "On" : "Off"}
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
