"use client";

import { useNotificationPreferences } from "@/components/notifications/useNotificationPreferences";
import type { NotificationPreference } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export default function NotificationPreferencesPage() {
  const { preferencesQuery, updatePreferences } = useNotificationPreferences();
  const prefs: NotificationPreference = preferencesQuery.data ?? {};

  if (preferencesQuery.isLoading) {
    return (
      <Card variant="bordered" className="p-6 text-sm text-foreground/70">
        Loading notification preferences...
      </Card>
    );
  }

  const toggle = (field: keyof NotificationPreference) => {
    updatePreferences.mutate({ [field]: !prefs[field] });
  };

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Account
        </p>
        <h1 className="text-3xl font-semibold">Notifications</h1>
        <p className="mt-2 text-sm text-foreground/70">
          Control how you receive alerts and promotions.
        </p>
      </div>

      <Card variant="bordered" className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-semibold">Email order updates</p>
            <p className="text-sm text-foreground/70">Shipping, delivery, and status alerts.</p>
          </div>
          <Button
            variant={prefs.email_order_updates ? "primary" : "secondary"}
            size="sm"
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
            onClick={() => toggle("email_back_in_stock")}
          >
            {prefs.email_back_in_stock ? "On" : "Off"}
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
            onClick={() => toggle("push_enabled")}
          >
            {prefs.push_enabled ? "On" : "Off"}
          </Button>
        </div>
      </Card>
    </div>
  );
}
