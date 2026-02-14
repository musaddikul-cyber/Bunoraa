"use client";

import * as React from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { formatMoney } from "@/lib/checkout";
import { cn } from "@/lib/utils";
import type { ShippingMethodOption, StoreLocation } from "@/lib/types";

type CheckoutShippingStepProps = {
  shippingRates: ShippingMethodOption[];
  shippingRatesLoading?: boolean;
  shippingRatesError?: string | null;
  pickupLocations: StoreLocation[];
  defaultShippingType?: "delivery" | "pickup";
  defaultRateId?: string | null;
  defaultMethodCode?: string | null;
  defaultPickupId?: string | null;
  defaultInstructions?: string | null;
  currencyCode?: string;
  onSubmit: (payload: {
    shipping_type: "delivery" | "pickup";
    shipping_rate_id?: string;
    pickup_location_id?: string;
    delivery_instructions?: string;
  }) => Promise<void>;
  onSelectionChange?: (payload: {
    shipping_type: "delivery" | "pickup";
    shipping_rate_id?: string;
    pickup_location_id?: string;
  }) => void;
  onBack: () => void;
  isSubmitting?: boolean;
  isAutoSaving?: boolean;
};

export function CheckoutShippingStep({
  shippingRates,
  shippingRatesLoading,
  shippingRatesError,
  pickupLocations,
  defaultShippingType = "delivery",
  defaultRateId,
  defaultMethodCode,
  defaultPickupId,
  defaultInstructions,
  currencyCode,
  onSubmit,
  onSelectionChange,
  onBack,
  isSubmitting,
  isAutoSaving,
}: CheckoutShippingStepProps) {
  const [shippingType, setShippingType] = React.useState<"delivery" | "pickup">(
    defaultShippingType
  );
  const [selectedRateId, setSelectedRateId] = React.useState(defaultRateId || "");
  const [selectedPickupId, setSelectedPickupId] = React.useState(
    defaultPickupId || ""
  );
  const [deliveryInstructions, setDeliveryInstructions] = React.useState(
    defaultInstructions || ""
  );
  const [error, setError] = React.useState<string | null>(null);
  const visibleRates = React.useMemo(() => {
    if (!shippingRates.length) return [];
    if (!selectedRateId) return shippingRates.slice(0, 5);
    const selectedIndex = shippingRates.findIndex(
      (rate) => (rate.rate_id || rate.id) === selectedRateId
    );
    const base = shippingRates.slice(0, 5);
    if (selectedIndex === -1) return base;
    const alreadyVisible = base.some(
      (rate) => (rate.rate_id || rate.id) === selectedRateId
    );
    if (alreadyVisible) return base;
    const selectedRate = shippingRates[selectedIndex];
    return [selectedRate, ...base.slice(0, 4)];
  }, [shippingRates, selectedRateId]);

  const formatHours = React.useCallback((hours: StoreLocation["hours"]) => {
    if (!hours) return "";
    if (typeof hours === "string") return hours;
    if (typeof hours !== "object") return "";
    const entries = Object.entries(hours)
      .filter(([, value]) => value)
      .map(([day, value]) => {
        const label = day.replace(/\b\w/g, (char) => char.toUpperCase());
        return `${label}: ${value}`;
      });
    return entries.join(" • ");
  }, []);

  React.useEffect(() => {
    if (!onSelectionChange) return;
    if (shippingType === "delivery") {
      if (!selectedRateId) return;
      onSelectionChange({
        shipping_type: "delivery",
        shipping_rate_id: selectedRateId,
      });
      return;
    }
    if (!selectedPickupId) return;
    onSelectionChange({
      shipping_type: "pickup",
      pickup_location_id: selectedPickupId,
    });
  }, [shippingType, selectedRateId, selectedPickupId, onSelectionChange]);

  React.useEffect(() => {
    setShippingType(defaultShippingType);
  }, [defaultShippingType]);

  React.useEffect(() => {
    if (defaultRateId) {
      setSelectedRateId(defaultRateId);
    }
  }, [defaultRateId]);

  React.useEffect(() => {
    if (!defaultMethodCode) return;
    if (selectedRateId) return;
    const match = shippingRates.find(
      (rate) =>
        rate.code === defaultMethodCode ||
        rate.name === defaultMethodCode ||
        rate.method_id === defaultMethodCode
    );
    const nextId = match?.rate_id || match?.id || "";
    if (nextId) setSelectedRateId(nextId);
  }, [defaultMethodCode, shippingRates, selectedRateId]);

  React.useEffect(() => {
    if (defaultPickupId) {
      setSelectedPickupId(defaultPickupId);
    }
  }, [defaultPickupId]);

  React.useEffect(() => {
    if (shippingType !== "delivery") return;
    if (!visibleRates.length) return;
    const selectedVisible = visibleRates.some(
      (rate) => (rate.rate_id || rate.id) === selectedRateId
    );
    if (!selectedRateId || !selectedVisible) {
      const firstRate = visibleRates[0];
      const nextId = firstRate?.rate_id || firstRate?.id || "";
      if (nextId) {
        setSelectedRateId(nextId);
      }
    }
  }, [visibleRates, selectedRateId, shippingType]);

  React.useEffect(() => {
    if (!selectedPickupId && pickupLocations.length) {
      setSelectedPickupId(pickupLocations[0].id);
    }
  }, [pickupLocations, selectedPickupId]);

  const handleSubmit = async () => {
    setError(null);
    if (shippingType === "delivery") {
      if (!selectedRateId) {
        setError("Select a delivery method to continue.");
        return;
      }
      await onSubmit({
        shipping_type: "delivery",
        shipping_rate_id: selectedRateId,
        delivery_instructions: deliveryInstructions.trim() || undefined,
      });
      return;
    }

    if (!selectedPickupId && pickupLocations.length) {
      setError("Select a pickup location to continue.");
      return;
    }

    await onSubmit({
      shipping_type: "pickup",
      pickup_location_id: selectedPickupId || undefined,
      delivery_instructions: deliveryInstructions.trim() || undefined,
    });
  };

  return (
    <Card variant="bordered" className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-foreground/60">
          Step 2
        </p>
        <h2 className="text-xl font-semibold">Shipping method</h2>
        <p className="text-sm text-foreground/60">
          Choose delivery or pickup that works best for you.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {(["delivery", "pickup"] as const).map((type) => (
          <button
            key={type}
            type="button"
            className={cn(
              "rounded-xl border px-4 py-3 text-left text-sm transition",
              shippingType === type
                ? "border-primary bg-primary/10"
                : "border-border bg-card hover:bg-muted"
            )}
            onClick={() => setShippingType(type)}
          >
            <p className="font-semibold">
              {type === "delivery" ? "Delivery" : "Store pickup"}
            </p>
            <p className="text-xs text-foreground/60">
              {type === "delivery"
                ? "Ship to your address with live rates."
                : "Collect from a nearby pickup point."}
            </p>
          </button>
        ))}
      </div>

      {shippingType === "delivery" ? (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <p className="font-semibold">Delivery options</p>
            {shippingRatesLoading ? (
              <span className="text-xs text-foreground/60">Fetching rates...</span>
            ) : isAutoSaving ? (
              <span className="text-xs text-foreground/60">Saving...</span>
            ) : null}
          </div>
          {shippingRatesError ? (
            <p className="text-xs text-rose-500">{shippingRatesError}</p>
          ) : null}
          {shippingRatesLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((idx) => (
                <div key={idx} className="h-14 w-full rounded-xl bg-muted animate-pulse" />
              ))}
            </div>
          ) : visibleRates.length ? (
            <div className="space-y-2">
              {visibleRates.map((rate) => (
                <label
                  key={rate.rate_id || rate.id}
                  className={cn(
                    "flex cursor-pointer flex-col gap-3 rounded-xl border px-4 py-3 text-sm sm:flex-row sm:items-center sm:justify-between",
                    selectedRateId === (rate.rate_id || rate.id)
                      ? "border-primary bg-primary/10"
                      : "border-border bg-card hover:bg-muted"
                  )}
                >
                  <div className="flex items-start gap-3">
                    <input
                      type="radio"
                      className="mt-1 h-4 w-4"
                      name="shipping_rate"
                      checked={selectedRateId === (rate.rate_id || rate.id)}
                      onChange={() =>
                        setSelectedRateId(rate.rate_id || rate.id || "")
                      }
                    />
                    <div>
                      <p className="font-semibold">{rate.name}</p>
                      {rate.description ? (
                        <p className="text-xs text-foreground/60">{rate.description}</p>
                      ) : null}
                      <p className="text-xs text-foreground/60">
                        {rate.delivery_estimate || "Delivery estimate available at checkout"}
                      </p>
                      {rate.carrier?.name ? (
                        <p className="text-xs text-foreground/50">
                          Carrier: {rate.carrier.name}
                        </p>
                      ) : null}
                    </div>
                  </div>
                  <div className="self-end text-right sm:self-auto">
                    <p className="font-semibold">{rate.rate_display}</p>
                    {rate.is_express ? (
                      <span className="text-xs uppercase tracking-wide text-primary">
                        Express
                      </span>
                    ) : null}
                  </div>
                </label>
              ))}
            </div>
          ) : (
            <p className="text-sm text-foreground/60">
              No delivery methods available for this address.
            </p>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <p className="font-semibold">Pickup locations</p>
            {isAutoSaving ? (
              <span className="text-xs text-foreground/60">Saving...</span>
            ) : null}
          </div>
          {pickupLocations.length ? (
            <div className="space-y-2">
              {pickupLocations.map((location) => (
                <label
                  key={location.id}
                  className={cn(
                    "flex cursor-pointer items-start gap-3 rounded-xl border px-4 py-3 text-sm",
                    selectedPickupId === location.id
                      ? "border-primary bg-primary/10"
                      : "border-border bg-card hover:bg-muted"
                  )}
                >
                  <input
                    type="radio"
                    className="mt-1 h-4 w-4"
                    name="pickup_location"
                    checked={selectedPickupId === location.id}
                    onChange={() => setSelectedPickupId(location.id)}
                  />
                  <div>
                    <p className="font-semibold">{location.name}</p>
                    <p className="text-xs text-foreground/60">
                      {location.full_address ||
                        location.address ||
                        [
                          location.address_line1,
                          location.address_line2,
                          location.city,
                          location.state,
                          location.postal_code,
                          location.country,
                        ]
                          .filter(Boolean)
                          .join(", ")}
                    </p>
                    {location.phone || location.email ? (
                      <p className="text-xs text-foreground/50">
                        {[location.phone, location.email].filter(Boolean).join(" • ")}
                      </p>
                    ) : null}
                    {(() => {
                      const hoursText = formatHours(location.hours);
                      if (!hoursText) return null;
                      return (
                        <p className="text-xs text-foreground/50">
                          Hours: {hoursText}
                        </p>
                      );
                    })()}
                    {location.min_pickup_time_hours ? (
                      <p className="text-xs text-foreground/50">
                        Ready in ~{location.min_pickup_time_hours} hours
                      </p>
                    ) : null}
                    {location.max_hold_days ? (
                      <p className="text-xs text-foreground/50">
                        Held for up to {location.max_hold_days} days
                      </p>
                    ) : null}
                  </div>
                  <div className="ml-auto text-right text-xs text-foreground/60">
                    {location.pickup_fee !== undefined && location.pickup_fee !== null ? (
                      <span>
                        {Number(location.pickup_fee) > 0
                          ? `Fee ${formatMoney(
                              location.pickup_fee,
                              currencyCode
                            )}`
                          : "Free"}
                      </span>
                    ) : null}
                  </div>
                </label>
              ))}
            </div>
          ) : (
            <p className="text-sm text-foreground/60">
              No pickup locations available yet.
            </p>
          )}
        </div>
      )}

      <div className="space-y-2">
        <label className="block text-sm">
          Delivery instructions (optional)
          <textarea
            rows={3}
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
            autoComplete="shipping street-address"
            value={deliveryInstructions}
            onChange={(event) => setDeliveryInstructions(event.target.value)}
          />
        </label>
      </div>

      {error ? <p className="text-sm text-rose-500">{error}</p> : null}

      <div className="flex flex-col gap-3 sm:flex-row sm:justify-between">
        <Button type="button" variant="secondary" className="w-full sm:w-auto" onClick={onBack}>
          Back
        </Button>
        <Button
          type="button"
          className="w-full sm:w-auto sm:min-w-[220px]"
          onClick={handleSubmit}
          disabled={isSubmitting}
        >
          {isSubmitting ? "Saving..." : "Continue to payment"}
        </Button>
      </div>
    </Card>
  );
}
