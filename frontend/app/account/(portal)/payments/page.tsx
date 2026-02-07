"use client";

import * as React from "react";
import { usePaymentMethods } from "@/components/payments/usePaymentMethods";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export default function PaymentsPage() {
  const { methodsQuery, removeMethod, setDefault, saveMethod, setupIntent } =
    usePaymentMethods();
  const [paymentMethodId, setPaymentMethodId] = React.useState("");
  const [clientSecret, setClientSecret] = React.useState<string | null>(null);

  const handleAdd = async () => {
    if (!paymentMethodId) return;
    await saveMethod.mutateAsync(paymentMethodId);
    setPaymentMethodId("");
  };

  const handleSetupIntent = async () => {
    const response = await setupIntent.mutateAsync();
    setClientSecret(response.client_secret);
  };

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Account
        </p>
        <h1 className="text-3xl font-semibold">Payments</h1>
        <p className="mt-2 text-sm text-foreground/70">
          Manage saved payment methods and default billing preferences.
        </p>
      </div>

      <Card variant="bordered" className="space-y-4">
        <h2 className="text-lg font-semibold">Saved methods</h2>
        {methodsQuery.isLoading ? (
          <p className="text-sm text-foreground/70">Loading payment methods...</p>
        ) : methodsQuery.data?.length ? (
          <div className="space-y-3">
            {methodsQuery.data.map((method) => (
              <div
                key={method.id}
                className="flex flex-col gap-3 rounded-xl border border-border p-4 sm:flex-row sm:items-center sm:justify-between"
              >
                <div>
                  <p className="text-sm text-foreground/60">
                    {method.type_display || "Payment method"}
                  </p>
                  <p className="text-base font-semibold">
                    {method.display_name || "Saved method"}
                  </p>
                  {method.is_default ? (
                    <span className="text-xs uppercase tracking-[0.2em] text-primary">
                      Default
                    </span>
                  ) : null}
                </div>
                <div className="flex flex-wrap gap-2">
                  {!method.is_default ? (
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => setDefault.mutate(method.id)}
                    >
                      Set default
                    </Button>
                  ) : null}
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => removeMethod.mutate(method.id)}
                  >
                    Remove
                  </Button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-foreground/70">
            No saved payment methods.
          </p>
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card variant="bordered" className="space-y-4">
          <h2 className="text-lg font-semibold">Add a payment method</h2>
          <p className="text-sm text-foreground/70">
            Paste a Stripe payment method ID to attach it to your account.
          </p>
          <label className="block text-sm">
            Payment method ID
            <input
              className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
              value={paymentMethodId}
              onChange={(event) => setPaymentMethodId(event.target.value)}
              placeholder="pm_..."
            />
          </label>
          <Button
            onClick={handleAdd}
            disabled={saveMethod.isPending || !paymentMethodId}
          >
            {saveMethod.isPending ? "Saving..." : "Save method"}
          </Button>
        </Card>

        <Card variant="bordered" className="space-y-4">
          <h2 className="text-lg font-semibold">Setup intent</h2>
          <p className="text-sm text-foreground/70">
            Generate a client secret for Stripe Elements integration.
          </p>
          <Button
            variant="secondary"
            onClick={handleSetupIntent}
            disabled={setupIntent.isPending}
          >
            {setupIntent.isPending ? "Generating..." : "Generate setup intent"}
          </Button>
          {clientSecret ? (
            <div className="rounded-xl border border-border bg-muted p-3 text-xs break-all">
              {clientSecret}
            </div>
          ) : null}
        </Card>
      </div>
    </div>
  );
}
