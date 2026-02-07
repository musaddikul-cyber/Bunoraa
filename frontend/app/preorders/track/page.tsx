"use client";

import * as React from "react";
import { useSearchParams } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { useToast } from "@/components/ui/ToastProvider";
import { usePreorderTracking } from "@/components/preorders/usePreorderData";
import { formatMoney } from "@/lib/checkout";
import type { Preorder } from "@/lib/types";

const formatDateTime = (value?: string | null) => {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
};

const getErrorMessage = (error: unknown) =>
  error instanceof Error ? error.message : "Unable to track preorder.";

export default function PreorderTrackPage() {
  const searchParams = useSearchParams();
  const { push } = useToast();
  const tracking = usePreorderTracking();
  const [form, setForm] = React.useState({ preorder_number: "", email: "" });
  const [result, setResult] = React.useState<Preorder | null>(null);

  React.useEffect(() => {
    const number = searchParams.get("order") || searchParams.get("preorder");
    if (number && !form.preorder_number) {
      setForm((prev) => ({ ...prev, preorder_number: number }));
    }
  }, [searchParams, form.preorder_number]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!form.preorder_number.trim() || !form.email.trim()) {
      push("Enter your preorder number and email.", "error");
      return;
    }
    try {
      const data = await tracking.mutateAsync({
        preorder_number: form.preorder_number.trim(),
        email: form.email.trim(),
      });
      setResult(data);
    } catch (error) {
      push(getErrorMessage(error), "error");
      setResult(null);
    }
  };

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-12 sm:px-6 sm:py-16">
      <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
        <Card variant="bordered" className="space-y-4">
          <h1 className="text-2xl font-semibold">Track preorder</h1>
          <p className="text-sm text-foreground/70">
            Enter your preorder number and the email used during submission.
          </p>
          <form className="space-y-4" onSubmit={handleSubmit}>
            <label className="block text-sm">
              Preorder number
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                placeholder="PRE-2026-0001"
                value={form.preorder_number}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, preorder_number: event.target.value }))
                }
              />
            </label>
            <label className="block text-sm">
              Email
              <input
                type="email"
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                placeholder="you@example.com"
                value={form.email}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, email: event.target.value }))
                }
              />
            </label>
            <Button type="submit" className="w-full" disabled={tracking.isPending}>
              {tracking.isPending ? "Checking..." : "Track preorder"}
            </Button>
          </form>
        </Card>

        <Card variant="bordered" className="space-y-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
              Status
            </p>
            <h2 className="text-lg font-semibold">
              {result ? result.status_display || result.status : "Awaiting lookup"}
            </h2>
          </div>
          {result ? (
            <div className="space-y-4 text-sm text-foreground/70">
              <div className="rounded-xl border border-border bg-muted/30 p-3">
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                  Preorder
                </p>
                <p className="mt-1 font-semibold">#{result.preorder_number}</p>
                <p>Quantity: {result.quantity}</p>
                <p>
                  Estimated total: {formatMoney(result.estimated_price || 0, result.currency)}
                </p>
                <p>
                  Deposit required: {formatMoney(result.deposit_required || 0, result.currency)}
                </p>
              </div>

              {result.status_history?.length ? (
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                    Timeline
                  </p>
                  <div className="mt-2 space-y-2">
                    {[...result.status_history].reverse().map((entry) => (
                      <div key={entry.id} className="border-l-2 border-primary/40 pl-3">
                        <p className="text-xs text-foreground/60">
                          {formatDateTime(entry.created_at)}
                        </p>
                        <p className="font-medium">
                          {entry.to_status_display || entry.to_status}
                        </p>
                        {entry.notes ? (
                          <p className="text-xs text-foreground/60">{entry.notes}</p>
                        ) : null}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {result.quotes?.length ? (
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                    Quotes
                  </p>
                  <div className="mt-2 space-y-2">
                    {result.quotes.map((quote) => (
                      <div key={quote.id} className="rounded-xl border border-border p-3">
                        <p className="font-semibold">{quote.quote_number}</p>
                        <p>Status: {quote.status}</p>
                        <p>Total: {formatMoney(quote.total || 0, result.currency)}</p>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {result.payments?.length ? (
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                    Payments
                  </p>
                  <div className="mt-2 space-y-2">
                    {result.payments.map((payment) => (
                      <div key={payment.id} className="flex items-center justify-between">
                        <span>{payment.payment_type}</span>
                        <span>{formatMoney(payment.amount || 0, result.currency)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          ) : (
            <p className="text-sm text-foreground/60">
              Submit the form to see your preorder timeline and quote status.
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}
