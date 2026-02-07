"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { Preorder } from "@/lib/types";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";
import { formatMoney } from "@/lib/checkout";
import { usePreorderStatistics } from "@/components/preorders/usePreorderData";

const formatDate = (value?: string | null) => {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
};

export default function MyPreordersPage() {
  const statsQuery = usePreorderStatistics();
  const preordersQuery = useQuery({
    queryKey: ["preorders", "list"],
    queryFn: async () => {
      const response = await apiFetch<Preorder[]>("/preorders/orders/");
      return response.data;
    },
  });

  return (
    <AuthGate title="Preorders" description="Sign in to view your preorders.">
      <div className="mx-auto w-full max-w-6xl px-4 py-10 sm:px-6 sm:py-12">
        <div className="mb-6">
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Preorders
          </p>
          <h1 className="text-3xl font-semibold">My preorders</h1>
          <p className="mt-2 text-sm text-foreground/60">
            Track progress, review quotes, and manage payments for custom work.
          </p>
        </div>

        {statsQuery.data ? (
          <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Card variant="bordered" className="space-y-1 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                Total
              </p>
              <p className="text-xl font-semibold">{statsQuery.data.total ?? 0}</p>
              <p className="text-xs text-foreground/60">
                Value {formatMoney(statsQuery.data.total_value || 0, "BDT")}
              </p>
            </Card>
            <Card variant="bordered" className="space-y-1 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                Pending
              </p>
              <p className="text-xl font-semibold">
                {statsQuery.data.pending ?? 0}
              </p>
              <p className="text-xs text-foreground/60">Quotes in review</p>
            </Card>
            <Card variant="bordered" className="space-y-1 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                In production
              </p>
              <p className="text-xl font-semibold">
                {statsQuery.data.in_production ?? 0}
              </p>
              <p className="text-xs text-foreground/60">Active work</p>
            </Card>
            <Card variant="bordered" className="space-y-1 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                Delivered
              </p>
              <p className="text-xl font-semibold">
                {statsQuery.data.delivered ?? 0}
              </p>
              <p className="text-xs text-foreground/60">Completed orders</p>
            </Card>
          </div>
        ) : null}

        {preordersQuery.isLoading ? (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Loading preorders...
          </Card>
        ) : preordersQuery.data?.length ? (
          <div className="grid gap-4 md:grid-cols-2">
            {preordersQuery.data.map((order) => (
              <Card key={order.id} variant="bordered" className="space-y-4 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                  <span className="text-foreground/60">
                    #{order.preorder_number}
                  </span>
                  <span className="rounded-full bg-muted px-2 py-1 text-xs">
                    {order.status_display || order.status}
                  </span>
                </div>
                <div className="space-y-1 text-sm text-foreground/70">
                  {order.category_name ? <p>{order.category_name}</p> : null}
                  <p>Created: {formatDate(order.created_at)}</p>
                  <p>
                    Total:{" "}
                    {formatMoney(
                      order.total_amount || order.estimated_price || "0",
                      order.currency || undefined
                    )}
                  </p>
                  {order.deposit_required ? (
                    <p>
                      Deposit due:{" "}
                      {formatMoney(order.deposit_required, order.currency || undefined)}
                    </p>
                  ) : null}
                  {order.amount_remaining ? (
                    <p>
                      Remaining:{" "}
                      {formatMoney(order.amount_remaining, order.currency || undefined)}
                    </p>
                  ) : null}
                </div>
                <Link
                  className="text-sm font-semibold text-primary"
                  href={`/preorders/order/${order.preorder_number}/`}
                >
                  View details
                </Link>
              </Card>
            ))}
          </div>
        ) : (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            No preorders found. Start a custom request to see it here.
          </Card>
        )}
      </div>
    </AuthGate>
  );
}
