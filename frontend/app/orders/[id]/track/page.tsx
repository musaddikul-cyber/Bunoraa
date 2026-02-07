"use client";

import * as React from "react";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { AuthGate } from "@/components/auth/AuthGate";
import { useOrders } from "@/components/orders/useOrders";
import { apiFetch } from "@/lib/api";
import { Card } from "@/components/ui/Card";
import { isUuid, resolveOrderId } from "@/lib/orders";

export default function OrderTrackPage() {
  const params = useParams();
  const rawId = params?.id as string;
  const ordersQuery = useOrders();

  const resolvedId = React.useMemo(() => {
    if (!rawId) return null;
    if (isUuid(rawId)) return rawId;
    const list = ordersQuery.data?.data ?? [];
    return resolveOrderId(rawId, list);
  }, [rawId, ordersQuery.data]);

  const trackQuery = useQuery({
    queryKey: ["orders", resolvedId, "track"],
    queryFn: async () => {
      const response = await apiFetch<Record<string, unknown>>(
        `/orders/${resolvedId}/track/`
      );
      return response.data;
    },
    enabled: Boolean(resolvedId),
  });

  return (
    <AuthGate title="Track order" description="Sign in to track your order.">
      <div className="mx-auto w-full max-w-3xl px-6 py-12">
        <div className="mb-6">
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Order tracking
          </p>
          <h1 className="text-3xl font-semibold">#{rawId || ""}</h1>
        </div>

        {!rawId ? (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Missing order identifier.
          </Card>
        ) : !isUuid(rawId) && ordersQuery.isLoading ? (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Resolving order number...
          </Card>
        ) : !resolvedId ? (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Order not found in your account.
          </Card>
        ) : trackQuery.isLoading ? (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Loading tracking information...
          </Card>
        ) : trackQuery.data ? (
          <Card variant="bordered" className="space-y-2 p-6 text-sm text-foreground/70">
            <p>
              Status: {String(trackQuery.data.status_display || trackQuery.data.status)}
            </p>
            <p>
              Tracking number: {String(trackQuery.data.tracking_number || "Pending")}
            </p>
            {trackQuery.data.tracking_url ? (
              <a className="text-primary" href={String(trackQuery.data.tracking_url)}>
                View carrier tracking
              </a>
            ) : null}
          </Card>
        ) : (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Tracking information is not available.
          </Card>
        )}
      </div>
    </AuthGate>
  );
}
