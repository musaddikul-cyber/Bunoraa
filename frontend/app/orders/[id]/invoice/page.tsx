"use client";

import * as React from "react";
import { useParams } from "next/navigation";
import { AuthGate } from "@/components/auth/AuthGate";
import { useOrders } from "@/components/orders/useOrders";
import { Card } from "@/components/ui/Card";
import { isUuid, resolveOrderId } from "@/lib/orders";

export default function OrderInvoicePage() {
  const params = useParams();
  const rawId = params?.id as string;
  const ordersQuery = useOrders();

  const resolvedId = React.useMemo(() => {
    if (!rawId) return null;
    if (isUuid(rawId)) return rawId;
    const list = ordersQuery.data?.data ?? [];
    return resolveOrderId(rawId, list);
  }, [rawId, ordersQuery.data]);

  return (
    <AuthGate title="Invoice" description="Sign in to view invoices.">
      <div className="mx-auto w-full max-w-3xl px-6 py-12">
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
            Invoice not available for this order.
          </Card>
        ) : (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Invoice rendering for order #{rawId} is not available via API yet.
          </Card>
        )}
      </div>
    </AuthGate>
  );
}
