"use client";

import * as React from "react";
import Link from "next/link";
import { useOrders } from "@/components/orders/useOrders";
import { Card } from "@/components/ui/Card";

function formatDateTime(value?: string | null) {
  if (!value) return "Unknown time";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export default function AccountOrdersPage() {
  const [query, setQuery] = React.useState("");
  const ordersQuery = useOrders({
    q: query.trim() || undefined,
    ordering: "newest",
  });

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Account
        </p>
        <h1 className="text-3xl font-semibold">Orders</h1>
        <p className="mt-2 text-sm text-foreground/70">
          Track every purchase, delivery state, and timeline.
        </p>
      </div>

      <Card variant="bordered" className="p-4">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by order number or tracking"
          className="h-10 w-full rounded-xl border border-border bg-transparent px-3 text-sm"
        />
      </Card>

      {ordersQuery.isLoading ? (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          Loading orders...
        </Card>
      ) : ordersQuery.isError ? (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          Could not load orders.
        </Card>
      ) : ordersQuery.data?.data?.length ? (
        <div className="space-y-4">
          {ordersQuery.data.data.map((order) => (
            <Card
              key={order.id}
              variant="bordered"
              className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between"
            >
              <div>
                <p className="text-sm text-foreground/60">Order {order.order_number}</p>
                <p className="text-lg font-semibold">
                  {order.status_display || order.status}
                </p>
                <p className="text-xs text-foreground/60">
                  {order.item_count} items • {formatDateTime(order.created_at)}
                </p>
              </div>
              <div className="text-right">
                <p className="text-lg font-semibold">{order.total}</p>
                <Link className="text-sm text-primary" href={`/orders/${order.id}/`}>
                  View details
                </Link>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          You have no orders yet.
        </Card>
      )}
    </div>
  );
}
