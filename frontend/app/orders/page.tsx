"use client";

import * as React from "react";
import Link from "next/link";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";
import { useOrders } from "@/components/orders/useOrders";

const STATUS_OPTIONS = [
  { value: "all", label: "All statuses" },
  { value: "pending", label: "Pending" },
  { value: "confirmed", label: "Confirmed" },
  { value: "processing", label: "Processing" },
  { value: "shipped", label: "Shipped" },
  { value: "delivered", label: "Delivered" },
  { value: "cancelled", label: "Cancelled" },
  { value: "refunded", label: "Refunded" },
];

const SORT_OPTIONS = [
  { value: "newest", label: "Newest first" },
  { value: "oldest", label: "Oldest first" },
  { value: "total_high", label: "Highest total" },
  { value: "total_low", label: "Lowest total" },
  { value: "status", label: "Status" },
];

function formatDateTime(value?: string | null) {
  if (!value) return "Unknown time";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export default function OrdersPage() {
  const [query, setQuery] = React.useState("");
  const [status, setStatus] = React.useState("all");
  const [sort, setSort] = React.useState("newest");

  const trimmedQuery = query.trim();
  const ordersQuery = useOrders({
    q: trimmedQuery || undefined,
    status: status === "all" ? undefined : status,
    ordering: sort || undefined,
  });

  return (
    <AuthGate title="Orders" description="Sign in to view your orders.">
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-5xl px-4 py-12 sm:px-6">
          <div className="mb-6">
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Orders
            </p>
            <h1 className="text-2xl font-semibold sm:text-3xl">Your orders</h1>
            <p className="mt-2 text-sm text-foreground/70">
              Search, sort, and track every order with timestamped updates.
            </p>
          </div>

          <Card variant="bordered" className="mb-6 grid gap-3 p-4 sm:grid-cols-3">
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search by order number or tracking"
              className="h-10 rounded-xl border border-border bg-transparent px-3 text-sm"
            />
            <select
              value={status}
              onChange={(event) => setStatus(event.target.value)}
              className="h-10 rounded-xl border border-border bg-card px-3 text-sm text-foreground"
            >
              {STATUS_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <select
              value={sort}
              onChange={(event) => setSort(event.target.value)}
              className="h-10 rounded-xl border border-border bg-card px-3 text-sm text-foreground"
            >
              {SORT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </Card>

          {ordersQuery.isLoading ? (
            <p className="text-sm text-foreground/60">Loading orders...</p>
          ) : ordersQuery.isError ? (
            <p className="text-sm text-foreground/60">Could not load orders.</p>
          ) : ordersQuery.data?.data?.length ? (
            <div className="space-y-4">
              {ordersQuery.data.data.map((order) => (
                <Card key={order.id} variant="bordered" className="p-4 sm:p-5">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                    <div className="space-y-1">
                      <p className="text-xs uppercase tracking-[0.18em] text-foreground/60">
                        Order {order.order_number}
                      </p>
                      <p className="text-lg font-semibold">
                        {order.status_display || order.status}
                      </p>
                      <p className="text-sm text-foreground/65">
                        Placed on {formatDateTime(order.created_at)}
                      </p>
                    </div>
                    <div className="text-left sm:text-right">
                      <p className="text-sm text-foreground/60">{order.item_count} items</p>
                      <p className="text-lg font-semibold">{order.total}</p>
                      <Link className="text-sm text-primary" href={`/orders/${order.id}/`}>
                        View details
                      </Link>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <Card variant="bordered" className="p-6">
              <p className="text-sm text-foreground/60">No orders found for this filter.</p>
            </Card>
          )}
        </div>
      </div>
    </AuthGate>
  );
}
