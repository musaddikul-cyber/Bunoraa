"use client";

import Link from "next/link";
import { AuthGate } from "@/components/auth/AuthGate";
import { useOrders } from "@/components/orders/useOrders";
import { Card } from "@/components/ui/Card";

export default function OrdersTrackPage() {
  const ordersQuery = useOrders();

  return (
    <AuthGate title="Track orders" description="Sign in to track your orders.">
      <div className="mx-auto w-full max-w-4xl px-6 py-12">
        <div className="mb-6">
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Orders
          </p>
          <h1 className="text-3xl font-semibold">Track an order</h1>
        </div>
        {ordersQuery.isLoading ? (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Loading orders...
          </Card>
        ) : ordersQuery.data?.data?.length ? (
          <div className="space-y-4">
            {ordersQuery.data.data.map((order) => (
              <Card key={order.id} variant="bordered" className="flex items-center justify-between p-4">
                <div>
                  <p className="text-sm text-foreground/60">Order #{order.order_number}</p>
                  <p className="text-sm">{order.status_display || order.status}</p>
                </div>
                <Link className="text-primary" href={`/orders/${order.id}/`}>View details</Link>
              </Card>
            ))}
          </div>
        ) : (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            No orders available for tracking.
          </Card>
        )}
      </div>
    </AuthGate>
  );
}
