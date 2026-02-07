"use client";

import Link from "next/link";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";
import { useOrders } from "@/components/orders/useOrders";

export default function OrdersPage() {
  const ordersQuery = useOrders();

  return (
    <AuthGate title="Orders" description="Sign in to view your orders.">
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-4xl px-6 py-16">
          <div className="mb-8">
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Orders
            </p>
            <h1 className="text-2xl font-semibold">Your orders</h1>
          </div>

          {ordersQuery.isLoading ? (
            <p className="text-sm text-foreground/60">Loading orders...</p>
          ) : ordersQuery.isError ? (
            <p className="text-sm text-foreground/60">Could not load orders.</p>
          ) : ordersQuery.data?.data?.length ? (
            <div className="space-y-4">
              {ordersQuery.data.data.map((order) => (
                <Card key={order.id} variant="bordered" className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-foreground/60">Order {order.order_number}</p>
                    <p className="text-lg font-semibold">{order.status_display || order.status}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-foreground/60">{order.item_count} items</p>
                    <p className="text-lg font-semibold">{order.total}</p>
                    <Link className="text-sm text-primary" href={`/orders/${order.id}/`}>
                      View
                    </Link>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <Card variant="bordered">
              <p className="text-sm text-foreground/60">You have no orders yet.</p>
            </Card>
          )}
        </div>
      </div>
    </AuthGate>
  );
}
