"use client";

import Link from "next/link";
import { useOrders } from "@/components/orders/useOrders";
import { Card } from "@/components/ui/Card";

export default function AccountOrdersPage() {
  const ordersQuery = useOrders();

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Account
        </p>
        <h1 className="text-3xl font-semibold">Orders</h1>
        <p className="mt-2 text-sm text-foreground/70">
          Track recent purchases and delivery status.
        </p>
      </div>

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
                <p className="text-sm text-foreground/60">
                  Order {order.order_number}
                </p>
                <p className="text-lg font-semibold">
                  {order.status_display || order.status}
                </p>
                <p className="text-xs text-foreground/60">
                  {order.item_count} items
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
