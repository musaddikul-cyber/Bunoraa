"use client";

import Link from "next/link";
import { useAuth } from "@/components/auth/useAuth";
import { useOrders } from "@/components/orders/useOrders";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export default function DashboardPage() {
  const { profileQuery } = useAuth();
  const ordersQuery = useOrders();

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Account
        </p>
        <h1 className="text-3xl font-semibold">Dashboard</h1>
        <p className="mt-2 text-foreground/70">
          Welcome back, {profileQuery.data?.first_name || "there"}.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card variant="bordered" className="space-y-3">
          <h2 className="text-lg font-semibold">Quick links</h2>
          <div className="flex flex-wrap gap-2">
            <Button asChild variant="secondary" size="sm">
              <Link href="/account/profile/">Profile</Link>
            </Button>
            <Button asChild variant="secondary" size="sm">
              <Link href="/account/addresses/">Addresses</Link>
            </Button>
            <Button asChild variant="secondary" size="sm">
              <Link href="/account/orders/">Orders</Link>
            </Button>
            <Button asChild variant="secondary" size="sm">
              <Link href="/wishlist/">Wishlist</Link>
            </Button>
          </div>
        </Card>

        <Card variant="bordered" className="space-y-3">
          <h2 className="text-lg font-semibold">Recent orders</h2>
          {ordersQuery.isLoading ? (
            <p className="text-sm text-foreground/60">Loading orders...</p>
          ) : ordersQuery.data?.data?.length ? (
            <ul className="space-y-2 text-sm">
              {ordersQuery.data.data.slice(0, 3).map((order) => (
                <li key={order.id} className="flex items-center justify-between">
                  <span>#{order.order_number}</span>
                  <Link className="text-primary" href={`/orders/${order.id}/`}>
                    View
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-foreground/60">No orders yet.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
