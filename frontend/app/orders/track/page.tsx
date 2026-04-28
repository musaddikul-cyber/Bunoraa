"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { AuthGate } from "@/components/auth/AuthGate";
import { useOrders } from "@/components/orders/useOrders";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { apiFetch, ApiError } from "@/lib/api";

type GuestLookupResponse = {
  order_id: string;
  order_number: string;
  access_token: string;
  detail_url: string;
  track_url: string;
  status?: string | null;
  status_display?: string | null;
  tracking_number?: string | null;
  tracking_url?: string | null;
};

export default function OrdersTrackPage() {
  const router = useRouter();
  const { hasToken } = useAuthContext();
  const ordersQuery = useOrders({ enabled: hasToken });
  const [orderNumber, setOrderNumber] = React.useState("");
  const [email, setEmail] = React.useState("");
  const [lookupError, setLookupError] = React.useState<string | null>(null);

  const guestLookup = useMutation({
    mutationFn: async () => {
      const response = await apiFetch<GuestLookupResponse>("/orders/lookup/", {
        method: "POST",
        body: {
          order_number: orderNumber.trim(),
          email: email.trim(),
        },
        allowGuest: true,
      });
      return response.data;
    },
    onSuccess: (data) => {
      setLookupError(null);
      router.push(data.track_url);
    },
    onError: (error) => {
      if (error instanceof ApiError) {
        setLookupError(error.message);
        return;
      }
      setLookupError("We couldn't find that order. Please double-check the details.");
    },
  });

  if (!hasToken) {
    return (
      <div className="mx-auto w-full max-w-4xl px-3 sm:px-5 py-12">
        <div className="mb-6">
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Orders
          </p>
          <h1 className="text-3xl font-semibold">Track a guest order</h1>
          <p className="mt-2 text-sm text-foreground/70">
            Enter the order number and email you used at checkout.
          </p>
        </div>

        <Card variant="bordered" className="space-y-4 p-6">
          <label className="block text-sm">
            Order number
            <input
              type="text"
              value={orderNumber}
              onChange={(event) => setOrderNumber(event.target.value)}
              placeholder="ORD-20260428-ABCD"
              className="mt-2 h-10 w-full rounded-xl border border-border bg-card px-3 text-sm"
            />
          </label>
          <label className="block text-sm">
            Email
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              className="mt-2 h-10 w-full rounded-xl border border-border bg-card px-3 text-sm"
            />
          </label>
          {lookupError ? (
            <p className="text-sm text-rose-500">{lookupError}</p>
          ) : null}
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <Button
              type="button"
              onClick={() => guestLookup.mutate()}
              disabled={!orderNumber.trim() || !email.trim() || guestLookup.isPending}
            >
              {guestLookup.isPending ? "Looking up..." : "Track order"}
            </Button>
            <Link
              href="/account/login/?next=%2Forders%2Ftrack%2F"
              className="text-sm text-primary"
            >
              Sign in for account orders
            </Link>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <AuthGate title="Track orders" description="Sign in to track your orders.">
      <div className="mx-auto w-full max-w-4xl px-3 sm:px-5 py-12">
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
              <Card
                key={order.id}
                variant="bordered"
                className="flex items-center justify-between p-4"
              >
                <div>
                  <p className="text-sm text-foreground/60">Order #{order.order_number}</p>
                  <p className="text-sm">{order.status_display || order.status}</p>
                </div>
                <Link className="text-primary" href={`/orders/${order.id}/`}>
                  View details
                </Link>
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
