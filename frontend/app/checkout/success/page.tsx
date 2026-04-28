"use client";
import * as React from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { AuthGate } from "@/components/auth/AuthGate";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { useToast } from "@/components/ui/ToastProvider";
import { apiFetch } from "@/lib/api";
import { formatAddressLine } from "@/lib/address";
import type { OrderDetail } from "@/lib/types";

async function fetchOrder(orderId: string, accessToken?: string | null) {
  const response = await apiFetch<OrderDetail>(`/orders/${orderId}/`, {
    allowGuest: Boolean(accessToken),
    params: accessToken ? { access_token: accessToken } : undefined,
  });
  return response.data;
}

export default function CheckoutSuccessPage() {
  const { hasToken } = useAuthContext();
  const searchParams = useSearchParams();
  const { push } = useToast();
  const orderId = searchParams.get("order_id");
  const orderNumber = searchParams.get("order_number");
  const accessToken = searchParams.get("access_token");
  const allowGuest = Boolean(accessToken);

  const handleCopyOrderNumber = React.useCallback(
    async (value?: string | null) => {
      if (!value) return;
      try {
        await navigator.clipboard.writeText(value);
        push("Order number copied.", "success");
      } catch {
        push("Could not copy order number.", "error");
      }
    },
    [push]
  );

  const orderQuery = useQuery({
    queryKey: ["orders", orderId, accessToken || ""],
    queryFn: () => fetchOrder(orderId as string, accessToken),
    enabled: Boolean(orderId && (hasToken || accessToken)),
  });

  return (
    <AuthGate nextHref="/checkout" allowGuest={allowGuest}>
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-4xl px-3 sm:px-5 py-16">
          <Card variant="bordered" className="space-y-6">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-foreground/60">
                Order confirmed
              </p>
              <h1 className="text-3xl font-semibold">Thank you for your purchase</h1>
              <p className="mt-2 text-sm text-foreground/60">
                We&apos;re processing your order now.
              </p>
            </div>

            {orderQuery.isLoading ? (
              <p className="text-sm text-foreground/60">Loading order details...</p>
            ) : orderQuery.isError ? (
              <div className="space-y-2">
                <p className="text-sm text-foreground/60">
                  We couldn&apos;t load full order details yet.
                </p>
                {orderNumber ? (
                  <button
                    type="button"
                    className="text-sm font-semibold underline underline-offset-4 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                    onClick={() => handleCopyOrderNumber(orderNumber)}
                    title="Copy order number"
                  >
                    Order #{orderNumber}
                  </button>
                ) : null}
                <p className="text-xs text-foreground/60">
                  If you need help, please contact support with your order number.
                </p>
              </div>
            ) : orderQuery.data ? (
              <div className="space-y-3 text-sm">
                <button
                  type="button"
                  className="text-left text-sm text-foreground/60 underline underline-offset-4 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                  onClick={() => handleCopyOrderNumber(orderQuery.data?.order_number)}
                  title="Copy order number"
                >
                  Order #{orderQuery.data.order_number}
                </button>
                {orderQuery.data.payment_status &&
                orderQuery.data.payment_status !== "succeeded" ? (
                  <p className="text-sm text-amber-600">
                    Payment status: {orderQuery.data.payment_status}
                  </p>
                ) : null}
                <p className="text-lg font-semibold">
                  Total {orderQuery.data.total}
                </p>
                <div className="rounded-xl border border-border bg-card p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                    Shipping to
                  </p>
                  <p className="mt-2 font-semibold">
                    {[orderQuery.data.shipping_address?.first_name, orderQuery.data.shipping_address?.last_name]
                      .filter(Boolean)
                      .join(" ") || "Recipient"}
                  </p>
                  <p className="text-foreground/70">
                    {formatAddressLine(orderQuery.data.shipping_address)}
                  </p>
                </div>
              </div>
            ) : orderNumber ? (
              <div className="space-y-2 text-sm">
                <button
                  type="button"
                  className="text-left text-sm text-foreground/60 underline underline-offset-4 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                  onClick={() => handleCopyOrderNumber(orderNumber)}
                  title="Copy order number"
                >
                  Order #{orderNumber}
                </button>
                <p className="text-sm text-foreground/60">
                  Your order is confirmed. We&apos;ll email you with updates.
                </p>
              </div>
            ) : (
              <p className="text-sm text-foreground/60">
                Your order is confirmed.
              </p>
            )}

            <div className="flex flex-wrap gap-3">
              {orderId ? (
                <Button asChild>
                  <Link
                    href={
                      accessToken
                        ? `/orders/${orderId}/?access_token=${encodeURIComponent(accessToken)}`
                        : "/orders/"
                    }
                  >
                    {accessToken ? "View order details" : "View orders"}
                  </Link>
                </Button>
              ) : null}
              <Button asChild variant="secondary">
                <Link href="/">Continue shopping</Link>
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </AuthGate>
  );
}
