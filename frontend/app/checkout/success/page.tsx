"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { apiFetch } from "@/lib/api";
import { formatAddressLine } from "@/lib/address";
import type { OrderDetail } from "@/lib/types";

async function fetchOrder(orderId: string) {
  const response = await apiFetch<OrderDetail>(`/orders/${orderId}/`);
  return response.data;
}

export default function CheckoutSuccessPage() {
  const searchParams = useSearchParams();
  const orderId = searchParams.get("order_id");
  const orderNumber = searchParams.get("order_number");

  const orderQuery = useQuery({
    queryKey: ["orders", orderId],
    queryFn: () => fetchOrder(orderId as string),
    enabled: Boolean(orderId),
  });

  return (
    <AuthGate nextHref="/checkout">
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-4xl px-4 sm:px-6 py-16">
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
                  <p className="text-sm font-semibold">Order #{orderNumber}</p>
                ) : null}
                <p className="text-xs text-foreground/60">
                  If you need help, please contact support with your order number.
                </p>
              </div>
            ) : orderQuery.data ? (
              <div className="space-y-3 text-sm">
                <p className="text-sm text-foreground/60">
                  Order #{orderQuery.data.order_number}
                </p>
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
                <p className="text-sm text-foreground/60">Order #{orderNumber}</p>
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
              <Button asChild>
                <Link href="/orders/">View orders</Link>
              </Button>
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
