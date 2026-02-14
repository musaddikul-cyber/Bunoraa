"use client";

import * as React from "react";
import { useParams } from "next/navigation";
import Image from "next/image";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { OrderDetail } from "@/lib/types";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";
import { useOrders } from "@/components/orders/useOrders";
import { isUuid, resolveOrderId } from "@/lib/orders";
import { formatMoney } from "@/lib/checkout";
import { formatAddressLine } from "@/lib/address";

export default function OrderDetailPage() {
  const params = useParams();
  const rawId = params?.id as string;
  const ordersQuery = useOrders();

  const resolvedId = React.useMemo(() => {
    if (!rawId) return null;
    if (isUuid(rawId)) return rawId;
    const list = ordersQuery.data?.data ?? [];
    return resolveOrderId(rawId, list);
  }, [rawId, ordersQuery.data]);

  const orderQuery = useQuery({
    queryKey: ["orders", resolvedId],
    queryFn: async () => {
      const response = await apiFetch<OrderDetail>(`/orders/${resolvedId}/`);
      return response.data;
    },
    enabled: Boolean(resolvedId),
  });

  return (
    <AuthGate title="Order detail" description="Sign in to view order details.">
      <div className="mx-auto w-full max-w-4xl px-4 sm:px-6 py-12">
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
            Order not found in your account.
          </Card>
        ) : orderQuery.isLoading ? (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Loading order...
          </Card>
        ) : orderQuery.data ? (
          (() => {
            const order = orderQuery.data;
            const currency = order.currency || undefined;
            const formatAmount = (value: string | number | null | undefined) =>
              formatMoney(value, currency);
            const hasDiscount = Number(order.discount || 0) > 0;
            const hasGiftWrap = Boolean(order.gift_wrap) && Number(order.gift_wrap_cost || 0) > 0;
            const hasPaymentFee = Number(order.payment_fee_amount || 0) > 0;
            const createdAt = order.created_at ? new Date(order.created_at) : null;
            const shippingName = [
              order.shipping_address?.first_name,
              order.shipping_address?.last_name,
            ]
              .filter(Boolean)
              .join(" ");
            const billingName = [
              order.billing_address?.first_name,
              order.billing_address?.last_name,
            ]
              .filter(Boolean)
              .join(" ");
            return (
              <div className="space-y-6">
                <div className="space-y-2">
                  <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
                    Order
                  </p>
                  <h1 className="text-3xl font-semibold">
                    #{order.order_number}
                  </h1>
                  <div className="flex flex-wrap gap-3 text-sm text-foreground/70">
                    <span>
                      Status: {order.status_display || order.status}
                    </span>
                    {order.payment_status ? (
                      <span>Payment: {order.payment_status}</span>
                    ) : null}
                    {createdAt ? (
                      <span>Placed {createdAt.toLocaleString()}</span>
                    ) : null}
                  </div>
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  <Card variant="bordered" className="p-4 space-y-2">
                    <h2 className="text-lg font-semibold">Shipping</h2>
                    <p className="text-sm text-foreground/70">
                      {order.shipping_method_display || order.shipping_method || "Shipping"}
                    </p>
                    <p className="text-sm font-semibold">
                      {shippingName || "Recipient"}
                    </p>
                    <p className="text-sm text-foreground/70">
                      {formatAddressLine(order.shipping_address)}
                    </p>
                    {order.pickup_location ? (
                      <div className="text-xs text-foreground/60">
                        Store pickup: {order.pickup_location.name}
                      </div>
                    ) : null}
                    {order.tracking_number ? (
                      <p className="text-xs text-foreground/60">
                        Tracking:{" "}
                        {order.tracking_url ? (
                          <a
                            className="underline"
                            href={order.tracking_url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {order.tracking_number}
                          </a>
                        ) : (
                          order.tracking_number
                        )}
                      </p>
                    ) : null}
                  </Card>

                  <Card variant="bordered" className="p-4 space-y-2">
                    <h2 className="text-lg font-semibold">Billing</h2>
                    <p className="text-sm text-foreground/70">
                      {order.payment_method_display || order.payment_method || "Payment"}
                    </p>
                    <p className="text-sm font-semibold">
                      {billingName || shippingName || "Billing contact"}
                    </p>
                    <p className="text-sm text-foreground/70">
                      {formatAddressLine(order.billing_address)}
                    </p>
                    {hasPaymentFee ? (
                      <p className="text-xs text-foreground/60">
                        {order.payment_fee_label
                          ? `${order.payment_fee_label}: `
                          : "Payment fee: "}
                        {formatAmount(order.payment_fee_amount)}
                      </p>
                    ) : null}
                  </Card>
                </div>

                <Card variant="bordered" className="p-4">
                  <h2 className="text-lg font-semibold">Items</h2>
                  <div className="mt-4 space-y-4">
                    {order.items?.map((item) => (
                      <div key={item.id} className="flex gap-4">
                        <div className="relative h-16 w-16 overflow-hidden rounded-xl bg-muted">
                          {item.product_image ? (
                            <Image
                              src={item.product_image}
                              alt={item.product_name}
                              fill
                              className="object-cover"
                            />
                          ) : null}
                        </div>
                        <div className="flex-1">
                          <p className="text-sm font-semibold">{item.product_name}</p>
                          {item.variant_name ? (
                            <p className="text-xs text-foreground/60">
                              {item.variant_name}
                            </p>
                          ) : null}
                          <p className="text-xs text-foreground/60">
                            Qty {item.quantity}
                          </p>
                        </div>
                        <div className="text-right text-sm">
                          {formatAmount(item.line_total || item.unit_price)}
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>

                <Card variant="bordered" className="p-4">
                  <h2 className="text-lg font-semibold">Summary</h2>
                  <div className="mt-3 space-y-2 text-sm text-foreground/70">
                    <div className="flex items-center justify-between">
                      <span>Subtotal</span>
                      <span>{formatAmount(order.subtotal)}</span>
                    </div>
                    {hasDiscount ? (
                      <div className="flex items-center justify-between">
                        <span>Discount</span>
                        <span>-{formatAmount(order.discount)}</span>
                      </div>
                    ) : null}
                    <div className="flex items-center justify-between">
                      <span>Shipping</span>
                      <span>{formatAmount(order.shipping_cost)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Tax</span>
                      <span>{formatAmount(order.tax)}</span>
                    </div>
                    {hasGiftWrap ? (
                      <div className="flex items-center justify-between">
                        <span>Gift wrap</span>
                        <span>{formatAmount(order.gift_wrap_cost)}</span>
                      </div>
                    ) : null}
                    {hasPaymentFee ? (
                      <div className="flex items-center justify-between">
                        <span>
                          {order.payment_fee_label || "Payment fee"}
                        </span>
                        <span>{formatAmount(order.payment_fee_amount)}</span>
                      </div>
                    ) : null}
                    <div className="flex items-center justify-between text-base font-semibold text-foreground">
                      <span>Total</span>
                      <span>{formatAmount(order.total)}</span>
                    </div>
                  </div>
                </Card>
              </div>
            );
          })()
        ) : (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Order not found.
          </Card>
        )}
      </div>
    </AuthGate>
  );
}
