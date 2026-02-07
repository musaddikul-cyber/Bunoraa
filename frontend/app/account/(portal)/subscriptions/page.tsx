"use client";

import { useSubscriptions } from "@/components/subscriptions/useSubscriptions";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export default function SubscriptionsPage() {
  const { subscriptionsQuery, cancelSubscription, resumeSubscription } =
    useSubscriptions();

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Account
        </p>
        <h1 className="text-3xl font-semibold">Subscriptions</h1>
        <p className="mt-2 text-sm text-foreground/70">
          Manage recurring plans and billing cycles.
        </p>
      </div>

      {subscriptionsQuery.isLoading ? (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          Loading subscriptions...
        </Card>
      ) : subscriptionsQuery.data?.length ? (
        <div className="space-y-4">
          {subscriptionsQuery.data.map((subscription) => (
            <Card key={subscription.id} variant="bordered" className="space-y-4">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm text-foreground/60">
                    {subscription.plan?.name || "Plan"}
                  </p>
                  <p className="text-lg font-semibold capitalize">
                    {subscription.status || "active"}
                  </p>
                  {subscription.next_billing_at ? (
                    <p className="text-xs text-foreground/60">
                      Next billing: {subscription.next_billing_at}
                    </p>
                  ) : null}
                </div>
                <div className="flex gap-2">
                  {subscription.status === "active" ? (
                    <Button
                      variant="secondary"
                      onClick={() => cancelSubscription.mutate(subscription.id)}
                    >
                      Cancel
                    </Button>
                  ) : (
                    <Button
                      variant="secondary"
                      onClick={() => resumeSubscription.mutate(subscription.id)}
                    >
                      Resume
                    </Button>
                  )}
                </div>
              </div>
              <div className="rounded-xl bg-muted p-3 text-sm text-foreground/70">
                {subscription.plan?.description || "Manage this plan in your billing settings."}
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          No active subscriptions.
        </Card>
      )}
    </div>
  );
}
