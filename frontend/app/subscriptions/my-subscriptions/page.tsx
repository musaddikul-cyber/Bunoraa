"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { Subscription } from "@/lib/types";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";

export default function MySubscriptionsPage() {
  const subscriptionsQuery = useQuery({
    queryKey: ["subscriptions", "list"],
    queryFn: async () => {
      const response = await apiFetch<Subscription[]>("/subscriptions/");
      return response.data;
    },
  });

  return (
    <AuthGate title="Subscriptions" description="Sign in to manage subscriptions.">
      <div className="mx-auto w-full max-w-4xl px-4 sm:px-6 py-12">
        <div className="mb-6">
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Subscriptions
          </p>
          <h1 className="text-3xl font-semibold">My subscriptions</h1>
        </div>
        {subscriptionsQuery.isLoading ? (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Loading subscriptions...
          </Card>
        ) : subscriptionsQuery.data?.length ? (
          <div className="space-y-4">
            {subscriptionsQuery.data.map((sub) => (
              <Card key={sub.id} variant="bordered" className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-foreground/60">{sub.plan?.name}</p>
                    <p className="text-base font-semibold">{sub.status}</p>
                  </div>
                  <Link className="text-primary" href={`/subscriptions/subscription/${sub.id}/`}>
                    Manage
                  </Link>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            You have no active subscriptions.
          </Card>
        )}
      </div>
    </AuthGate>
  );
}
