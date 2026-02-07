"use client";

import { useParams } from "next/navigation";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { Subscription } from "@/lib/types";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import Link from "next/link";

export default function SubscriptionDetailPage() {
  const params = useParams();
  const id = params?.id as string;

  const subscriptionQuery = useQuery({
    queryKey: ["subscriptions", id],
    queryFn: async () => {
      const response = await apiFetch<Subscription>(`/subscriptions/subscriptions/${id}/`);
      return response.data;
    },
    enabled: Boolean(id),
  });

  const cancel = useMutation({
    mutationFn: async () => {
      return apiFetch(`/subscriptions/subscriptions/${id}/cancel/`, { method: "POST" });
    },
  });

  const resume = useMutation({
    mutationFn: async () => {
      return apiFetch(`/subscriptions/subscriptions/${id}/resume/`, { method: "POST" });
    },
  });

  return (
    <AuthGate title="Subscription" description="Sign in to manage your subscription.">
      <div className="mx-auto w-full max-w-4xl px-6 py-12">
        {subscriptionQuery.isLoading ? (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Loading subscription...
          </Card>
        ) : subscriptionQuery.data ? (
          <Card variant="bordered" className="space-y-4 p-6">
            <h1 className="text-2xl font-semibold">{subscriptionQuery.data.plan?.name}</h1>
            <p className="text-sm text-foreground/70">Status: {subscriptionQuery.data.status}</p>
            <div className="flex flex-wrap gap-2">
              <Button variant="secondary" onClick={() => resume.mutate()}>
                Reactivate
              </Button>
              <Button variant="ghost" onClick={() => cancel.mutate()}>
                Cancel
              </Button>
              <Button asChild variant="secondary">
                <Link href={`/subscriptions/subscription/${id}/change-plan/`}>Change plan</Link>
              </Button>
            </div>
          </Card>
        ) : (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Subscription not found.
          </Card>
        )}
      </div>
    </AuthGate>
  );
}
