"use client";

import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { SubscriptionPlan } from "@/lib/types";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export default function SubscriptionPlanPage() {
  const params = useParams();
  const router = useRouter();
  const id = params?.id as string;

  const planQuery = useQuery({
    queryKey: ["subscriptions", "plan", id],
    queryFn: async () => {
      const response = await apiFetch<SubscriptionPlan>(`/subscriptions/plans/${id}/`);
      return response.data;
    },
    enabled: Boolean(id),
  });

  const subscribe = useMutation({
    mutationFn: async () => {
      return apiFetch("/subscriptions/subscriptions/", {
        method: "POST",
        body: { plan_id: id, quantity: 1 },
      });
    },
    onSuccess: () => router.push("/subscriptions/my-subscriptions/"),
  });

  return (
    <AuthGate title="Subscribe" description="Sign in to subscribe to a plan.">
      <div className="mx-auto w-full max-w-4xl px-6 py-12">
        {planQuery.isLoading ? (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Loading plan...
          </Card>
        ) : planQuery.data ? (
          <Card variant="bordered" className="space-y-4 p-6">
            <h1 className="text-2xl font-semibold">{planQuery.data.name}</h1>
            <p className="text-sm text-foreground/70">{planQuery.data.description}</p>
            <p className="text-lg font-semibold">
              {planQuery.data.price_amount} {planQuery.data.currency} / {planQuery.data.interval}
            </p>
            <Button onClick={() => subscribe.mutate()} disabled={subscribe.isPending}>
              {subscribe.isPending ? "Subscribing..." : "Subscribe"}
            </Button>
          </Card>
        ) : (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Plan not found.
          </Card>
        )}
      </div>
    </AuthGate>
  );
}
