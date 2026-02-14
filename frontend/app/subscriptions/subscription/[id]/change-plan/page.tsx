"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { SubscriptionPlan } from "@/lib/types";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export default function ChangePlanPage() {
  const params = useParams();
  const router = useRouter();
  const id = params?.id as string;
  const [planId, setPlanId] = useState<string | null>(null);

  const plansQuery = useQuery({
    queryKey: ["subscriptions", "plans"],
    queryFn: async () => {
      const response = await apiFetch<SubscriptionPlan[]>("/subscriptions/plans/");
      return response.data;
    },
  });

  const changePlan = useMutation({
    mutationFn: async () => {
      if (!planId) return null;
      return apiFetch(`/subscriptions/${id}/change_plan/`, {
        method: "POST",
        body: { plan_id: planId, proration_behavior: "none" },
      });
    },
    onSuccess: () => router.push(`/subscriptions/subscription/${id}/`),
  });

  return (
    <AuthGate title="Change plan" description="Sign in to update your subscription.">
      <div className="mx-auto w-full max-w-4xl px-4 sm:px-6 py-12">
        <Card variant="bordered" className="space-y-4 p-6">
          <h1 className="text-2xl font-semibold">Select a new plan</h1>
          {plansQuery.isLoading ? (
            <p className="text-sm text-foreground/70">Loading plans...</p>
          ) : (
            <div className="space-y-2">
              {plansQuery.data?.map((plan) => (
                <label key={plan.id} className="flex items-center gap-3 text-sm">
                  <input
                    type="radio"
                    name="plan"
                    value={plan.id}
                    checked={planId === plan.id}
                    onChange={() => setPlanId(plan.id)}
                  />
                  {plan.name} ({plan.price_amount} {plan.currency})
                </label>
              ))}
            </div>
          )}
          <Button
            onClick={() => changePlan.mutate()}
            disabled={!planId || changePlan.isPending}
          >
            Change plan
          </Button>
        </Card>
      </div>
    </AuthGate>
  );
}
