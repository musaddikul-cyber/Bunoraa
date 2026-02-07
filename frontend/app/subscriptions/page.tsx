import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { SubscriptionPlan } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export const revalidate = 600;

async function getPlans() {
  const response = await apiFetch<SubscriptionPlan[]>("/subscriptions/plans/", {
    next: { revalidate },
  });
  return response.data;
}

export default async function SubscriptionsLandingPage() {
  const plans = await getPlans();

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-12">
      <div className="mb-8">
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Subscriptions
        </p>
        <h1 className="text-3xl font-semibold">Choose a plan</h1>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        {plans.map((plan) => (
          <Card key={plan.id} variant="bordered" className="space-y-4 p-6">
            <h2 className="text-xl font-semibold">{plan.name}</h2>
            <p className="text-sm text-foreground/70">{plan.description}</p>
            <p className="text-lg font-semibold">
              {plan.price_amount} {plan.currency} / {plan.interval}
            </p>
            <Button asChild variant="primary-gradient">
              <Link href={`/subscriptions/plans/${plan.id}/`}>View plan</Link>
            </Button>
          </Card>
        ))}
      </div>
    </div>
  );
}
