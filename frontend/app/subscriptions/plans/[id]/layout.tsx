import type { Metadata } from "next";
import { apiFetch } from "@/lib/api";
import type { SubscriptionPlan } from "@/lib/types";
import { buildPageMetadata } from "@/lib/seo";

export const revalidate = 600;

async function tryGetPlan(id: string) {
  try {
    const response = await apiFetch<SubscriptionPlan>(`/subscriptions/plans/${id}/`, {
      next: { revalidate },
    });
    return response.data;
  } catch {
    return null;
  }
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>;
}): Promise<Metadata> {
  const { id } = await params;
  const plan = await tryGetPlan(id);
  return buildPageMetadata({
    title: plan?.name ? `${plan.name} Subscription Plan` : "Subscription Plan",
    description:
      plan?.description || "Review Bunoraa subscription plan details and pricing.",
    path: `/subscriptions/plans/${id}/`,
  });
}

export default function SubscriptionPlanDetailLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
