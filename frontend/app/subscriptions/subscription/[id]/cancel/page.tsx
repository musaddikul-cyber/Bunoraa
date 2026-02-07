"use client";

import { useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { apiFetch } from "@/lib/api";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";

export default function SubscriptionCancelPage() {
  const router = useRouter();
  const params = useParams();
  const id = params?.id as string;

  useEffect(() => {
    if (!id) return;
    apiFetch(`/subscriptions/subscriptions/${id}/cancel/`, { method: "POST" })
      .finally(() => router.push(`/subscriptions/subscription/${id}/`));
  }, [id, router]);

  return (
    <AuthGate>
      <div className="mx-auto w-full max-w-md px-6 py-12">
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          Cancelling subscription...
        </Card>
      </div>
    </AuthGate>
  );
}
