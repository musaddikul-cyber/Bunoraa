import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { Subscription } from "@/lib/types";

const subsKey = ["subscriptions"] as const;

async function fetchSubscriptions() {
  const response = await apiFetch<Subscription[]>("/subscriptions/");
  return response.data;
}

export function useSubscriptions() {
  const queryClient = useQueryClient();

  const subscriptionsQuery = useQuery({
    queryKey: subsKey,
    queryFn: fetchSubscriptions,
  });

  const cancelSubscription = useMutation({
    mutationFn: async (id: string) => {
      return apiFetch(`/subscriptions/${id}/cancel/`, { method: "POST" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: subsKey });
    },
  });

  const resumeSubscription = useMutation({
    mutationFn: async (id: string) => {
      return apiFetch(`/subscriptions/${id}/resume/`, { method: "POST" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: subsKey });
    },
  });

  const changePlan = useMutation({
    mutationFn: async ({ id, plan_id }: { id: string; plan_id: string }) => {
      return apiFetch(`/subscriptions/${id}/change_plan/`, {
        method: "POST",
        body: { plan_id },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: subsKey });
    },
  });

  return {
    subscriptionsQuery,
    cancelSubscription,
    resumeSubscription,
    changePlan,
  };
}
