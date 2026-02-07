import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { UserPreferences } from "@/lib/types";

const preferencesKey = ["account", "preferences"] as const;

async function fetchPreferences() {
  const response = await apiFetch<UserPreferences>("/accounts/preferences/");
  return response.data;
}

export function usePreferences() {
  const queryClient = useQueryClient();

  const preferencesQuery = useQuery({
    queryKey: preferencesKey,
    queryFn: fetchPreferences,
  });

  const updatePreferences = useMutation({
    mutationFn: async (payload: Partial<UserPreferences>) => {
      const response = await apiFetch<UserPreferences>("/accounts/preferences/", {
        method: "PATCH",
        body: payload,
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: preferencesKey });
    },
  });

  return {
    preferencesQuery,
    updatePreferences,
  };
}
