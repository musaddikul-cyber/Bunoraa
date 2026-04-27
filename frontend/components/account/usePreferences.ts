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
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    refetchOnWindowFocus: false,
  });

  const updatePreferences = useMutation({
    mutationFn: async (payload: Partial<UserPreferences>) => {
      const response = await apiFetch<UserPreferences>("/accounts/preferences/", {
        method: "PATCH",
        body: payload,
      });
      return response.data;
    },
    onSuccess: (data) => {
      // Immediately update cache with server response
      queryClient.setQueryData(preferencesKey, data);
    },
  });

  return {
    preferencesQuery,
    updatePreferences,
  };
}
