import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { NotificationPreference } from "@/lib/types";

const prefKey = ["notifications", "preferences"] as const;

async function fetchPreferences() {
  const response = await apiFetch<NotificationPreference>(
    "/notifications/preferences/"
  );
  return response.data;
}

export function useNotificationPreferences() {
  const queryClient = useQueryClient();

  const preferencesQuery = useQuery({
    queryKey: prefKey,
    queryFn: fetchPreferences,
  });

  const updatePreferences = useMutation({
    mutationFn: async (payload: Partial<NotificationPreference>) => {
      const response = await apiFetch<NotificationPreference>(
        "/notifications/preferences/",
        { method: "PUT", body: payload }
      );
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: prefKey });
    },
  });

  return {
    preferencesQuery,
    updatePreferences,
  };
}
