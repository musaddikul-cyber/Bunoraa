import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch, ApiError } from "@/lib/api";
import { useAuthContext } from "@/components/providers/AuthProvider";
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
  const { hasToken } = useAuthContext();

  const preferencesQuery = useQuery({
    queryKey: prefKey,
    queryFn: fetchPreferences,
    enabled: hasToken,
    retry: (count, error) => {
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
        return false;
      }
      return count < 2;
    },
  });

  const updatePreferences = useMutation({
    mutationFn: async (payload: Partial<NotificationPreference>) => {
      const response = await apiFetch<NotificationPreference>(
        "/notifications/preferences/",
        { method: "PUT", body: payload }
      );
      return response.data;
    },
    onSuccess: (data) => {
      queryClient.setQueryData(prefKey, data);
    },
  });

  return {
    preferencesQuery,
    updatePreferences,
  };
}
