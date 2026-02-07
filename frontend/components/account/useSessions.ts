import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { UserSession } from "@/lib/types";

const sessionKey = ["account", "sessions"] as const;

async function fetchSessions() {
  const response = await apiFetch<UserSession[]>("/accounts/sessions/");
  return response.data;
}

export function useSessions() {
  const queryClient = useQueryClient();

  const sessionsQuery = useQuery({
    queryKey: sessionKey,
    queryFn: fetchSessions,
  });

  const revokeSession = useMutation({
    mutationFn: async (sessionId: string) => {
      return apiFetch(`/accounts/sessions/${sessionId}/revoke/`, { method: "POST" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: sessionKey });
    },
  });

  const revokeOthers = useMutation({
    mutationFn: async () => {
      return apiFetch("/accounts/sessions/revoke_others/", { method: "POST" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: sessionKey });
    },
  });

  return {
    sessionsQuery,
    revokeSession,
    revokeOthers,
  };
}
