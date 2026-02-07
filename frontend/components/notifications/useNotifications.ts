import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { getAccessToken } from "@/lib/auth";
import type { NotificationItem } from "@/lib/types";

const listKey = ["notifications"] as const;
const unreadKey = ["notifications", "unread"] as const;

async function fetchNotifications() {
  const response = await apiFetch<NotificationItem[]>("/notifications/");
  return response.data;
}

async function fetchUnreadCount() {
  const response = await apiFetch<{ count: number }>("/notifications/unread_count/");
  return response.data;
}

export function useNotifications() {
  const queryClient = useQueryClient();
  const hasToken = Boolean(getAccessToken());

  const notificationsQuery = useQuery({
    queryKey: listKey,
    queryFn: fetchNotifications,
    enabled: hasToken,
  });

  const unreadCountQuery = useQuery({
    queryKey: unreadKey,
    queryFn: fetchUnreadCount,
    enabled: hasToken,
  });

  const markAllRead = useMutation({
    mutationFn: async () => {
      return apiFetch("/notifications/mark_all_read/", { method: "POST" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: listKey });
      queryClient.invalidateQueries({ queryKey: unreadKey });
    },
  });

  const markRead = useMutation({
    mutationFn: async (ids: string[]) => {
      return apiFetch("/notifications/mark_read/", {
        method: "POST",
        body: { notification_ids: ids },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: listKey });
      queryClient.invalidateQueries({ queryKey: unreadKey });
    },
  });

  return {
    notificationsQuery,
    unreadCountQuery,
    markAllRead,
    markRead,
  };
}
