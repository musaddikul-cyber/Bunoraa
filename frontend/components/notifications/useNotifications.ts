import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { useAuthContext } from "@/components/providers/AuthProvider";
import type { NotificationItem } from "@/lib/types";

const listKey = ["notifications"] as const;
const unreadKey = ["notifications", "unread"] as const;

export type NotificationFilters = {
  unread?: boolean;
  category?: string;
  type?: string;
  priority?: string;
  status?: string;
};

async function fetchNotifications(filters?: NotificationFilters) {
  const response = await apiFetch<NotificationItem[]>("/notifications/", {
    params: filters,
  });
  return response.data;
}

async function fetchUnreadCount() {
  const response = await apiFetch<{ count: number }>("/notifications/unread_count/");
  return response.data;
}

export function useNotifications(filters?: NotificationFilters) {
  const queryClient = useQueryClient();
  const { hasToken } = useAuthContext();

  const notificationsQuery = useQuery({
    queryKey: [...listKey, filters],
    queryFn: () => fetchNotifications(filters),
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
