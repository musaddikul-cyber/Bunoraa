"use client";

import Link from "next/link";
import { useNotifications } from "@/components/notifications/useNotifications";

export function NotificationBell() {
  const { unreadCountQuery } = useNotifications();
  const count = unreadCountQuery.data?.count || 0;

  return (
    <Link
      href="/notifications/"
      className="relative inline-flex items-center p-2 text-sm"
      aria-label="Notifications"
    >
      <svg
        aria-hidden="true"
        viewBox="0 0 24 24"
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M15 17h5l-1.4-1.4A3 3 0 0 1 18 14v-4.5a6 6 0 1 0-12 0V14a3 3 0 0 1-.6 1.6L4 17h5" />
        <path d="M9 17a3 3 0 0 0 6 0" />
      </svg>
      <span className="sr-only">Notifications</span>
      {count > 0 ? (
        <span className="absolute -right-2 -top-2 rounded-full bg-accent px-2 py-0.5 text-xs text-white">
          {count}
        </span>
      ) : null}
    </Link>
  );
}
