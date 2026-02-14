"use client";

import Link from "next/link";
import { cn } from "@/lib/utils";

export function NotificationBell({
  className,
  count = 0,
}: {
  className?: string;
  count?: number;
}) {
  return (
    <Link
      href="/notifications/"
      className={cn(
        "relative inline-flex h-11 w-11 items-center justify-center rounded-full border border-border/80 bg-card/90 text-sm leading-none text-foreground shadow-soft transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:h-10 sm:w-10",
        className
      )}
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
