"use client";

import Link from "next/link";
import { Bell } from "lucide-react";
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
        "relative inline-flex h-11 w-11 items-center justify-center rounded-full border border-border/80 bg-card/90 text-sm leading-none text-foreground shadow-soft transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
        className
      )}
      aria-label="Notifications"
    >
      <Bell aria-hidden="true" className="h-5 w-5" strokeWidth={1.8} />
      <span className="sr-only">Notifications</span>
      {count > 0 ? (
        <span className="absolute -right-2 -top-2 rounded-full bg-accent px-2 py-0.5 text-xs text-white">
          {count}
        </span>
      ) : null}
    </Link>
  );
}
