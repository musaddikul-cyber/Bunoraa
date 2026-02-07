import * as React from "react";
import { cn } from "@/lib/utils";

function Star({ filled }: { filled: boolean }) {
  return (
    <svg
      viewBox="0 0 20 20"
      aria-hidden="true"
      className={cn("h-4 w-4", filled ? "text-accent-500" : "text-border")}
      fill="currentColor"
    >
      <path d="M10 1.6l2.5 5.1 5.6.8-4 3.9.9 5.6-5-2.7-5 2.7.9-5.6-4-3.9 5.6-.8L10 1.6z" />
    </svg>
  );
}

export function RatingStars({
  rating = 0,
  count,
  className,
  showCount = true,
}: {
  rating?: number | null;
  count?: number | null;
  className?: string;
  showCount?: boolean;
}) {
  const safeRating = Number.isFinite(rating as number) ? Number(rating) : 0;
  const rounded = Math.round(safeRating);

  return (
    <div className={cn("flex items-center gap-1 text-xs text-foreground/70", className)}>
      <div className="flex items-center gap-0.5">
        {Array.from({ length: 5 }).map((_, index) => (
          <Star key={index} filled={index < rounded} />
        ))}
      </div>
      {showCount && typeof count === "number" ? (
        <span className="text-xs text-foreground/60">({count})</span>
      ) : null}
    </div>
  );
}