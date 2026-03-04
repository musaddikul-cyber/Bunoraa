import * as React from "react";
import { Star as StarIcon } from "lucide-react";
import { cn } from "@/lib/utils";

function RatingStar({ filled }: { filled: boolean }) {
  return (
    <StarIcon
      aria-hidden="true"
      className={cn("h-4 w-4", filled ? "fill-accent-500 text-accent-500" : "text-border")}
      strokeWidth={1.8}
    />
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
          <RatingStar key={index} filled={index < rounded} />
        ))}
      </div>
      {showCount && typeof count === "number" ? (
        <span className="text-xs text-foreground/60">({count})</span>
      ) : null}
    </div>
  );
}
