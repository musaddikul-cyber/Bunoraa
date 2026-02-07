import * as React from "react";
import { cn } from "@/lib/utils";

export type SwatchOption = {
  label: string;
  value: string;
  color?: string | null;
};

export function ProductSwatches({
  options,
  selected,
  onSelect,
  className,
}: {
  options: SwatchOption[];
  selected?: string | null;
  onSelect?: (value: string) => void;
  className?: string;
}) {
  if (!options.length) return null;

  return (
    <div className={cn("flex flex-wrap gap-2", className)}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onSelect?.(option.value)}
          className={cn(
            "flex items-center gap-2 rounded-full border px-3 py-1 text-xs",
            selected === option.value
              ? "border-primary text-primary"
              : "border-border text-foreground/70"
          )}
        >
          {option.color ? (
            <span
              className="h-3 w-3 rounded-full border border-border"
              style={{ backgroundColor: option.color }}
            />
          ) : null}
          {option.label}
        </button>
      ))}
    </div>
  );
}