import * as React from "react";
import { cn } from "@/lib/utils";

type CardVariant = "default" | "bordered" | "glass" | "modern-gradient";

const variantClasses: Record<CardVariant, string> = {
  default: "bg-card text-foreground shadow-soft",
  bordered: "bg-card text-foreground border border-border",
  glass:
    "bg-card/70 text-foreground border border-border/60 backdrop-blur-xl shadow-soft",
  "modern-gradient":
    "bg-gradient-to-br from-[hsl(var(--primary)/0.08)] via-[hsl(var(--accent)/0.08)] to-[hsl(var(--primary)/0.18)] text-foreground shadow-soft",
};

export type CardProps = React.HTMLAttributes<HTMLDivElement> & {
  variant?: CardVariant;
};

export function Card({ className, variant = "default", ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-2xl p-5 transition-shadow",
        variantClasses[variant],
        className
      )}
      {...props}
    />
  );
}
