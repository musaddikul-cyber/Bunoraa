import * as React from "react";
import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "primary-gradient";
type ButtonSize = "sm" | "md" | "lg";

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-primary text-white shadow-soft hover:bg-primary-700 hover:shadow-soft-lg",
  secondary:
    "border border-border bg-card text-foreground hover:bg-muted",
  ghost: "bg-transparent text-foreground hover:bg-muted",
  "primary-gradient":
    "bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 text-white shadow-soft hover:opacity-95",
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: "h-8 px-4 text-sm",
  md: "h-10 px-6 text-sm",
  lg: "h-11 px-8 text-base",
};

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
  asChild?: boolean;
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", asChild, children, ...props }, ref) => {
    const classes = cn(
      "inline-flex items-center justify-center gap-2 rounded-xl font-semibold transition",
      "disabled:opacity-50 disabled:cursor-not-allowed",
      variantClasses[variant],
      sizeClasses[size],
      className
    );

    if (asChild && React.isValidElement(children)) {
      const child = children as React.ReactElement<React.HTMLAttributes<HTMLElement>>;
      return React.cloneElement(child, {
        className: cn(classes, child.props.className),
        ...props,
      });
    }

    return (
      <button ref={ref} className={classes} {...props}>
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";
