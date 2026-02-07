import { cn } from "@/lib/utils";

type LoadingScreenProps = {
  title?: string;
  subtitle?: string;
  fullScreen?: boolean;
  className?: string;
};

export function LoadingScreen({
  title = "Bunoraa",
  subtitle = "Curating your next discovery.",
  fullScreen = false,
  className,
}: LoadingScreenProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-busy="true"
      className={cn(
        "relative flex w-full items-center justify-center overflow-hidden bg-background text-foreground",
        fullScreen ? "min-h-screen" : "min-h-[60vh]",
        className
      )}
    >
      <div className="pointer-events-none absolute -top-32 left-1/2 h-64 w-64 -translate-x-1/2 rounded-full bg-gradient-to-br from-primary/20 via-accent/10 to-transparent blur-3xl" />
      <div className="pointer-events-none absolute bottom-0 right-0 h-72 w-72 translate-x-1/3 rounded-full bg-gradient-to-tr from-primary/10 via-accent/10 to-transparent blur-3xl" />
      <div className="relative z-10 flex flex-col items-center gap-4 text-center">
        <span className="text-xs font-semibold uppercase tracking-[0.4em] text-foreground/50">
          Loading
        </span>
        <h1 className="font-display text-3xl font-semibold tracking-tight sm:text-4xl">
          {title}
        </h1>
        <p className="max-w-md text-sm text-foreground/70 sm:text-base">
          {subtitle}
        </p>
        <div className="mt-2 h-1 w-40 overflow-hidden rounded-full bg-muted">
          <div className="h-full w-1/2 animate-shimmer bg-gradient-to-r from-transparent via-primary/40 to-transparent motion-reduce:animate-none" />
        </div>
      </div>
    </div>
  );
}
