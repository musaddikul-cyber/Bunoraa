 "use client";

import * as React from "react";
import { cn } from "@/lib/utils";

type LoadingScreenProps = {
  title?: string;
  subtitle?: string;
  fullScreen?: boolean;
  className?: string;
};

export function LoadingScreen({
  title,
  subtitle,
  fullScreen = false,
  className,
}: LoadingScreenProps) {
  const [progress, setProgress] = React.useState(6);
  const [dots, setDots] = React.useState("..");

  React.useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");

    if (media.matches) {
      const reducedTimer = window.setInterval(() => {
        setProgress((prev) => (prev >= 94 ? prev : Math.min(94, prev + 6)));
      }, 900);
      return () => window.clearInterval(reducedTimer);
    }

    const start = window.performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const elapsed = now - start;
      const next = 6 + (94 - 6) * (1 - Math.exp(-elapsed / 2200));
      setProgress(Math.min(94, next));
      frame = window.requestAnimationFrame(tick);
    };

    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, []);

  React.useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (media.matches) {
      setDots("..");
      return;
    }

    const frames = [".", "..", "..."];
    let index = 1;
    const timer = window.setInterval(() => {
      index = (index + 1) % frames.length;
      setDots(frames[index]);
    }, 500);

    return () => window.clearInterval(timer);
  }, []);

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
          Loading{dots}
        </span>
        {title ? (
          <h1 className="font-display text-3xl font-semibold tracking-tight sm:text-4xl">
            {title}
          </h1>
        ) : null}
        {subtitle ? (
          <p className="max-w-md text-sm text-foreground/70 sm:text-base">
            {subtitle}
          </p>
        ) : null}
        <div className="mt-2 w-44 space-y-1.5">
          <div
            role="progressbar"
            aria-label="Loading progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(progress)}
            className="h-1.5 overflow-hidden rounded-full bg-muted"
          >
            <div
              className="h-full rounded-full bg-gradient-to-r from-primary via-accent to-primary transition-[width] duration-300 ease-out motion-reduce:transition-none"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-[11px] font-medium tracking-[0.08em] text-foreground/55">
            {Math.round(progress)}%
          </p>
        </div>
      </div>
    </div>
  );
}
