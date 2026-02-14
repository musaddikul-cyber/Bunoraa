"use client";

import * as React from "react";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

type MobileHeaderVisibilityProps = {
  children: React.ReactNode;
  className?: string;
};

export function MobileHeaderVisibility({
  children,
  className,
}: MobileHeaderVisibilityProps) {
  const pathname = usePathname();
  const [hidden, setHidden] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const lastScrollYRef = React.useRef(0);
  const tickingRef = React.useRef(false);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    setHidden(false);
    lastScrollYRef.current = window.scrollY;
  }, [pathname]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;

    const root = document.documentElement;
    const mobileQuery = window.matchMedia("(max-width: 639px)");
    const updateHeaderOffset = () => {
      if (!mobileQuery.matches) {
        root.style.setProperty("--mobile-header-offset", "var(--header-offset)");
        return;
      }

      if (hidden) {
        root.style.setProperty("--mobile-header-offset", "0px");
        return;
      }

      const height = containerRef.current?.offsetHeight ?? 0;
      root.style.setProperty("--mobile-header-offset", `${height}px`);
    };

    const raf = window.requestAnimationFrame(updateHeaderOffset);
    window.addEventListener("resize", updateHeaderOffset);
    mobileQuery.addEventListener("change", updateHeaderOffset);

    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener("resize", updateHeaderOffset);
      mobileQuery.removeEventListener("change", updateHeaderOffset);
      root.style.setProperty("--mobile-header-offset", "var(--header-offset)");
    };
  }, [hidden, pathname]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;

    const mobileQuery = window.matchMedia("(max-width: 639px)");

    const resetState = () => {
      if (!mobileQuery.matches) {
        setHidden(false);
      }
      lastScrollYRef.current = window.scrollY;
    };

    const updateVisibility = () => {
      tickingRef.current = false;

      if (!mobileQuery.matches) {
        setHidden(false);
        lastScrollYRef.current = window.scrollY;
        return;
      }

      const currentScrollY = window.scrollY;
      const delta = currentScrollY - lastScrollYRef.current;

      if (Math.abs(delta) < 6) return;

      if (currentScrollY <= 16 || delta < 0) {
        setHidden(false);
      } else if (currentScrollY > 80 && delta > 0) {
        setHidden(true);
      }

      lastScrollYRef.current = currentScrollY;
    };

    const onScroll = () => {
      if (tickingRef.current) return;
      tickingRef.current = true;
      window.requestAnimationFrame(updateVisibility);
    };

    resetState();
    window.addEventListener("scroll", onScroll, { passive: true });
    mobileQuery.addEventListener("change", resetState);

    return () => {
      window.removeEventListener("scroll", onScroll);
      mobileQuery.removeEventListener("change", resetState);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className={cn(
        "sticky top-0 z-40 transform transition-transform duration-300 ease-out",
        hidden ? "-translate-y-full sm:translate-y-0" : "translate-y-0",
        className
      )}
    >
      {children}
    </div>
  );
}
