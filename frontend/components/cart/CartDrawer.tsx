"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { MiniCart } from "@/components/cart/MiniCart";

export function CartDrawer({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const originalOverflow = React.useRef<string | null>(null);

  React.useEffect(() => {
    if (!isOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen, onClose]);

  React.useEffect(() => {
    if (!isOpen) return;
    const original = document.body.style.overflow;
    originalOverflow.current = original;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = originalOverflow.current || "";
      originalOverflow.current = null;
    };
  }, [isOpen]);

  if (!isOpen) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50"
      aria-hidden={!isOpen}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className={cn(
          "absolute inset-0 bg-black/40 transition-opacity",
          isOpen ? "opacity-100" : "opacity-0"
        )}
        onClick={onClose}
      />
      <aside
        className={cn(
          "absolute right-0 top-0 flex h-[100dvh] w-full max-w-[420px] transform flex-col bg-background px-4 pb-[calc(1rem+env(safe-area-inset-bottom))] pt-[calc(0.75rem+env(safe-area-inset-top))] shadow-xl transition-transform sm:p-6",
          isOpen ? "translate-x-0" : "translate-x-full"
        )}
        onClick={(event) => event.stopPropagation()}
      >
        <MiniCart title="Your cart" onClose={onClose} className="h-full min-h-0" />
      </aside>
    </div>
  );
}
