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
    <div className="fixed inset-0 z-50" aria-hidden={!isOpen}>
      <div
        className={cn(
          "absolute inset-0 bg-black/40 transition-opacity",
          isOpen ? "opacity-100" : "opacity-0"
        )}
        onClick={onClose}
      />
      <aside
        className={cn(
          "absolute right-0 top-0 h-full w-full max-w-sm transform bg-background p-6 shadow-xl transition-transform",
          isOpen ? "translate-x-0" : "translate-x-full"
        )}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Your cart</h2>
          <button className="text-sm text-foreground/60" onClick={onClose}>
            Close
          </button>
        </div>
        <MiniCart />
      </aside>
    </div>
  );
}
