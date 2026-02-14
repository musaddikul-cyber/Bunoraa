"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { SearchBar } from "@/components/search/SearchBar";
import type { MenuPage } from "@/lib/types";
import { cn } from "@/lib/utils";

type Category = { id: string; name: string; slug: string };

export function MobileNav({
  categories,
  menuPages,
}: {
  categories: Category[];
  menuPages: MenuPage[];
}) {
  const pathname = usePathname();
  const [open, setOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement | null>(null);
  const closeButtonRef = React.useRef<HTMLButtonElement | null>(null);
  const wasOpenRef = React.useRef(false);

  const normalizePath = React.useCallback((value: string) => {
    if (value.length > 1 && value.endsWith("/")) {
      return value.slice(0, -1);
    }
    return value;
  }, []);

  const isActiveLink = React.useCallback(
    (href: string) => {
      const current = normalizePath(pathname || "/");
      const target = normalizePath(href);
      if (target === "/") return current === "/";
      return current === target || current.startsWith(`${target}/`);
    },
    [pathname, normalizePath]
  );

  const navLinkClass = React.useCallback(
    (href: string) =>
      cn(
        "block rounded-xl border px-3 py-2.5 text-sm transition-colors",
        "border-transparent text-foreground/90 hover:border-border hover:bg-muted hover:text-foreground",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40",
        isActiveLink(href) && "border-primary/25 bg-primary/10 text-primary"
      ),
    [isActiveLink]
  );

  const closeNav = React.useCallback(() => {
    setOpen(false);
  }, []);

  React.useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeNav();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
    };
  }, [open, closeNav]);

  React.useEffect(() => {
    if (!open) return;
    const originalOverflow = document.body.style.overflow;
    const originalTouchAction = document.body.style.touchAction;
    document.body.style.overflow = "hidden";
    document.body.style.touchAction = "none";
    return () => {
      document.body.style.overflow = originalOverflow;
      document.body.style.touchAction = originalTouchAction;
    };
  }, [open]);

  React.useEffect(() => {
    if (!open) return;
    closeButtonRef.current?.focus();
  }, [open]);

  React.useEffect(() => {
    if (open) {
      wasOpenRef.current = true;
      return;
    }
    if (!wasOpenRef.current) return;
    triggerRef.current?.focus();
    wasOpenRef.current = false;
  }, [open]);

  React.useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <div className="lg:hidden">
      <button
        ref={triggerRef}
        type="button"
        className="inline-flex items-center justify-center rounded-full border border-border bg-card p-2 text-sm text-foreground shadow-soft transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls="mobile-navigation-panel"
        aria-label="Open menu"
      >
        <span className="flex flex-col gap-1" aria-hidden="true">
          <span className="h-0.5 w-5 rounded-full bg-foreground/90" />
          <span className="h-0.5 w-5 rounded-full bg-foreground/90" />
          <span className="h-0.5 w-5 rounded-full bg-foreground/90" />
        </span>
      </button>

      {open ? (
        <div
          className="fixed inset-0 z-50 h-[100svh] supports-[height:100dvh]:h-[100dvh]"
          role="dialog"
          aria-modal="true"
          aria-labelledby="mobile-navigation-title"
          onClick={(event) => {
            if (event.target === event.currentTarget) closeNav();
          }}
        >
          <div className="absolute inset-0 bg-foreground/35 backdrop-blur-[1px]" onClick={closeNav} />
          <aside
            id="mobile-navigation-panel"
            className="absolute inset-y-0 left-0 flex h-[100svh] min-h-[100svh] w-full max-w-[22rem] flex-col border-r border-border bg-background px-5 pb-[max(1.25rem,env(safe-area-inset-bottom))] pt-[max(1.25rem,env(safe-area-inset-top))] text-foreground shadow-2xl supports-[height:100dvh]:h-[100dvh] supports-[height:100dvh]:min-h-[100dvh]"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <p id="mobile-navigation-title" className="text-lg font-semibold">
                Menu
              </p>
              <button
                ref={closeButtonRef}
                type="button"
                className="rounded-lg border border-border px-2.5 py-1.5 text-sm text-foreground transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                onClick={closeNav}
              >
                Close
              </button>
            </div>

            <div className="mt-4 shrink-0">
              <SearchBar />
            </div>

            <nav className="mt-6 min-h-0 flex-1 space-y-6 overflow-y-auto pb-4 pr-2 text-sm scrollbar-thin">
              <div className="space-y-2">
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/70">
                  Shop
                </p>
                <Link
                  className={navLinkClass("/products/")}
                  href="/products/"
                  onClick={closeNav}
                >
                  Products
                </Link>
                <Link
                  className={navLinkClass("/collections/")}
                  href="/collections/"
                  onClick={closeNav}
                >
                  Collections
                </Link>
                <Link
                  className={navLinkClass("/bundles/")}
                  href="/bundles/"
                  onClick={closeNav}
                >
                  Bundles
                </Link>
                <Link
                  className={cn(
                    "block rounded-xl border px-3 py-2.5 text-sm font-semibold transition-colors",
                    "border-primary/30 bg-primary/10 text-primary hover:bg-primary/15",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40",
                    isActiveLink("/preorders/") && "border-primary/50 bg-primary/20"
                  )}
                  href="/preorders/"
                  onClick={closeNav}
                >
                  Preorders
                </Link>
                <Link
                  className={navLinkClass("/categories/")}
                  href="/categories/"
                  onClick={closeNav}
                >
                  Categories
                </Link>
                {categories.slice(0, 10).map((category) => (
                  <Link
                    key={category.id}
                    className={navLinkClass(`/categories/${category.slug}/`)}
                    href={`/categories/${category.slug}/`}
                    onClick={closeNav}
                  >
                    {category.name}
                  </Link>
                ))}
                {categories.length === 0 ? (
                  <p className="rounded-xl border border-dashed border-border px-3 py-2.5 text-foreground/70">
                    No categories available.
                  </p>
                ) : null}
              </div>

              <div className="space-y-2">
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/70">
                  Account
                </p>
                <Link
                  className={navLinkClass("/account/login/")}
                  href="/account/login/"
                  onClick={closeNav}
                >
                  Sign in
                </Link>
                <Link
                  className={navLinkClass("/orders/")}
                  href="/orders/"
                  onClick={closeNav}
                >
                  Orders
                </Link>
                <Link
                  className={navLinkClass("/wishlist/")}
                  href="/wishlist/"
                  onClick={closeNav}
                >
                  Wishlist
                </Link>
                <Link
                  className={navLinkClass("/cart/")}
                  href="/cart/"
                  onClick={closeNav}
                >
                  Cart
                </Link>
              </div>

              {menuPages.length ? (
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/70">
                    Pages
                  </p>
                  {menuPages.slice(0, 8).map((page) => (
                    <Link
                      key={page.id}
                      className={navLinkClass(`/pages/${page.slug}/`)}
                      href={`/pages/${page.slug}/`}
                      onClick={closeNav}
                    >
                      {page.title}
                    </Link>
                  ))}
                </div>
              ) : null}

              <div className="space-y-2 pb-2">
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/70">
                  Support
                </p>
                <Link
                  className={navLinkClass("/contact/")}
                  href="/contact/"
                  onClick={closeNav}
                >
                  Contact
                </Link>
                <Link
                  className={navLinkClass("/faq/")}
                  href="/faq/"
                  onClick={closeNav}
                >
                  FAQ
                </Link>
              </div>
            </nav>
          </aside>
        </div>
      ) : null}
    </div>
  );
}
