"use client";

import * as React from "react";
import Link from "next/link";
import { SearchBar } from "@/components/search/SearchBar";
import type { MenuPage } from "@/lib/types";

type Category = { id: string; name: string; slug: string };

export function MobileNav({
  categories,
  menuPages,
}: {
  categories: Category[];
  menuPages: MenuPage[];
}) {
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => {
    if (!open) return;
    const original = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = original;
    };
  }, [open]);

  return (
    <div className="lg:hidden">
      <button
        type="button"
        className="inline-flex items-center rounded-full p-2 text-sm"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label="Open menu"
      >
        <span className="flex flex-col gap-1" aria-hidden="true">
          <span className="h-0.5 w-5 rounded-full bg-foreground" />
          <span className="h-0.5 w-5 rounded-full bg-foreground" />
          <span className="h-0.5 w-5 rounded-full bg-foreground" />
        </span>
      </button>

      {open ? (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black" onClick={() => setOpen(false)} />
          <aside className="absolute left-0 top-0 h-full w-full max-w-xs border-r border-border bg-[hsl(var(--background))] p-5 shadow-2xl">
            <div className="flex items-center justify-between">
              <p className="text-lg font-semibold">Browse</p>
              <button
                type="button"
                className="text-sm text-foreground/60"
                onClick={() => setOpen(false)}
              >
                Close
              </button>
            </div>

            <div className="mt-4">
              <SearchBar />
            </div>

            <nav className="mt-6 space-y-4 text-sm">
              <div className="space-y-2">
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/50">
                  Shop
                </p>
                <Link
                  className="block rounded-lg px-2 py-2 text-foreground/80 hover:bg-muted"
                  href="/products/"
                  onClick={() => setOpen(false)}
                >
                  Products
                </Link>
                <Link
                  className="block rounded-lg border border-primary/20 bg-primary/10 px-2 py-2 font-semibold text-primary hover:bg-primary/20"
                  href="/preorders/"
                  onClick={() => setOpen(false)}
                >
                  Preorders
                </Link>
                <Link
                  className="block rounded-lg px-2 py-2 text-foreground/80 hover:bg-muted"
                  href="/categories/"
                  onClick={() => setOpen(false)}
                >
                  Categories
                </Link>
                {categories.slice(0, 8).map((category) => (
                  <Link
                    key={category.id}
                    className="block rounded-lg px-2 py-2 text-foreground/80 hover:bg-muted"
                    href={`/categories/${category.slug}/`}
                    onClick={() => setOpen(false)}
                  >
                    {category.name}
                  </Link>
                ))}
              </div>

              {menuPages.length ? (
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/50">
                    Pages
                  </p>
                  {menuPages.slice(0, 6).map((page) => (
                    <Link
                      key={page.id}
                      className="block rounded-lg px-2 py-2 text-foreground/80 hover:bg-muted"
                      href={`/pages/${page.slug}/`}
                      onClick={() => setOpen(false)}
                    >
                      {page.title}
                    </Link>
                  ))}
                </div>
              ) : null}
            </nav>
          </aside>
        </div>
      ) : null}
    </div>
  );
}
