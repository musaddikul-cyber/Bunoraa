"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/account/dashboard/", label: "Dashboard" },
  { href: "/account/profile/", label: "Profile" },
  { href: "/account/orders/", label: "Orders" },
  { href: "/account/addresses/", label: "Addresses" },
  { href: "/account/payments/", label: "Payments" },
  { href: "/account/subscriptions/", label: "Subscriptions" },
  { href: "/account/notifications/", label: "Notifications" },
  { href: "/account/preferences/", label: "Preferences" },
  { href: "/account/security/", label: "Security" },
  { href: "/account/privacy/", label: "Privacy" },
];

export function AccountShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [mobileNavOpen, setMobileNavOpen] = React.useState(false);

  const activeItem =
    NAV_ITEMS.find((item) => pathname.startsWith(item.href)) || NAV_ITEMS[0];

  React.useEffect(() => {
    setMobileNavOpen(false);
  }, [pathname]);

  React.useEffect(() => {
    if (!mobileNavOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileNavOpen(false);
    };
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [mobileNavOpen]);

  return (
    <AuthGate
      title="Account access"
      description="Sign in to manage your account."
      nextHref={pathname}
    >
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-12">
          <div className="mb-4 lg:hidden">
            <Button
              type="button"
              variant="secondary"
              size="sm"
              className="w-full justify-between rounded-xl px-4 text-sm"
              onClick={() => setMobileNavOpen(true)}
            >
              <span>Account menu</span>
              <span className="truncate text-foreground/65">{activeItem?.label}</span>
            </Button>
          </div>
          <div className="flex flex-col gap-6 lg:grid lg:grid-cols-[240px_1fr]">
            <aside className="hidden space-y-4 lg:block">
              <Card variant="bordered" className="space-y-3">
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                  Account
                </p>
                <nav className="flex flex-col gap-1">
                  {NAV_ITEMS.map((item) => {
                    const active = pathname.startsWith(item.href);
                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        className={cn(
                          "rounded-lg px-3 py-2 text-sm transition",
                          active
                            ? "bg-muted font-semibold text-foreground"
                            : "text-foreground/70 hover:bg-muted"
                        )}
                      >
                        {item.label}
                      </Link>
                    );
                  })}
                </nav>
              </Card>
              <Card variant="modern-gradient" className="space-y-2">
                <p className="text-sm font-semibold">Need help?</p>
                <p className="text-sm text-foreground/70">
                  Reach out to our support team for any account changes or data
                  requests.
                </p>
                <Button asChild size="sm" variant="secondary">
                  <Link href="/contact/">Contact support</Link>
                </Button>
              </Card>
            </aside>
            <main className="space-y-6">{children}</main>
          </div>
        </div>
      </div>

      {mobileNavOpen ? (
        <div
          className="fixed inset-0 z-[90] lg:hidden"
          role="dialog"
          aria-modal="true"
          aria-label="Account navigation"
          onClick={(event) => {
            if (event.target === event.currentTarget) setMobileNavOpen(false);
          }}
        >
          <button
            type="button"
            className="absolute inset-0 bg-black/55"
            aria-label="Close account navigation"
            onClick={() => setMobileNavOpen(false)}
          />
          <aside className="absolute right-0 top-0 flex h-full w-full max-w-xs flex-col border-l border-border bg-background p-4 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/55">
                  Account
                </p>
                <p className="text-sm font-semibold">{activeItem?.label}</p>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="rounded-lg px-3"
                onClick={() => setMobileNavOpen(false)}
              >
                Close
              </Button>
            </div>
            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
              <Card variant="bordered" className="space-y-2 p-3">
                <nav className="flex flex-col gap-1">
                  {NAV_ITEMS.map((item) => {
                    const active = pathname.startsWith(item.href);
                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        className={cn(
                          "rounded-lg px-3 py-2 text-sm transition",
                          active
                            ? "bg-muted font-semibold text-foreground"
                            : "text-foreground/70 hover:bg-muted"
                        )}
                      >
                        {item.label}
                      </Link>
                    );
                  })}
                </nav>
              </Card>
              <Card variant="modern-gradient" className="space-y-2 p-3">
                <p className="text-sm font-semibold">Need help?</p>
                <p className="text-sm text-foreground/70">
                  Reach out to support for account updates or data requests.
                </p>
                <Button asChild size="sm" variant="secondary" className="w-full">
                  <Link href="/contact/">Contact support</Link>
                </Button>
              </Card>
            </div>
          </aside>
        </div>
      ) : null}
    </AuthGate>
  );
}
