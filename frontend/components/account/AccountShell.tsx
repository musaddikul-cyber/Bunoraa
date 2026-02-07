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

  return (
    <AuthGate
      title="Account access"
      description="Sign in to manage your account."
      nextHref={pathname}
    >
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-7xl px-6 py-12">
          <div className="flex flex-col gap-6 lg:grid lg:grid-cols-[240px_1fr]">
            <aside className="space-y-4">
              <Card variant="bordered" className="space-y-3">
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                  Account
                </p>
                <nav className="hidden flex-col gap-1 lg:flex">
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
                <div className="flex gap-2 overflow-x-auto lg:hidden">
                  {NAV_ITEMS.map((item) => {
                    const active = pathname.startsWith(item.href);
                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        className={cn(
                          "whitespace-nowrap rounded-full border px-3 py-1 text-xs",
                          active
                            ? "border-primary text-primary"
                            : "border-border text-foreground/70"
                        )}
                      >
                        {item.label}
                      </Link>
                    );
                  })}
                </div>
              </Card>
              <Card variant="modern-gradient" className="space-y-2">
                <p className="text-sm font-semibold">Need help?</p>
                <p className="text-sm text-foreground/70">
                  Reach out to our support team for any account changes or data
                  requests.
                </p>
                <Button asChild size="sm" variant="secondary">
                  <Link href="/contacts/">Contact support</Link>
                </Button>
              </Card>
            </aside>
            <main className="space-y-6">{children}</main>
          </div>
        </div>
      </div>
    </AuthGate>
  );
}
