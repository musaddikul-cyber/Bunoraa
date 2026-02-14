"use client";

import * as React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useAuth } from "@/components/auth/useAuth";

export function AuthGate({
  children,
  title = "Authentication required",
  description = "Please sign in to continue.",
  nextHref,
}: {
  children: React.ReactNode;
  title?: string;
  description?: string;
  nextHref?: string;
}) {
  const { hasToken } = useAuth();
  const [mounted, setMounted] = React.useState(false);
  const loginHref = nextHref
    ? `/account/login/?next=${encodeURIComponent(nextHref)}`
    : "/account/login/";

  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-3xl px-4 sm:px-6 py-20">
          <Card variant="bordered" className="space-y-3">
            <div className="h-5 w-40 animate-pulse rounded bg-muted" />
            <div className="h-4 w-64 animate-pulse rounded bg-muted" />
            <div className="h-10 w-32 animate-pulse rounded bg-muted" />
          </Card>
        </div>
      </div>
    );
  }

  if (!hasToken) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-3xl px-4 sm:px-6 py-20">
          <Card variant="bordered" className="space-y-4">
            <h1 className="text-2xl font-semibold">{title}</h1>
            <p className="text-sm text-foreground/70">{description}</p>
            <Button asChild variant="primary-gradient">
              <Link href={loginHref}>Sign in</Link>
            </Button>
          </Card>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
