"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { apiFetch } from "@/lib/api";
import { setTokens } from "@/lib/auth";

export default function OAuthCallbackPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [error, setError] = React.useState<string | null>(null);

  const nextUrl = searchParams.get("next") || "/account/profile/";

  React.useEffect(() => {
    let cancelled = false;

    const finalize = async () => {
      try {
        const response = await apiFetch<{ access: string; refresh: string }>(
          "/accounts/social/token/",
          { method: "GET", skipAuth: true }
        );
        if (cancelled) return;
        setTokens(response.data.access, response.data.refresh, true);
        router.replace(nextUrl);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Google sign-in failed.");
      }
    };

    finalize();
    return () => {
      cancelled = true;
    };
  }, [nextUrl, router]);

  if (!error) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-md px-3 sm:px-5 py-20">
        <Card variant="bordered" className="space-y-4 text-center">
          <h1 className="text-xl font-semibold">Google sign-in failed</h1>
          <p className="text-sm text-foreground/70">{error}</p>
          <Button asChild variant="secondary" className="w-full">
            <Link href="/account/login/">Back to sign in</Link>
          </Button>
        </Card>
      </div>
    </div>
  );
}
