"use client";

import Link from "next/link";
import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useAuth } from "@/components/auth/useAuth";
import { apiFetch } from "@/lib/api";
import { setTokens } from "@/lib/auth";
import { decodeRequestOptions, encodeCredential } from "../../../../lib/webauthn";

const schema = z.object({
  email: z.string().email("Enter a valid email"),
  password: z.string().min(1, "Password is required"),
  remember: z.boolean().optional(),
});

type FormValues = z.infer<typeof schema>;

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login, verifyMfa } = useAuth();
  const [mfaToken, setMfaToken] = React.useState<string | null>(null);
  const [mfaMethods, setMfaMethods] = React.useState<string[]>([]);
  const [mfaMethod, setMfaMethod] = React.useState<string>("totp");
  const [mfaCode, setMfaCode] = React.useState("");
  const [mfaError, setMfaError] = React.useState<string | null>(null);
  const [passkeyPending, setPasskeyPending] = React.useState(false);
  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { remember: true },
  });

  const nextUrl = searchParams.get("next") || "/account/profile/";

  const onSubmit = async (values: FormValues) => {
    const result = await login.mutateAsync({
      email: values.email,
      password: values.password,
      remember: Boolean(values.remember),
    });
    if (result?.mfa_required) {
      setMfaToken(result.mfa_token || null);
      setMfaMethods(result.methods || []);
      setMfaMethod(result.methods?.[0] || "totp");
      return;
    }
    router.push(nextUrl);
  };

  const handleVerifyMfa = async () => {
    if (!mfaToken) return;
    setMfaError(null);
    try {
      await verifyMfa.mutateAsync({
        mfa_token: mfaToken,
        method: mfaMethod as "totp" | "backup_code" | "passkey",
        code: mfaMethod === "passkey" ? undefined : mfaCode,
        remember: Boolean(form.getValues("remember")),
      });
      router.push(nextUrl);
    } catch (err) {
      setMfaError(err instanceof Error ? err.message : "Verification failed.");
    }
  };

  const handlePasskeyLogin = async () => {
    const email = form.getValues("email");
    if (!email) {
      setMfaError("Enter your email to use passkey sign-in.");
      return;
    }
    if (typeof window === "undefined" || !window.PublicKeyCredential || !navigator.credentials) {
      setMfaError("Passkeys are not supported on this device or browser.");
      return;
    }
    setPasskeyPending(true);
    setMfaError(null);
    try {
      const optionsResponse = await apiFetch<PublicKeyCredentialRequestOptions>(
        "/accounts/webauthn/login/options/",
        { method: "POST", body: { email } }
      );
      const options = decodeRequestOptions(optionsResponse.data);
      const credential = (await navigator.credentials.get({
        publicKey: options,
      })) as PublicKeyCredential | null;
      if (!credential) throw new Error("Passkey sign-in was cancelled.");
      const verifyResponse = await apiFetch<{
        access: string;
        refresh: string;
      }>("/accounts/webauthn/login/verify/", {
        method: "POST",
        body: { email, credential: encodeCredential(credential) },
      });
      setTokens(
        verifyResponse.data.access,
        verifyResponse.data.refresh,
        Boolean(form.getValues("remember"))
      );
      router.push(nextUrl);
    } catch (err) {
      setMfaError(err instanceof Error ? err.message : "Passkey sign-in failed.");
    } finally {
      setPasskeyPending(false);
    }
  };

  const handlePasskeyMfa = async () => {
    if (!mfaToken) return;
    if (typeof window === "undefined" || !window.PublicKeyCredential || !navigator.credentials) {
      setMfaError("Passkeys are not supported on this device or browser.");
      return;
    }
    setPasskeyPending(true);
    setMfaError(null);
    try {
      const optionsResponse = await apiFetch<PublicKeyCredentialRequestOptions>(
        "/accounts/webauthn/login/options/",
        { method: "POST", body: { mfa_token: mfaToken } }
      );
      const options = decodeRequestOptions(optionsResponse.data);
      const credential = (await navigator.credentials.get({
        publicKey: options,
      })) as PublicKeyCredential | null;
      if (!credential) throw new Error("Passkey verification was cancelled.");
      await verifyMfa.mutateAsync({
        mfa_token: mfaToken,
        method: "passkey",
        credential: encodeCredential(credential),
        remember: Boolean(form.getValues("remember")),
      });
      router.push(nextUrl);
    } catch (err) {
      setMfaError(err instanceof Error ? err.message : "Passkey verification failed.");
    } finally {
      setPasskeyPending(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-md px-6 py-20">
        <Card variant="bordered" className="space-y-6">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Account
            </p>
            <h1 className="text-2xl font-semibold">Sign in</h1>
          </div>

          {!mfaToken ? (
            <form className="space-y-4" onSubmit={form.handleSubmit(onSubmit)}>
              <label className="block text-sm">
                Email
                <input
                  className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                  type="email"
                  {...form.register("email")}
                />
              </label>

              <label className="block text-sm">
                Password
                <input
                  className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                  type="password"
                  {...form.register("password")}
                />
              </label>

              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" {...form.register("remember")} />
                Remember me
              </label>

              {login.isError ? (
                <p className="text-sm text-red-500">
                  {login.error instanceof Error ? login.error.message : "Login failed."}
                </p>
              ) : null}

              {mfaError ? (
                <p className="text-sm text-red-500">{mfaError}</p>
              ) : null}

              <Button type="submit" className="w-full" disabled={login.isPending}>
                {login.isPending ? "Signing in..." : "Sign in"}
              </Button>
              <Button
                type="button"
                variant="secondary"
                className="w-full"
                onClick={handlePasskeyLogin}
                disabled={passkeyPending}
              >
                {passkeyPending ? "Waiting for passkey..." : "Use passkey instead"}
              </Button>
            </form>
          ) : (
            <div className="space-y-4">
              <p className="text-sm text-foreground/70">
                Multi-factor authentication is enabled for this account.
              </p>
              <div className="flex flex-wrap gap-2">
                {mfaMethods.includes("totp") ? (
                  <Button
                    type="button"
                    variant={mfaMethod === "totp" ? "primary" : "secondary"}
                    size="sm"
                    onClick={() => setMfaMethod("totp")}
                  >
                    Authenticator
                  </Button>
                ) : null}
                {mfaMethods.includes("backup_code") ? (
                  <Button
                    type="button"
                    variant={mfaMethod === "backup_code" ? "primary" : "secondary"}
                    size="sm"
                    onClick={() => setMfaMethod("backup_code")}
                  >
                    Backup code
                  </Button>
                ) : null}
                {mfaMethods.includes("passkey") ? (
                  <Button
                    type="button"
                    variant={mfaMethod === "passkey" ? "primary" : "secondary"}
                    size="sm"
                    onClick={() => setMfaMethod("passkey")}
                  >
                    Passkey
                  </Button>
                ) : null}
              </div>

              {mfaMethod === "passkey" ? (
                <Button
                  type="button"
                  className="w-full"
                  onClick={handlePasskeyMfa}
                  disabled={passkeyPending}
                >
                  {passkeyPending ? "Waiting for passkey..." : "Verify with passkey"}
                </Button>
              ) : (
                <>
                  <label className="block text-sm">
                    {mfaMethod === "backup_code" ? "Backup code" : "Verification code"}
                    <input
                      className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                      value={mfaCode}
                      onChange={(event) => setMfaCode(event.target.value)}
                    />
                  </label>
                  {mfaError ? (
                    <p className="text-sm text-red-500">{mfaError}</p>
                  ) : null}
                  <Button
                    type="button"
                    className="w-full"
                    onClick={handleVerifyMfa}
                    disabled={verifyMfa.isPending}
                  >
                    {verifyMfa.isPending ? "Verifying..." : "Verify and continue"}
                  </Button>
                </>
              )}
            </div>
          )}

          <div className="flex items-center justify-between text-sm text-foreground/70">
            <Link className="text-primary" href="/account/forgot-password/">
              Forgot password?
            </Link>
            <Link className="text-primary" href="/account/register/">
              Create account
            </Link>
          </div>
        </Card>
      </div>
    </div>
  );
}
