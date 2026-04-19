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
import { buildGoogleOAuthUrl } from "@/lib/oauth";

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
  const [showPassword, setShowPassword] = React.useState(false);
  const [passwordValue, setPasswordValue] = React.useState("");
  const [showForgotDialog, setShowForgotDialog] = React.useState(false);
  const [forgotSent, setForgotSent] = React.useState(false);
  const forgotForm = useForm<{ email: string }>({
    resolver: zodResolver(z.object({ email: z.string().email("Enter a valid email") })),
  });
  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { remember: true },
  });

  const nextUrl = searchParams.get("next") || "/account/profile/";
  const callbackPath = `/account/oauth/callback/?next=${encodeURIComponent(nextUrl)}`;
  const googleOAuthUrl = buildGoogleOAuthUrl(callbackPath);

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
      <div className="mx-auto w-full max-w-md px-3 sm:px-5 py-20">
        <Card variant="bordered" className="space-y-6">
          <div>
            <h1 className="text-2xl font-semibold">Sign in</h1>
          </div>

          <Button asChild variant="secondary" className="w-full">
            <a href={googleOAuthUrl} className="inline-flex items-center justify-center gap-2">
              <svg className="h-5 w-5" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Continue with Google
            </a>
          </Button>

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
                <div className="relative mt-2">
                  <input
                    className={`w-full rounded-lg border border-border bg-card px-3 py-2 pr-12 transition-all ${showPassword ? "text-base tracking-normal font-normal" : "text-xl tracking-widest font-bold [-webkit-text-stroke:1px_currentColor]"}`}
                    type={showPassword ? "text" : "password"}
                    {...form.register("password", { onChange: (e) => setPasswordValue(e.target.value) })}
                  />
                  {passwordValue && (
                    <button
                      type="button"
                      onClick={() => setShowPassword((v) => !v)}
                      className="absolute right-0 top-1/2 -translate-y-1/2 text-foreground/60 hover:text-foreground p-2"
                      aria-label={showPassword ? "Hide password" : "Show password"}
                    >
                      {!showPassword ? (
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7c.78 0 1.53-.09 2.24-.26"/><path d="M2 2l20 20"/></svg>
                      ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
                      )}
                    </button>
                  )}
                </div>
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
            <button
              type="button"
              onClick={() => setShowForgotDialog(true)}
              className="text-primary hover:underline"
            >
              Forgot password?
            </button>
            <Link className="text-primary" href="/account/register/">
              Create account
            </Link>
          </div>
        </Card>

        {/* Forgot Password Dialog */}
        {showForgotDialog && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
            onClick={() => setShowForgotDialog(false)}
          >
            <Card
              variant="bordered"
              className="w-full max-w-md space-y-6"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">Reset password</h2>
                <button
                  type="button"
                  onClick={() => setShowForgotDialog(false)}
                  className="text-foreground/60 hover:text-foreground"
                  aria-label="Close dialog"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
                </button>
              </div>

              {forgotSent ? (
                <p className="text-sm text-foreground/70">
                  If an account exists with that email, a reset link has been sent.
                </p>
              ) : (
                <form
                  className="space-y-4"
                  onSubmit={forgotForm.handleSubmit(async (values) => {
                    await apiFetch("/accounts/password/reset/request/", {
                      method: "POST",
                      body: values,
                    });
                    setForgotSent(true);
                  })}
                >
                  <label className="block text-sm">
                    Email
                    <input
                      className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                      type="email"
                      {...forgotForm.register("email")}
                    />
                  </label>
                  <Button type="submit" className="w-full">
                    Send reset link
                  </Button>
                </form>
              )}
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
