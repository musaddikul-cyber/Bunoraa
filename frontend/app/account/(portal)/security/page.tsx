"use client";

import * as React from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { apiFetch } from "@/lib/api";
import { useMfa } from "@/components/account/useMfa";
import { useSessions } from "@/components/account/useSessions";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import {
  decodeCreationOptions,
  encodeCredential,
} from "../../../../lib/webauthn";

const passwordSchema = z
  .object({
    current_password: z.string().min(1, "Current password is required"),
    new_password: z.string().min(8, "Password must be at least 8 characters"),
    new_password_confirm: z
      .string()
      .min(8, "Confirm your new password"),
  })
  .refine((data) => data.new_password === data.new_password_confirm, {
    message: "Passwords do not match",
    path: ["new_password_confirm"],
  });

type PasswordForm = z.infer<typeof passwordSchema>;

export default function SecurityPage() {
  const form = useForm<PasswordForm>({
    resolver: zodResolver(passwordSchema),
  });
  const {
    statusQuery,
    setupTotp,
    verifyTotp,
    disableTotp,
    regenerateBackupCodes,
    registerPasskeyOptions,
    registerPasskeyVerify,
    passkeysQuery,
    removePasskey,
  } = useMfa();
  const { sessionsQuery, revokeSession, revokeOthers } = useSessions();

  const [totpSecret, setTotpSecret] = React.useState<string | null>(null);
  const [totpUri, setTotpUri] = React.useState<string | null>(null);
  const [totpCode, setTotpCode] = React.useState("");
  const [backupCodes, setBackupCodes] = React.useState<string[]>([]);
  const [passkeyName, setPasskeyName] = React.useState("");

  const handlePasswordChange = async (values: PasswordForm) => {
    await apiFetch("/accounts/password/change/", {
      method: "POST",
      body: values,
    });
    form.reset();
  };

  const handleTotpSetup = async () => {
    const response = await setupTotp.mutateAsync();
    setTotpSecret(response.secret);
    setTotpUri(response.otpauth_url);
  };

  const handleTotpVerify = async () => {
    const response = await verifyTotp.mutateAsync(totpCode);
    setBackupCodes(response.backup_codes || []);
    setTotpCode("");
  };

  const handleTotpDisable = async () => {
    await disableTotp.mutateAsync(totpCode);
    setTotpCode("");
  };

  const handleBackupCodes = async () => {
    const response = await regenerateBackupCodes.mutateAsync();
    setBackupCodes(response.backup_codes || []);
  };

  const handleRegisterPasskey = async () => {
    if (typeof window === "undefined" || !window.PublicKeyCredential) return;
    const options = await registerPasskeyOptions.mutateAsync();
    const decoded = decodeCreationOptions(
      options as unknown as PublicKeyCredentialCreationOptions
    );
    const credential = (await navigator.credentials.create({
      publicKey: decoded,
    })) as PublicKeyCredential | null;
    if (!credential) return;
    await registerPasskeyVerify.mutateAsync({
      credential: encodeCredential(credential),
      nickname: passkeyName,
    });
    setPasskeyName("");
  };

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Account
        </p>
        <h1 className="text-3xl font-semibold">Security</h1>
        <p className="mt-2 text-sm text-foreground/70">
          Manage password, MFA, passkeys, and active sessions.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card variant="bordered" className="space-y-4">
          <h2 className="text-lg font-semibold">Change password</h2>
          <p className="text-sm text-foreground/70">
            Use a long passphrase and avoid reusing passwords across services.
          </p>
          <form className="space-y-3" onSubmit={form.handleSubmit(handlePasswordChange)}>
            <label className="block text-sm">
              Current password
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                type="password"
                {...form.register("current_password")}
              />
            </label>
            <label className="block text-sm">
              New password
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                type="password"
                {...form.register("new_password")}
              />
            </label>
            <label className="block text-sm">
              Confirm new password
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                type="password"
                {...form.register("new_password_confirm")}
              />
            </label>
            <Button type="submit">Update password</Button>
          </form>
        </Card>

        <Card variant="bordered" className="space-y-4">
          <h2 className="text-lg font-semibold">Multi-factor authentication</h2>
          <p className="text-sm text-foreground/70">
            Add an authenticator app or passkey to protect your account.
          </p>
          <div className="rounded-lg border border-border bg-muted p-3 text-sm">
            Status: {statusQuery.data?.enabled ? "Enabled" : "Not enabled"}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" onClick={handleTotpSetup}>
              Set up authenticator
            </Button>
            <Button variant="secondary" onClick={handleBackupCodes}>
              Generate backup codes
            </Button>
          </div>
          {totpSecret ? (
            <div className="space-y-2 rounded-lg border border-border bg-muted p-3 text-sm">
              <p>Secret: {totpSecret}</p>
              {totpUri ? (
                <a className="text-primary" href={totpUri}>
                  Add to authenticator
                </a>
              ) : null}
            </div>
          ) : null}
          <label className="block text-sm">
            Verification code
            <input
              className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
              value={totpCode}
              onChange={(event) => setTotpCode(event.target.value)}
            />
          </label>
          <div className="flex flex-wrap gap-2">
            <Button onClick={handleTotpVerify} disabled={verifyTotp.isPending}>
              Enable MFA
            </Button>
            <Button variant="ghost" onClick={handleTotpDisable}>
              Disable MFA
            </Button>
          </div>
          {backupCodes.length ? (
            <div className="rounded-lg border border-border bg-muted p-3 text-sm">
              <p className="font-semibold">Backup codes</p>
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                {backupCodes.map((code) => (
                  <span key={code} className="rounded bg-card px-2 py-1">
                    {code}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
        </Card>
      </div>

      <Card variant="bordered" className="space-y-4">
        <h2 className="text-lg font-semibold">Passkeys</h2>
        <p className="text-sm text-foreground/70">
          Register a passkey to sign in quickly without a password.
        </p>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <input
            className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
            placeholder="Passkey nickname (optional)"
            value={passkeyName}
            onChange={(event) => setPasskeyName(event.target.value)}
          />
          <Button onClick={handleRegisterPasskey}>
            Add passkey
          </Button>
        </div>
        {passkeysQuery.data?.length ? (
          <div className="space-y-2">
            {passkeysQuery.data.map((passkey) => (
              <div
                key={passkey.id}
                className="flex flex-col gap-2 rounded-lg border border-border p-3 sm:flex-row sm:items-center sm:justify-between"
              >
                <div>
                  <p className="font-semibold">
                    {passkey.nickname || "Passkey"}
                  </p>
                  <p className="text-xs text-foreground/60">
                    Added {passkey.created_at}
                  </p>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => removePasskey.mutate(passkey.id)}
                >
                  Remove
                </Button>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-foreground/60">No passkeys registered.</p>
        )}
      </Card>

      <Card variant="bordered" className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold">Active sessions</h2>
            <p className="text-sm text-foreground/70">
              Manage devices currently signed in.
            </p>
          </div>
          <Button variant="secondary" onClick={() => revokeOthers.mutate()}>
            Sign out other devices
          </Button>
        </div>
        {sessionsQuery.isLoading ? (
          <p className="text-sm text-foreground/60">Loading sessions...</p>
        ) : sessionsQuery.data?.length ? (
          <div className="space-y-2">
            {sessionsQuery.data.map((session) => (
              <div
                key={session.id}
                className="flex flex-col gap-2 rounded-lg border border-border p-3 sm:flex-row sm:items-center sm:justify-between"
              >
                <div>
                  <p className="font-semibold">
                    {session.device_type || "Device"} · {session.browser || "Browser"}
                  </p>
                  <p className="text-xs text-foreground/60">
                    {session.ip_address || "Unknown IP"} · Last activity {session.last_activity}
                  </p>
                  {session.is_current ? (
                    <span className="text-xs uppercase tracking-[0.2em] text-primary">
                      Current session
                    </span>
                  ) : null}
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => revokeSession.mutate(session.id)}
                  disabled={session.is_current}
                >
                  Revoke
                </Button>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-foreground/60">No active sessions.</p>
        )}
      </Card>
    </div>
  );
}
