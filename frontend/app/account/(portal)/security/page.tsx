"use client";

import * as React from "react";
import Image from "next/image";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { apiFetch } from "@/lib/api";
import { useMfa } from "@/components/account/useMfa";
import { useSessions } from "@/components/account/useSessions";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import qrcodeGenerator from "qrcode-generator";
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

function buildBackupCodesText(codes: string[]) {
  const issuedOn = new Date().toISOString();
  return [
    "Bunoraa Backup Codes",
    `Issued: ${issuedOn}`,
    "",
    "Each code can be used once. Keep these codes in a safe place.",
    "",
    ...codes,
    "",
  ].join("\n");
}

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

  const [totpFlow, setTotpFlow] = React.useState<"idle" | "setup" | "disable">(
    "idle"
  );
  const [totpSecret, setTotpSecret] = React.useState<string | null>(null);
  const [totpUri, setTotpUri] = React.useState<string | null>(null);
  const [totpQrDataUrl, setTotpQrDataUrl] = React.useState<string | null>(null);
  const [totpSetupCode, setTotpSetupCode] = React.useState("");
  const [totpDisableCode, setTotpDisableCode] = React.useState("");
  const [backupCodes, setBackupCodes] = React.useState<string[]>([]);
  const [passkeyName, setPasskeyName] = React.useState("");
  const [mfaNotice, setMfaNotice] = React.useState<string | null>(null);

  const mfaEnabled = Boolean(statusQuery.data?.enabled);
  const passkeyPending =
    registerPasskeyOptions.isPending || registerPasskeyVerify.isPending;

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
    setTotpSetupCode("");
    setTotpFlow("setup");
    setMfaNotice(null);
  };

  const handleTotpVerify = async () => {
    const response = await verifyTotp.mutateAsync(totpSetupCode);
    setBackupCodes(response.backup_codes || []);
    setTotpSetupCode("");
    setTotpSecret(null);
    setTotpUri(null);
    setTotpFlow("idle");
    setMfaNotice("Authenticator app enabled. Download and store your backup codes.");
  };

  const handleTotpDisable = async () => {
    await disableTotp.mutateAsync(totpDisableCode);
    setTotpDisableCode("");
    setTotpFlow("idle");
    setTotpSecret(null);
    setTotpUri(null);
    setMfaNotice("Multi-factor authentication disabled.");
  };

  const handleBackupCodes = async () => {
    if (!mfaEnabled) {
      setMfaNotice("Enable MFA first. Backup codes are only used with MFA.");
      return;
    }
    setMfaNotice("Generating backup codes. This can take a few seconds...");
    const response = await regenerateBackupCodes.mutateAsync();
    setBackupCodes(response.backup_codes || []);
    setMfaNotice("New backup codes generated. Download and store them now.");
  };

  const handleBackupCodesDownload = React.useCallback(() => {
    if (typeof window === "undefined" || backupCodes.length === 0) return;
    const blob = new Blob([buildBackupCodesText(backupCodes)], {
      type: "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `bunoraa-backup-codes-${new Date()
      .toISOString()
      .slice(0, 10)}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [backupCodes]);

  const copyMfaValue = React.useCallback(async (value: string, label: string) => {
    if (typeof navigator === "undefined" || !navigator.clipboard) {
      setMfaNotice(`Unable to copy ${label.toLowerCase()} on this device.`);
      return;
    }
    try {
      await navigator.clipboard.writeText(value);
      setMfaNotice(`${label} copied.`);
    } catch {
      setMfaNotice(`Unable to copy ${label.toLowerCase()} on this device.`);
    }
  }, []);

  React.useEffect(() => {
    if (!mfaEnabled) return;
    if (totpFlow !== "setup") return;
    setTotpFlow("idle");
    setTotpSecret(null);
    setTotpUri(null);
  }, [mfaEnabled, totpFlow]);

  React.useEffect(() => {
    if (!mfaNotice) return;
    const timer = window.setTimeout(() => setMfaNotice(null), 5000);
    return () => window.clearTimeout(timer);
  }, [mfaNotice]);

  React.useEffect(() => {
    let active = true;
    if (!totpUri) {
      setTotpQrDataUrl(null);
      return () => {
        active = false;
      };
    }

    try {
      const qr = qrcodeGenerator(0, "M");
      qr.addData(totpUri);
      qr.make();
      const svg = qr.createSvgTag(5, 1);
      const dataUrl = `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(
        svg
      )}`;
      if (active) setTotpQrDataUrl(dataUrl);
    } catch {
      if (active) setTotpQrDataUrl(null);
    }

    return () => {
      active = false;
    };
  }, [totpUri]);

  const showTotpSetupForm = totpFlow === "setup" && !mfaEnabled && Boolean(totpSecret);
  const showTotpDisableForm = totpFlow === "disable" && mfaEnabled;

  const handleToggleDisableForm = () => {
    setTotpDisableCode("");
    setTotpFlow((prev) => (prev === "disable" ? "idle" : "disable"));
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
                autoComplete="current-password"
                {...form.register("current_password")}
              />
            </label>
            <label className="block text-sm">
              New password
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                type="password"
                autoComplete="new-password"
                {...form.register("new_password")}
              />
            </label>
            <label className="block text-sm">
              Confirm new password
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                type="password"
                autoComplete="new-password"
                {...form.register("new_password_confirm")}
              />
            </label>
            <Button type="submit" className="w-full sm:w-auto">
              Update password
            </Button>
          </form>
        </Card>

        <Card variant="bordered" className="space-y-4">
          <h2 className="text-lg font-semibold">Multi-factor authentication</h2>
          <p className="text-sm text-foreground/70">
            Add an authenticator app or passkey to protect your account.
          </p>
          <div className="rounded-lg border border-border bg-muted p-3 text-sm">
            <p>Status: {mfaEnabled ? "Enabled" : "Not enabled"}</p>
            <p className="mt-1 text-xs text-foreground/70">
              Backup codes remaining: {statusQuery.data?.backup_codes_remaining ?? 0}
            </p>
          </div>
          <div className="grid gap-2 sm:flex sm:flex-wrap">
            {!mfaEnabled ? (
              <Button
                variant="secondary"
                className="w-full sm:w-auto"
                onClick={handleTotpSetup}
                disabled={setupTotp.isPending}
              >
                {setupTotp.isPending ? "Preparing..." : "Set up authenticator"}
              </Button>
            ) : (
              <Button
                variant="secondary"
                className="w-full sm:w-auto"
                onClick={handleToggleDisableForm}
              >
                {showTotpDisableForm ? "Cancel disable" : "Disable MFA"}
              </Button>
            )}
            <Button
              variant="secondary"
              className="w-full sm:w-auto"
              onClick={handleBackupCodes}
              disabled={!mfaEnabled || regenerateBackupCodes.isPending}
            >
              {!mfaEnabled
                ? "Backup codes require MFA"
                : regenerateBackupCodes.isPending
                ? "Generating..."
                : "Generate backup codes"}
            </Button>
          </div>
          {!mfaEnabled ? (
            <p className="text-xs text-foreground/70">
              Generate backup codes after enabling MFA. They are recovery codes for your
              authenticator setup.
            </p>
          ) : null}
          {mfaNotice ? (
            <p className="rounded-lg border border-border bg-muted px-3 py-2 text-sm text-foreground/75">
              {mfaNotice}
            </p>
          ) : null}
          {showTotpSetupForm ? (
            <div className="space-y-2 rounded-lg border border-border bg-muted p-3 text-sm">
              <p className="font-semibold">Step 1: Scan QR code in your authenticator app</p>
              {totpQrDataUrl ? (
                <Image
                  src={totpQrDataUrl}
                  alt="Authenticator setup QR code"
                  width={176}
                  height={176}
                  unoptimized
                  loading="lazy"
                  decoding="async"
                  className="mx-auto h-44 w-44 rounded-lg border border-border bg-card p-1 sm:mx-0"
                />
              ) : totpUri ? (
                <p className="text-xs text-foreground/70">
                  Generating QR code...
                </p>
              ) : null}
              <p className="font-semibold">Step 2: Or enter this setup key manually</p>
              <div className="rounded border border-border bg-card px-2 py-1 font-mono text-xs break-all">
                {totpSecret}
              </div>
              <div className="grid gap-2 sm:flex sm:flex-wrap">
                <Button
                  size="sm"
                  variant="secondary"
                  className="w-full sm:w-auto"
                  onClick={() => copyMfaValue(totpSecret || "", "Setup key")}
                >
                  Copy setup key
                </Button>
                {totpUri ? (
                  <Button
                    size="sm"
                    variant="secondary"
                    className="w-full sm:w-auto"
                    onClick={() => copyMfaValue(totpUri, "Setup URI")}
                  >
                    Copy setup URI
                  </Button>
                ) : null}
              </div>
              <label className="block text-sm">
                Step 3: Enter 6-digit code from app
                <input
                  className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                  value={totpSetupCode}
                  onChange={(event) => setTotpSetupCode(event.target.value)}
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  placeholder="123456"
                />
              </label>
              <div className="grid gap-2 sm:flex sm:flex-wrap">
                <Button
                  className="w-full sm:w-auto"
                  onClick={handleTotpVerify}
                  disabled={verifyTotp.isPending || totpSetupCode.trim().length < 6}
                >
                  {verifyTotp.isPending ? "Enabling..." : "Enable MFA"}
                </Button>
                <Button
                  variant="ghost"
                  className="w-full sm:w-auto"
                  onClick={() => {
                    setTotpFlow("idle");
                    setTotpSecret(null);
                    setTotpUri(null);
                    setTotpSetupCode("");
                  }}
                >
                  Cancel
                </Button>
              </div>
            </div>
          ) : null}
          {showTotpDisableForm ? (
            <div className="space-y-2 rounded-lg border border-border bg-muted p-3 text-sm">
              <p className="font-semibold">
                Confirm with your authenticator code to disable MFA
              </p>
              <label className="block text-sm">
                Verification code
                <input
                  className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                  value={totpDisableCode}
                  onChange={(event) => setTotpDisableCode(event.target.value)}
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  placeholder="123456"
                />
              </label>
              <div className="grid gap-2 sm:flex sm:flex-wrap">
                <Button
                  variant="ghost"
                  className="w-full sm:w-auto"
                  onClick={handleTotpDisable}
                  disabled={disableTotp.isPending || totpDisableCode.trim().length < 6}
                >
                  {disableTotp.isPending ? "Disabling..." : "Disable MFA"}
                </Button>
                <Button
                  variant="ghost"
                  className="w-full sm:w-auto"
                  onClick={() => setTotpFlow("idle")}
                >
                  Cancel
                </Button>
              </div>
            </div>
          ) : null}
          {backupCodes.length ? (
            <div className="rounded-lg border border-border bg-muted p-3 text-sm">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="font-semibold">Backup codes</p>
                <Button
                  size="sm"
                  variant="secondary"
                  className="w-full sm:w-auto"
                  onClick={handleBackupCodesDownload}
                >
                  Download .txt
                </Button>
              </div>
              <p className="mt-1 text-xs text-foreground/70">
                Each code can be used once. Store them somewhere safe.
              </p>
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                {backupCodes.map((code) => (
                  <span key={code} className="rounded bg-card px-2 py-1 font-mono">
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
        <div className="grid gap-2 sm:grid-cols-[1fr_auto] sm:items-center">
          <input
            className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
            placeholder="Passkey nickname (optional)"
            value={passkeyName}
            onChange={(event) => setPasskeyName(event.target.value)}
          />
          <Button
            className="w-full sm:min-w-[12rem] lg:min-w-[14rem]"
            onClick={handleRegisterPasskey}
            disabled={passkeyPending}
          >
            {passkeyPending ? "Adding passkey..." : "Add passkey"}
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
                  className="w-full sm:w-auto"
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
          <Button
            variant="secondary"
            className="w-full sm:w-auto"
            onClick={() => revokeOthers.mutate()}
          >
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
                  className="w-full sm:w-auto"
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
