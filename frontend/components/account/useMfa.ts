import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { MfaStatus, WebAuthnCredential } from "@/lib/types";

const mfaKey = ["account", "mfa"] as const;
const passkeyKey = ["account", "passkeys"] as const;

async function fetchMfaStatus() {
  const response = await apiFetch<MfaStatus>("/accounts/mfa/status/");
  return response.data;
}

async function fetchPasskeys() {
  const response = await apiFetch<WebAuthnCredential[]>("/accounts/webauthn/credentials/");
  return response.data;
}

export function useMfa() {
  const queryClient = useQueryClient();

  const statusQuery = useQuery({
    queryKey: mfaKey,
    queryFn: fetchMfaStatus,
  });

  const passkeysQuery = useQuery({
    queryKey: passkeyKey,
    queryFn: fetchPasskeys,
  });

  const setupTotp = useMutation({
    mutationFn: async () => {
      const response = await apiFetch<{ secret: string; otpauth_url: string }>(
        "/accounts/mfa/totp/setup/",
        { method: "POST" }
      );
      return response.data;
    },
  });

  const verifyTotp = useMutation({
    mutationFn: async (code: string) => {
      const response = await apiFetch<{ backup_codes: string[] }>(
        "/accounts/mfa/totp/verify/",
        { method: "POST", body: { code } }
      );
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: mfaKey });
    },
  });

  const disableTotp = useMutation({
    mutationFn: async (code: string) => {
      return apiFetch("/accounts/mfa/totp/disable/", {
        method: "POST",
        body: { code },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: mfaKey });
    },
  });

  const regenerateBackupCodes = useMutation({
    mutationFn: async () => {
      const response = await apiFetch<{ backup_codes: string[] }>(
        "/accounts/mfa/backup-codes/regenerate/",
        { method: "POST" }
      );
      return response.data;
    },
  });

  const registerPasskeyOptions = useMutation({
    mutationFn: async () => {
      const response = await apiFetch<Record<string, unknown>>(
        "/accounts/webauthn/register/options/",
        { method: "POST" }
      );
      return response.data;
    },
  });

  const registerPasskeyVerify = useMutation({
    mutationFn: async (payload: { credential: unknown; nickname?: string }) => {
      return apiFetch("/accounts/webauthn/register/verify/", {
        method: "POST",
        body: payload,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: passkeyKey });
      queryClient.invalidateQueries({ queryKey: mfaKey });
    },
  });

  const removePasskey = useMutation({
    mutationFn: async (credentialId: string) => {
      return apiFetch(`/accounts/webauthn/credentials/${credentialId}/`, {
        method: "DELETE",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: passkeyKey });
      queryClient.invalidateQueries({ queryKey: mfaKey });
    },
  });

  return {
    statusQuery,
    passkeysQuery,
    setupTotp,
    verifyTotp,
    disableTotp,
    regenerateBackupCodes,
    registerPasskeyOptions,
    registerPasskeyVerify,
    removePasskey,
  };
}
