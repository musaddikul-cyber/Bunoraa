"use client";

import * as React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch, ApiError } from "@/lib/api";
import {
  AUTH_EVENT_NAME,
  clearTokens,
  getAccessToken,
  getActiveAccountId,
  getStoredAccounts,
  removeStoredAccount,
  setTokens,
  switchAccount as switchStoredAccount,
  upsertActiveAccountProfile,
  type StoredAuthAccount,
} from "@/lib/auth";
import type { UserProfile } from "@/lib/types";

type LoginInput = {
  email: string;
  password: string;
  remember: boolean;
};

type RegisterInput = {
  email: string;
  password: string;
  password_confirm: string;
  first_name?: string;
  last_name?: string;
  phone?: string;
};

type LoginResponse = {
  access?: string;
  refresh?: string;
  mfa_required?: boolean;
  mfa_token?: string;
  methods?: string[];
};

async function fetchProfile() {
  const response = await apiFetch<UserProfile>("/accounts/profile/", {
    method: "GET",
  });
  return response.data;
}

export function useAuth() {
  const queryClient = useQueryClient();
  const [hasToken, setHasToken] = React.useState(false);
  const [accounts, setAccounts] = React.useState<StoredAuthAccount[]>([]);
  const [activeAccountId, setActiveAccountId] = React.useState<string | null>(null);
  const lastKnownToken = React.useRef(getAccessToken() || "");

  const syncHasToken = React.useCallback(() => {
    const token = getAccessToken() || "";
    const nextHasToken = Boolean(token);
    setHasToken(nextHasToken);
    setAccounts(getStoredAccounts());
    setActiveAccountId(getActiveAccountId());

    if (lastKnownToken.current !== token) {
      lastKnownToken.current = token;
      queryClient.invalidateQueries({ queryKey: ["profile"] });
      queryClient.invalidateQueries({ queryKey: ["cart"] });
      queryClient.invalidateQueries({ queryKey: ["wishlist"] });
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
    }
  }, [queryClient]);

  React.useEffect(() => {
    syncHasToken();
  }, [syncHasToken]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = () => syncHasToken();
    window.addEventListener(AUTH_EVENT_NAME, handler);
    window.addEventListener("storage", handler);
    return () => {
      window.removeEventListener(AUTH_EVENT_NAME, handler);
      window.removeEventListener("storage", handler);
    };
  }, [syncHasToken]);

  const profileQuery = useQuery({
    queryKey: ["profile"],
    queryFn: fetchProfile,
    enabled: hasToken,
    retry: (count, error) => {
      if (error instanceof ApiError && error.status === 401) return false;
      return count < 2;
    },
  });

  React.useEffect(() => {
    if (!profileQuery.data) return;
    upsertActiveAccountProfile({
      email: profileQuery.data.email,
      first_name: profileQuery.data.first_name,
      full_name: profileQuery.data.full_name,
    });
  }, [profileQuery.data]);

  const login = useMutation({
    mutationFn: async ({ email, password, remember }: LoginInput) => {
      const response = await apiFetch<LoginResponse>(
        "/auth/token/",
        {
          method: "POST",
          body: { email, password },
        }
      );
      if (!response.data.mfa_required) {
        setTokens(response.data.access || "", response.data.refresh, remember);
      }
      return response.data;
    },
    onSuccess: () => {
      syncHasToken();
    },
  });

  const verifyMfa = useMutation({
    mutationFn: async (payload: {
      mfa_token: string;
      method: "totp" | "backup_code" | "passkey";
      code?: string;
      credential?: unknown;
      remember?: boolean;
    }) => {
      const response = await apiFetch<LoginResponse>("/accounts/mfa/verify/", {
        method: "POST",
        body: payload,
      });
      if (!response.data.mfa_required) {
        setTokens(
          response.data.access || "",
          response.data.refresh,
          Boolean(payload.remember)
        );
      }
      return response.data;
    },
    onSuccess: () => {
      syncHasToken();
    },
  });

  const register = useMutation({
    mutationFn: async (input: RegisterInput) => {
      const response = await apiFetch<UserProfile>("/accounts/register/", {
        method: "POST",
        body: input,
      });
      return response.data;
    },
  });

  const updateProfile = useMutation({
    mutationFn: async (payload: Partial<UserProfile>) => {
      const response = await apiFetch<UserProfile>("/accounts/profile/", {
        method: "PATCH",
        body: payload,
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["profile"] });
    },
  });

  const logout = React.useCallback(() => {
    clearTokens();
    syncHasToken();
    queryClient.removeQueries({ queryKey: ["profile"] });
  }, [queryClient, syncHasToken]);

  const logoutAll = React.useCallback(() => {
    clearTokens();
    syncHasToken();
    queryClient.removeQueries({ queryKey: ["profile"] });
  }, [queryClient, syncHasToken]);

  const switchAccount = React.useCallback(
    (accountId: string) => {
      const switched = switchStoredAccount(accountId);
      if (switched) {
        syncHasToken();
      }
      return switched;
    },
    [syncHasToken]
  );

  const removeAccount = React.useCallback(
    (accountId: string) => {
      const removed = removeStoredAccount(accountId);
      if (removed) {
        syncHasToken();
      }
      return removed;
    },
    [syncHasToken]
  );

  return {
    hasToken,
    accounts,
    activeAccountId,
    profileQuery,
    login,
    verifyMfa,
    register,
    updateProfile,
    switchAccount,
    removeAccount,
    logout,
    logoutAll,
  };
}
