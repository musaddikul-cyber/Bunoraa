"use client";

import * as React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch, ApiError } from "@/lib/api";
import { clearTokens, getAccessToken, setTokens } from "@/lib/auth";
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
  const hasToken = Boolean(getAccessToken());

  const profileQuery = useQuery({
    queryKey: ["profile"],
    queryFn: fetchProfile,
    enabled: hasToken,
    retry: (count, error) => {
      if (error instanceof ApiError && error.status === 401) return false;
      return count < 2;
    },
  });

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
      queryClient.invalidateQueries({ queryKey: ["profile"] });
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
      queryClient.invalidateQueries({ queryKey: ["profile"] });
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
    queryClient.removeQueries({ queryKey: ["profile"] });
  }, [queryClient]);

  return {
    hasToken,
    profileQuery,
    login,
    verifyMfa,
    register,
    updateProfile,
    logout,
  };
}
