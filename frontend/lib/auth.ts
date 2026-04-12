const ACCESS_KEY = "access_token";
const REFRESH_KEY = "refresh_token";
const ACCOUNTS_KEY = "bunoraa:auth_accounts";
const ACTIVE_ACCOUNT_KEY = "bunoraa:active_account";
const MAX_STORED_ACCOUNTS = 5;
export const AUTH_EVENT_NAME = "bunoraa:auth";

export type StoredAuthAccount = {
  id: string;
  access: string;
  refresh?: string;
  remember: boolean;
  email?: string;
  first_name?: string;
  full_name?: string;
  created_at: number;
  last_used_at: number;
};

type AccountProfileMeta = {
  email?: string | null;
  first_name?: string | null;
  full_name?: string | null;
};

function notifyAuthChange() {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(AUTH_EVENT_NAME));
}

function parseJwtPayload(token: string): Record<string, unknown> | null {
  if (typeof window === "undefined") return null;
  const parts = token.split(".");
  if (parts.length < 2) return null;
  try {
    const base64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    const padded = `${base64}${"=".repeat((4 - (base64.length % 4)) % 4)}`;
    const decoded = window.atob(padded);
    const payload = JSON.parse(decoded);
    if (payload && typeof payload === "object") {
      return payload as Record<string, unknown>;
    }
  } catch {
    return null;
  }
  return null;
}

function extractAccountId(access: string): string {
  const payload = parseJwtPayload(access);
  const candidates = [payload?.user_id, payload?.sub, payload?.email, payload?.jti];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate.trim();
    }
  }
  return `token-${access.slice(-24)}`;
}

function isStoredAuthAccount(value: unknown): value is StoredAuthAccount {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<StoredAuthAccount>;
  return (
    typeof candidate.id === "string" &&
    typeof candidate.access === "string" &&
    typeof candidate.remember === "boolean" &&
    typeof candidate.created_at === "number" &&
    typeof candidate.last_used_at === "number"
  );
}

function readStoredAccounts(): StoredAuthAccount[] {
  if (typeof window === "undefined") return [];
  const raw = window.localStorage.getItem(ACCOUNTS_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isStoredAuthAccount).slice(0, MAX_STORED_ACCOUNTS);
  } catch {
    return [];
  }
}

function writeStoredAccounts(accounts: StoredAuthAccount[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(
    ACCOUNTS_KEY,
    JSON.stringify(accounts.slice(0, MAX_STORED_ACCOUNTS))
  );
}

function getTokenStorageType() {
  if (typeof window === "undefined") return null;
  if (window.localStorage.getItem(ACCESS_KEY)) return "local";
  if (window.sessionStorage.getItem(ACCESS_KEY)) return "session";
  return null;
}

function getStorage(remember: boolean) {
  return remember ? window.localStorage : window.sessionStorage;
}

function setActiveAccountId(accountId: string) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(ACTIVE_ACCOUNT_KEY, accountId);
}

function clearActiveTokenPair() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(ACCESS_KEY);
  window.localStorage.removeItem(REFRESH_KEY);
  window.sessionStorage.removeItem(ACCESS_KEY);
  window.sessionStorage.removeItem(REFRESH_KEY);
  window.localStorage.removeItem(ACTIVE_ACCOUNT_KEY);
}

function activateStoredAccount(account: StoredAuthAccount) {
  if (typeof window === "undefined") return;
  const storage = getStorage(account.remember);
  storage.setItem(ACCESS_KEY, account.access);
  if (account.refresh) {
    storage.setItem(REFRESH_KEY, account.refresh);
  } else {
    storage.removeItem(REFRESH_KEY);
  }

  if (account.remember) {
    window.sessionStorage.removeItem(ACCESS_KEY);
    window.sessionStorage.removeItem(REFRESH_KEY);
  } else {
    window.localStorage.removeItem(ACCESS_KEY);
    window.localStorage.removeItem(REFRESH_KEY);
  }
  setActiveAccountId(account.id);
}

function mergeAndLimitAccounts(nextAccount: StoredAuthAccount, existing: StoredAuthAccount[]) {
  const deduped = existing.filter((account) => account.id !== nextAccount.id);
  return [nextAccount, ...deduped].slice(0, MAX_STORED_ACCOUNTS);
}

export function setTokens(access: string, refresh?: string, remember = true) {
  if (typeof window === "undefined") return;
  const now = Date.now();
  const accountId = extractAccountId(access);
  const existingAccounts = readStoredAccounts();
  const existingAccount = existingAccounts.find((account) => account.id === accountId);
  const nextAccount: StoredAuthAccount = {
    id: accountId,
    access,
    refresh: refresh || existingAccount?.refresh,
    remember,
    email: existingAccount?.email,
    first_name: existingAccount?.first_name,
    full_name: existingAccount?.full_name,
    created_at: existingAccount?.created_at || now,
    last_used_at: now,
  };
  const accounts = mergeAndLimitAccounts(nextAccount, existingAccounts);
  writeStoredAccounts(accounts);
  activateStoredAccount(nextAccount);
  notifyAuthChange();
}

export function getStoredAccounts(): StoredAuthAccount[] {
  return readStoredAccounts();
}

export function getActiveAccountId() {
  if (typeof window === "undefined") return null;
  const stored = window.localStorage.getItem(ACTIVE_ACCOUNT_KEY);
  if (stored) return stored;

  const token =
    window.localStorage.getItem(ACCESS_KEY) || window.sessionStorage.getItem(ACCESS_KEY);
  if (!token) return null;
  const accountId = extractAccountId(token);
  setActiveAccountId(accountId);
  return accountId;
}

export function switchAccount(accountId: string) {
  if (typeof window === "undefined") return false;
  const existingAccounts = readStoredAccounts();
  const target = existingAccounts.find((account) => account.id === accountId);
  if (!target) return false;

  const updatedTarget: StoredAuthAccount = {
    ...target,
    last_used_at: Date.now(),
  };
  const accounts = mergeAndLimitAccounts(updatedTarget, existingAccounts);
  writeStoredAccounts(accounts);
  activateStoredAccount(updatedTarget);
  notifyAuthChange();
  return true;
}

export function removeStoredAccount(accountId: string) {
  if (typeof window === "undefined") return false;
  const existingAccounts = readStoredAccounts();
  if (!existingAccounts.some((account) => account.id === accountId)) {
    return false;
  }
  const nextAccounts = existingAccounts.filter((account) => account.id !== accountId);
  writeStoredAccounts(nextAccounts);

  const activeAccountId = getActiveAccountId();
  if (activeAccountId === accountId) {
    if (nextAccounts.length) {
      activateStoredAccount(nextAccounts[0]);
    } else {
      clearActiveTokenPair();
    }
  }

  notifyAuthChange();
  return true;
}

export function logoutActiveAccount() {
  if (typeof window === "undefined") return;
  const activeAccountId = getActiveAccountId();
  if (!activeAccountId) {
    clearActiveTokenPair();
    notifyAuthChange();
    return;
  }
  removeStoredAccount(activeAccountId);
}

export function upsertActiveAccountProfile(meta: AccountProfileMeta) {
  if (typeof window === "undefined") return;
  const activeAccountId = getActiveAccountId();
  if (!activeAccountId) return;

  const existingAccounts = readStoredAccounts();
  let changed = false;
  const nextAccounts = existingAccounts.map((account) => {
    if (account.id !== activeAccountId) return account;
    const nextEmail = typeof meta.email === "string" ? meta.email : account.email;
    const nextFirstName =
      typeof meta.first_name === "string" ? meta.first_name : account.first_name;
    const nextFullName =
      typeof meta.full_name === "string" ? meta.full_name : account.full_name;
    if (
      nextEmail !== account.email ||
      nextFirstName !== account.first_name ||
      nextFullName !== account.full_name
    ) {
      changed = true;
    }
    return {
      ...account,
      email: nextEmail,
      first_name: nextFirstName,
      full_name: nextFullName,
    };
  });
  if (!changed) return;
  writeStoredAccounts(nextAccounts);
  notifyAuthChange();
}

export function clearTokens() {
  if (typeof window === "undefined") return;
  clearActiveTokenPair();
  window.localStorage.removeItem(ACCOUNTS_KEY);
  notifyAuthChange();
}

export function setAccessToken(access: string) {
  if (typeof window === "undefined") return;
  const activeAccountId = getActiveAccountId() || extractAccountId(access);
  const existingAccounts = readStoredAccounts();
  const existing = existingAccounts.find((account) => account.id === activeAccountId);

  if (!existing) {
    setTokens(access, undefined, getTokenStorageType() !== "session");
    return;
  }

  const updated: StoredAuthAccount = {
    ...existing,
    access,
    last_used_at: Date.now(),
  };
  const nextAccounts = mergeAndLimitAccounts(updated, existingAccounts);
  writeStoredAccounts(nextAccounts);
  activateStoredAccount(updated);
  notifyAuthChange();
}

export function getAccessToken() {
  if (typeof window === "undefined") return null;
  return (
    window.localStorage.getItem(ACCESS_KEY) ||
    window.sessionStorage.getItem(ACCESS_KEY)
  );
}

export function getRefreshToken() {
  if (typeof window === "undefined") return null;
  return (
    window.localStorage.getItem(REFRESH_KEY) ||
    window.sessionStorage.getItem(REFRESH_KEY)
  );
}

export function isAuthenticated() {
  return Boolean(getAccessToken());
}

export function getTokenStoragePreference() {
  return getTokenStorageType();
}
