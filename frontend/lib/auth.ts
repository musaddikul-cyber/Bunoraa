const ACCESS_KEY = "access_token";
const REFRESH_KEY = "refresh_token";
export const AUTH_EVENT_NAME = "bunoraa:auth";

function notifyAuthChange() {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(AUTH_EVENT_NAME));
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

export function setTokens(access: string, refresh?: string, remember = true) {
  if (typeof window === "undefined") return;
  const storage = getStorage(remember);
  storage.setItem(ACCESS_KEY, access);
  if (refresh) storage.setItem(REFRESH_KEY, refresh);

  if (remember) {
    window.sessionStorage.removeItem(ACCESS_KEY);
    window.sessionStorage.removeItem(REFRESH_KEY);
  } else {
    window.localStorage.removeItem(ACCESS_KEY);
    window.localStorage.removeItem(REFRESH_KEY);
  }
  notifyAuthChange();
}

export function clearTokens() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(ACCESS_KEY);
  window.localStorage.removeItem(REFRESH_KEY);
  window.sessionStorage.removeItem(ACCESS_KEY);
  window.sessionStorage.removeItem(REFRESH_KEY);
  notifyAuthChange();
}

export function setAccessToken(access: string) {
  if (typeof window === "undefined") return;
  const storageType = getTokenStorageType();
  const storage =
    storageType === "session" ? window.sessionStorage : window.localStorage;
  storage.setItem(ACCESS_KEY, access);
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
