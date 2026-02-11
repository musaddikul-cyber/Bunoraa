export function getOAuthBaseUrl(): string {
  const apiBase = (process.env.NEXT_PUBLIC_API_BASE_URL || "").trim();
  if (apiBase) {
    try {
      return new URL(apiBase).origin;
    } catch {
      if (apiBase.startsWith("/") && typeof window !== "undefined") {
        return window.location.origin;
      }
    }
  }
  return typeof window !== "undefined" ? window.location.origin : "";
}

export function buildGoogleOAuthUrl(callbackPath: string): string {
  const base = getOAuthBaseUrl().replace(/\/$/, "");
  const next = encodeURIComponent(callbackPath);
  const path = `/oauth/login/google-oauth2/?next=${next}`;
  return base ? `${base}${path}` : path;
}
