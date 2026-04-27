import { getBrowserApiOrigin } from "@/lib/api";

export function getOAuthBaseUrl(): string {
  return getBrowserApiOrigin();
}

export function buildGoogleOAuthUrl(callbackPath: string): string {
  const base = getOAuthBaseUrl().replace(/\/$/, "");
  const next = encodeURIComponent(callbackPath);
  const path = `/oauth/login/google-oauth2/?next=${next}`;
  return base ? `${base}${path}` : path;
}
