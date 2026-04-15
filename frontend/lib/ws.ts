export function normalizeWsUrl(baseUrl: string): string | null {
  if (!baseUrl) return null;
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  if (!trimmed) return null;

  if (trimmed.startsWith("ws://") || trimmed.startsWith("wss://")) {
    return trimmed;
  }

  if (trimmed.startsWith("http://")) {
    return `ws://${trimmed.slice("http://".length)}`;
  }

  if (trimmed.startsWith("https://")) {
    return `wss://${trimmed.slice("https://".length)}`;
  }

  if (trimmed.startsWith("/")) {
    if (typeof window === "undefined") return null;
    const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${scheme}//${window.location.host}${trimmed}`;
  }

  return trimmed;
}

export function getWsBaseUrl(): string | null {
  const envBase = (process.env.NEXT_PUBLIC_WS_BASE_URL || "").trim();
  if (envBase) {
    return normalizeWsUrl(envBase) || null;
  }

  if (typeof window === "undefined") {
    return null;
  }

  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}`;
}

export function buildWsUrl(path: string, token?: string | null): string | null {
  const base = getWsBaseUrl();
  if (!base) return null;
  const normalizedPath = base.endsWith("/ws") ? path.replace(/^\/ws/, "") : path;
  const url = `${base}${normalizedPath}`;
  if (!token) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}token=${encodeURIComponent(token)}`;
}
