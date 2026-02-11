import type { ApiResponse } from "@/lib/types";
import { clearTokens, getRefreshToken, setAccessToken } from "@/lib/auth";
import { getLocaleHeaders } from "@/lib/locale";

type ApiFetchOptions = {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  headers?: HeadersInit;
  params?: Record<string, string | number | boolean | Array<string | number | boolean> | undefined>;
  next?: { revalidate?: number };
  cache?: RequestCache;
  signal?: AbortSignal;
  retryOnCsrf?: boolean;
  retryOnAuth?: boolean;
  skipAuth?: boolean;
  allowGuest?: boolean;
  suppressError?: boolean;
  suppressErrorStatus?: number[];
};

const PUBLIC_API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "");
const INTERNAL_API_BASE_URL = (process.env.NEXT_INTERNAL_API_BASE_URL || "").replace(/\/$/, "");
const API_BASE_URL =
  typeof window === "undefined" && INTERNAL_API_BASE_URL
    ? INTERNAL_API_BASE_URL
    : PUBLIC_API_BASE_URL;
const FALLBACK_SITE_URL =
  (process.env.NEXT_PUBLIC_SITE_URL || "").replace(/\/$/, "") || "http://localhost:3000";
const DISABLE_BUILD_PRERENDER =
  process.env.NEXT_DISABLE_BUILD_PRERENDER === "true" ||
  process.env.NEXT_DISABLE_BUILD_PRERENDER === "1";
let refreshPromise: Promise<string | null> | null = null;

function ensureTrailingSlash(path: string) {
  if (!path.endsWith("/")) {
    return `${path}/`;
  }
  return path;
}

function buildUrl(path: string, params?: ApiFetchOptions["params"]) {
  const normalizedPath = ensureTrailingSlash(path.startsWith("/") ? path : `/${path}`);
  if (!API_BASE_URL) {
    throw new Error("NEXT_PUBLIC_API_BASE_URL is not set");
  }
  const base = API_BASE_URL;
  const url =
    base.startsWith("/")
      ? new URL(
          `${base}${normalizedPath}`,
          typeof window !== "undefined" ? window.location.origin : FALLBACK_SITE_URL
        )
      : new URL(`${base}${normalizedPath}`);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value === undefined || value === null || value === "") return;
      if (Array.isArray(value)) {
        value.forEach((item) => url.searchParams.append(key, String(item)));
      } else {
        url.searchParams.set(key, String(value));
      }
    });
  }
  return url;
}

function getAccessToken() {
  if (typeof window === "undefined") return null;
  return (
    window.localStorage.getItem("access_token") ||
    window.sessionStorage.getItem("access_token")
  );
}

function getCookie(name: string) {
  if (typeof document === "undefined") return "";
  const value = document.cookie
    .split(";")
    .map((c) => c.trim())
    .find((c) => c.startsWith(`${name}=`));
  return value ? decodeURIComponent(value.split("=")[1] || "") : "";
}

function setCookie(name: string, value: string) {
  if (typeof document === "undefined") return;
  const secure = window.location.protocol === "https:" ? "; secure" : "";
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; samesite=Lax${secure}`;
}

function extractErrorMessage(json: unknown): string | null {
  if (!json) return null;
  if (typeof json === "string" && json.trim()) return json.trim();
  if (Array.isArray(json) && json.length > 0) {
    return String(json[0]);
  }
  if (typeof json !== "object") return null;
  const record = json as Record<string, unknown>;
  if (typeof record.message === "string" && record.message.trim()) return record.message.trim();
  if (typeof record.error === "string" && record.error.trim()) return record.error.trim();
  if (typeof record.detail === "string" && record.detail.trim()) return record.detail.trim();
  if (Array.isArray(record.non_field_errors) && record.non_field_errors.length) {
    return String(record.non_field_errors[0]);
  }

  const humanize = (value: string) =>
    value
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase());

  const pickFieldError = (errors: Record<string, unknown>) => {
    for (const [key, value] of Object.entries(errors)) {
      if (!key) continue;
      if (Array.isArray(value) && value.length) {
        const message = String(value[0]);
        return key === "non_field_errors" ? message : `${humanize(key)}: ${message}`;
      }
      if (typeof value === "string" && value.trim()) {
        return key === "non_field_errors"
          ? value.trim()
          : `${humanize(key)}: ${value.trim()}`;
      }
    }
    return null;
  };

  if (record.errors && typeof record.errors === "object") {
    const errors = record.errors as Record<string, unknown>;
    const fieldMessage = pickFieldError(errors);
    if (fieldMessage) return fieldMessage;
  }

  const fieldMessage = pickFieldError(record);
  if (fieldMessage) return fieldMessage;

  for (const value of Object.values(record)) {
    if (Array.isArray(value) && value.length) return String(value[0]);
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return null;
}

export class ApiError extends Error {
  status: number;
  data?: unknown;
  path?: string;

  constructor(message: string, status: number, data?: unknown, path?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
    this.path = path;
  }
}

async function parseJsonSafe(response: Response) {
  try {
    const text = await response.text();
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch {
      return { _text: text };
    }
  } catch {
    return null;
  }
}

async function refreshAccessToken() {
  if (typeof window === "undefined") return null;
  const refresh = getRefreshToken();
  if (!refresh || !API_BASE_URL) return null;
  if (refreshPromise) return refreshPromise;

  refreshPromise = fetch(buildUrl("/auth/token/refresh/"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Requested-With": "XMLHttpRequest",
    },
    credentials: "include",
    body: JSON.stringify({ refresh }),
  })
    .then(async (response) => {
      if (!response.ok) return null;
      const json = await parseJsonSafe(response);
      const access =
        json?.access ||
        (json && typeof json === "object" && "data" in json ? json.data?.access : null);
      if (access) {
        setAccessToken(access);
        return access as string;
      }
      return null;
    })
    .catch(() => null)
    .finally(() => {
      refreshPromise = null;
    });

  return refreshPromise;
}

export async function apiFetch<T>(path: string, options: ApiFetchOptions = {}): Promise<ApiResponse<T>> {
  const {
    method = "GET",
    body,
    headers,
    params,
    next,
    cache,
    signal,
    retryOnCsrf = true,
    retryOnAuth = true,
    skipAuth = false,
    allowGuest = false,
    suppressError = false,
    suppressErrorStatus = [],
  } = options;

  const url = buildUrl(path, params);
  const token = skipAuth ? null : getAccessToken();
  const csrfToken = getCookie("csrftoken");
  const localeHeaders = getLocaleHeaders();
  const isFormData =
    typeof FormData !== "undefined" && body instanceof FormData;
  const forceNoStore = DISABLE_BUILD_PRERENDER && typeof window === "undefined";
  const effectiveCache = forceNoStore ? "no-store" : cache;
  const effectiveNext = forceNoStore ? undefined : next;

  const init: RequestInit & { next?: { revalidate?: number } } = {
    method,
    headers: {
      ...(isFormData ? {} : { "Content-Type": "application/json" }),
      "X-Requested-With": "XMLHttpRequest",
      ...(csrfToken ? { "X-CSRFToken": csrfToken } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...localeHeaders,
      ...headers,
    },
    credentials: "include",
    body:
      body && method !== "GET"
        ? isFormData
          ? body
          : JSON.stringify(body)
        : undefined,
    cache: effectiveCache,
    signal,
  };

  if (effectiveNext) {
    init.next = effectiveNext;
  }

  const response = await fetch(url, init);

  const json = await parseJsonSafe(response);

  if (!response.ok) {
    if (response.status === 401 && retryOnAuth && typeof window !== "undefined") {
      const refreshed = await refreshAccessToken();
      if (refreshed) {
        return apiFetch<T>(path, { ...options, retryOnAuth: false });
      }
      clearTokens();
      if (allowGuest && !skipAuth) {
        return apiFetch<T>(path, {
          ...options,
          skipAuth: true,
          retryOnAuth: false,
        });
      }
    }
    const newToken = json?.meta?.new_csrf_token;
    if (response.status === 403 && newToken && retryOnCsrf && typeof window !== "undefined") {
      setCookie("csrftoken", newToken);
      return apiFetch<T>(path, { ...options, retryOnCsrf: false });
    }
    const rawText =
      json &&
      typeof json === "object" &&
      "_text" in json &&
      typeof (json as { _text?: unknown })._text === "string"
        ? String((json as { _text?: string })._text)
        : "";
    const safeText = rawText && !rawText.includes("<") ? rawText.trim() : "";
    const extracted = extractErrorMessage(json);
    const message = extracted || safeText || response.statusText || "Request failed";
    const shouldSuppress =
      suppressError || (suppressErrorStatus && suppressErrorStatus.includes(response.status));
    if (typeof window !== "undefined" && !shouldSuppress) {
      const path = url.toString();
      console.error("API error", path, response.status, message, json);
    }
    throw new ApiError(message, response.status, json, url.toString());
  }

  if (json && typeof json === "object" && "data" in json) {
    return json as ApiResponse<T>;
  }

  if (json && typeof json === "object" && "_text" in json) {
    return {
      success: true,
      message: "OK",
      data: (json as { _text: string })._text as T,
      meta: null,
    };
  }

  if (json && typeof json === "object" && Array.isArray((json as { results?: unknown }).results)) {
    const resultJson = json as {
      results: T;
      count?: number;
      next?: string | null;
      previous?: string | null;
    };
    const pageSize = Array.isArray(resultJson.results) ? resultJson.results.length : 0;
    const pageParam = url.searchParams.get("page");
    const page = pageParam ? Number(pageParam) || 1 : 1;
    return {
      success: true,
      message: "OK",
      data: resultJson.results,
      meta: {
        pagination: {
          count: resultJson.count ?? pageSize,
          next: resultJson.next ?? null,
          previous: resultJson.previous ?? null,
          page,
          page_size: pageSize,
          total_pages:
            resultJson.count && pageSize > 0
              ? Math.max(1, Math.ceil(resultJson.count / pageSize))
              : 1,
        },
      },
    };
  }

  return {
    success: true,
    message: "OK",
    data: json as T,
    meta: null,
  };
}
