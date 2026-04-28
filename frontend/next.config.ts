import type { NextConfig } from "next";
import type { RemotePattern } from "next/dist/shared/lib/image-config";
import path from "path";

function toRemotePattern(urlString?: string): RemotePattern | null {
  if (!urlString) return null;
  try {
    const url = new URL(urlString);
    const protocol = url.protocol.replace(":", "");
    if (protocol !== "http" && protocol !== "https") {
      return null;
    }
    return {
      protocol,
      hostname: url.hostname,
      port: url.port || undefined,
      pathname: "/**",
    };
  } catch {
    return null;
  }
}

const mediaPattern = toRemotePattern(process.env.NEXT_PUBLIC_MEDIA_BASE_URL);
const apiPattern = toRemotePattern(process.env.NEXT_PUBLIC_API_BASE_URL);
const localhostPattern = {
  protocol: "http",
  hostname: "localhost",
  port: "8000",
  pathname: "/**",
} as const;

const remotePatterns = [mediaPattern, apiPattern, localhostPattern].filter(
  Boolean
) as RemotePattern[];

function toOrigin(urlString?: string): string | null {
  if (!urlString) return null;
  try {
    return new URL(urlString).origin;
  } catch {
    return null;
  }
}

function parseBooleanEnv(value?: string): boolean | null {
  if (!value) return null;
  const normalized = value.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;
  return null;
}

const apiProxyOrigin =
  toOrigin(process.env.NEXT_INTERNAL_API_BASE_URL) ||
  toOrigin(process.env.NEXT_PUBLIC_API_BASE_URL) ||
  null;
const siteOrigin =
  toOrigin(process.env.NEXT_PUBLIC_SITE_URL) ||
  toOrigin(process.env.URL) ||
  toOrigin(process.env.DEPLOY_PRIME_URL) ||
  null;
const apiProxyMode = parseBooleanEnv(process.env.NEXT_PUBLIC_API_USE_PROXY);
const shouldUseApiProxy =
  Boolean(apiProxyOrigin) &&
  (apiProxyMode ??
    Boolean(siteOrigin && apiProxyOrigin && apiProxyOrigin !== siteOrigin));
const shouldProxyMedia =
  !process.env.NEXT_PUBLIC_MEDIA_BASE_URL ||
  process.env.NEXT_PUBLIC_MEDIA_BASE_URL.startsWith("/");

const shouldDisableImageOptimization =
  process.env.NODE_ENV !== "production";

const nextConfig: NextConfig = {
  trailingSlash: true,
  images: {
    remotePatterns,
    unoptimized: shouldDisableImageOptimization,
    formats: ["image/avif", "image/webp"],
    // Optimize image loading with responsive device sizes
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    // Cache optimized images for longer period
    minimumCacheTTL: 60,
    qualities: [60, 64, 72, 75],
    dangerouslyAllowSVG: true,
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
  },
  // Enable compression and optimize bundle size
  compress: true,
  // Enable experimental optimizations
  experimental: {
    // Optimize package imports
    optimizePackageImports: [
      "@/components",
      "@/lib",
      "@/app",
    ],
  },
  turbopack: {
    resolveAlias: {
      "@": path.resolve(__dirname),
    },
  },
  async redirects() {
    return [
      { source: "/catalog", destination: "/", permanent: true },
      { source: "/catalog/", destination: "/", permanent: true },
      { source: "/catalog/products/:path*", destination: "/products/:path*", permanent: true },
      { source: "/catalog/category/:path*", destination: "/categories/:path*", permanent: true },
      { source: "/products/category/:path*", destination: "/categories/:path*", permanent: true },
      { source: "/categories/category/:path*", destination: "/categories/:path*", permanent: true },
      { source: "/account/", destination: "/account/profile/", permanent: false },
      { source: "/account/dashboard/", destination: "/account/profile/", permanent: false },
      {
        source: "/account/notifications/preferences/",
        destination: "/account/notifications/",
        permanent: false,
      },
    ];
  },
  async rewrites() {
    if (!apiProxyOrigin || !shouldUseApiProxy) return [];
    const rules = [
      { source: "/sitemap.xml", destination: `${apiProxyOrigin}/sitemap.xml` },
      { source: "/sitemap-:section.xml", destination: `${apiProxyOrigin}/sitemap-:section.xml` },
      { source: "/api/:path*", destination: `${apiProxyOrigin}/api/:path*` },
      { source: "/oauth/:path*", destination: `${apiProxyOrigin}/oauth/:path*` },
    ];
    if (shouldProxyMedia) {
      rules.push({ source: "/media/:path*", destination: `${apiProxyOrigin}/media/:path*` });
    }
    return rules;
  },
};

export default nextConfig;
