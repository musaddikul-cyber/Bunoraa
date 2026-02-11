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
const fallbackMediaPattern = {
  protocol: "https",
  hostname: "media.bunoraa.com",
  port: undefined,
  pathname: "/**",
} as const;

const remotePatterns = [mediaPattern, apiPattern, fallbackMediaPattern].filter(
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

const apiProxyOrigin =
  toOrigin(process.env.NEXT_API_PROXY_TARGET) ||
  toOrigin(process.env.NEXT_INTERNAL_API_BASE_URL) ||
  null;
const shouldProxyMedia =
  !process.env.NEXT_PUBLIC_MEDIA_BASE_URL ||
  process.env.NEXT_PUBLIC_MEDIA_BASE_URL.startsWith("/");

const disableImageOptimization =
  process.env.NEXT_IMAGE_UNOPTIMIZED === "true" ||
  process.env.NEXT_IMAGE_UNOPTIMIZED === "1";

const nextConfig: NextConfig = {
  trailingSlash: true,
  images: {
    remotePatterns,
    unoptimized: disableImageOptimization,
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
      { source: "/account/", destination: "/account/dashboard/", permanent: false },
    ];
  },
  async rewrites() {
    if (!apiProxyOrigin) return [];
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
