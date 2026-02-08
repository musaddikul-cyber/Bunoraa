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
};

export default nextConfig;
