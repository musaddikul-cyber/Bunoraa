import type { NextConfig } from "next";

function toRemotePattern(urlString?: string) {
  if (!urlString) return null;
  try {
    const url = new URL(urlString);
    return {
      protocol: url.protocol.replace(":", ""),
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
) as Array<NonNullable<ReturnType<typeof toRemotePattern>>>;

const nextConfig: NextConfig = {
  trailingSlash: true,
  images: {
    remotePatterns,
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
