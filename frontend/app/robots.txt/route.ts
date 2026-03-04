import { NextResponse } from "next/server";
import { SITE_URL } from "@/lib/seo";

const sitemapUrl = `${SITE_URL.replace(/\/$/, "")}/sitemap.xml`;

const body = `User-agent: *
Allow: /
Disallow: /api/
Disallow: /admin/
Disallow: /oauth/
Disallow: /email/
Disallow: /health/
Disallow: /api/schema/
Disallow: /api/schema/swagger-ui/
Disallow: /api/schema/redoc/

Sitemap: ${sitemapUrl}
`;

export function GET() {
  return new NextResponse(body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "public, max-age=3600",
    },
  });
}
