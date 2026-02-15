import { NextResponse } from "next/server";

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

Sitemap: https://bunoraa.com/sitemap.xml
`;

export function GET() {
  return new NextResponse(body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "public, max-age=3600",
    },
  });
}
