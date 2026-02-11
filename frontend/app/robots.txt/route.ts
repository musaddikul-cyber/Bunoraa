import { NextResponse } from "next/server";

const body = `User-agent: *
Allow: /
Disallow: /api/
Disallow: /admin/
Disallow: /account/
Disallow: /cart/
Disallow: /checkout/
Disallow: /orders/
Disallow: /wishlist/
Disallow: /compare/
Disallow: /notifications/
Disallow: /subscriptions/
Disallow: /preorders/
Disallow: /search/
Disallow: /static/
Disallow: /media/uploads/
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
