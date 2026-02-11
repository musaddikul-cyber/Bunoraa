import type { MetadataRoute } from "next";

const SITE_URL =
  (process.env.NEXT_PUBLIC_SITE_URL || "https://bunoraa.com").replace(/\/$/, "");

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        disallow: [
          "/api/",
          "/admin/",
          "/account/",
          "/cart/",
          "/checkout/",
          "/orders/",
          "/wishlist/",
          "/compare/",
          "/notifications/",
          "/subscriptions/",
          "/preorders/",
          "/search/",
          "/static/",
          "/media/uploads/",
          "/oauth/",
          "/email/",
          "/health/",
          "/api/schema/",
          "/api/schema/swagger-ui/",
          "/api/schema/redoc/",
        ],
      },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
