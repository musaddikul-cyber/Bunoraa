import type { MetadataRoute } from "next";
import { absoluteUrl } from "@/lib/seo";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      allow: "/",
      disallow: [
        "/api/",
        "/admin/",
        "/oauth/",
        "/email/",
        "/health/",
        "/api/schema/",
        "/api/schema/swagger-ui/",
        "/api/schema/redoc/",
      ],
    },
    sitemap: absoluteUrl("/sitemap.xml"),
  };
}
