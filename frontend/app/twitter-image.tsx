import { ImageResponse } from "next/og";
import { SITE_NAME } from "@/lib/seo";

export const alt = `${SITE_NAME} storefront`;
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = "image/png";

export default function TwitterImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background:
            "linear-gradient(140deg, rgb(28, 25, 23) 0%, rgb(121, 45, 35) 55%, rgb(245, 158, 11) 100%)",
          color: "white",
          fontFamily: "sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 16,
            textAlign: "center",
            maxWidth: 900,
            padding: "0 40px",
          }}
        >
          <div style={{ fontSize: 74, fontWeight: 700, letterSpacing: 1 }}>
            {SITE_NAME}
          </div>
          <div style={{ fontSize: 34, opacity: 0.9 }}>
            Curated products, bundles, and artisan collections
          </div>
        </div>
      </div>
    ),
    size
  );
}
