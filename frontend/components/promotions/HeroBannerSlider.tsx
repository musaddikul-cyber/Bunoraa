"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

export type HeroBanner = {
  id: string;
  title: string;
  subtitle?: string | null;
  image: string;
  image_mobile?: string | null;
  link_url?: string | null;
  link_text?: string | null;
  style_height?: string | null;
  style_width?: string | null;
  style_max_width?: string | null;
  style_border_radius?: string | null;
  style_border_width?: string | null;
  style_border_color?: string | null;
  style_background_color?: string | null;
  overlay_color?: string | null;
  overlay_opacity?: string | number | null;
  text_color?: string | null;
};

const toCssValue = (value?: string | null) =>
  value && value.trim() ? value.trim() : undefined;

const clampOpacity = (value: number) => Math.min(1, Math.max(0, value));

const hexToRgba = (hex: string, opacity: number) => {
  const clean = hex.replace("#", "").trim();
  if (clean.length !== 3 && clean.length !== 6) return hex;
  const full =
    clean.length === 3
      ? clean
          .split("")
          .map((c) => c + c)
          .join("")
      : clean;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  if ([r, g, b].some((v) => Number.isNaN(v))) return hex;
  return `rgba(${r}, ${g}, ${b}, ${clampOpacity(opacity)})`;
};

export function HeroBannerSlider({
  banners,
  className,
  intervalMs = 6000,
}: {
  banners: HeroBanner[];
  className?: string;
  intervalMs?: number;
}) {
  const [activeIndex, setActiveIndex] = React.useState(0);
  const total = banners.length;
  const defaultHeight = "420px";

  React.useEffect(() => {
    if (total <= 1) return;
    const timer = window.setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % total);
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs, total]);

  React.useEffect(() => {
    if (activeIndex >= total) {
      setActiveIndex(0);
    }
  }, [activeIndex, total]);

  if (!total) return null;

  const primary = banners[0];
  const activeBanner = banners[Math.min(activeIndex, total - 1)] || primary;
  const containerStyle: React.CSSProperties = {
    height: toCssValue(primary.style_height) || defaultHeight,
    width: toCssValue(primary.style_width),
    maxWidth: toCssValue(primary.style_max_width),
  };
  const wrapperStyle: React.CSSProperties = {
    borderRadius: toCssValue(activeBanner.style_border_radius),
  };

  return (
    <div className={cn("w-full", className)} style={containerStyle}>
      <div className="relative h-full overflow-hidden rounded-2xl" style={wrapperStyle}>
        {banners.map((banner, index) => {
          const isActive = index === activeIndex;
          const borderWidth = toCssValue(banner.style_border_width);
          const borderColor = banner.style_border_color || undefined;
          const resolvedBorderWidth = borderWidth || (borderColor ? "1px" : undefined);
          const borderRadius = toCssValue(banner.style_border_radius);
          const overlayOpacityValue =
            banner.overlay_opacity === null || banner.overlay_opacity === undefined
              ? 0.6
              : Number(banner.overlay_opacity);
          const resolvedOverlayOpacity = Number.isNaN(overlayOpacityValue)
            ? 0.6
            : overlayOpacityValue;
          const overlayColor = banner.overlay_color
            ? hexToRgba(banner.overlay_color, resolvedOverlayOpacity)
            : undefined;

          const slideStyle: React.CSSProperties = {
            height: "100%",
            borderRadius,
            borderWidth: resolvedBorderWidth,
            borderColor,
            borderStyle: resolvedBorderWidth || borderColor ? "solid" : undefined,
            backgroundColor: banner.style_background_color || undefined,
          };

          const textStyle: React.CSSProperties = {
            color: banner.text_color || undefined,
          };

          const overlayStyle: React.CSSProperties = overlayColor
            ? { backgroundColor: overlayColor }
            : {};

          const content = (
            <div
              className={cn(
                "relative w-full overflow-hidden",
                isActive ? "opacity-100" : "opacity-0"
              )}
              style={slideStyle}
            >
              <picture>
                {banner.image_mobile ? (
                  <source media="(max-width: 640px)" srcSet={banner.image_mobile} />
                ) : null}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={banner.image}
                  alt={banner.title}
                  className="h-full w-full object-cover"
                  style={{ borderRadius: "inherit" }}
                />
              </picture>
              <div
                className={cn(
                  "absolute inset-0 flex flex-col justify-end p-6 transition-opacity",
                  overlayColor
                    ? ""
                    : "bg-gradient-to-t from-black/60 via-black/10 to-transparent"
                )}
                style={{ ...overlayStyle, ...textStyle }}
              >
                <h2 className="text-2xl font-semibold">{banner.title}</h2>
                {banner.subtitle ? (
                  <p className="mt-2 text-sm opacity-90">{banner.subtitle}</p>
                ) : null}
                {banner.link_text ? (
                  <span className="mt-4 inline-flex w-fit items-center rounded-full border border-current/30 px-4 py-1.5 text-xs font-semibold uppercase tracking-[0.2em]">
                    {banner.link_text}
                  </span>
                ) : null}
              </div>
            </div>
          );

          return (
            <div
              key={banner.id}
              className={cn(
                "absolute inset-0 transition-opacity duration-700",
                isActive ? "opacity-100" : "opacity-0 pointer-events-none"
              )}
            >
              {banner.link_url ? (
                <a
                  href={banner.link_url}
                  className="block h-full w-full transition hover:opacity-95"
                >
                  {content}
                </a>
              ) : (
                content
              )}
            </div>
          );
        })}
      </div>

      {total > 1 ? (
        <div className="mt-3 flex items-center justify-center gap-2">
          {banners.map((banner, index) => (
            <button
              key={banner.id}
              type="button"
              className={cn(
                "h-2.5 w-2.5 rounded-full border border-border",
                index === activeIndex ? "bg-primary" : "bg-muted"
              )}
              onClick={() => setActiveIndex(index)}
              aria-label={`Show banner ${index + 1}`}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}
