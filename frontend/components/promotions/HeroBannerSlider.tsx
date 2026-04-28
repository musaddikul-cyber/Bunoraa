"use client";

import * as React from "react";
import Image from "next/image";
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
  content_vertical_position?: "top" | "center" | "bottom" | null;
  content_horizontal_alignment?: "left" | "center" | "right" | null;
  button_alignment?: "left" | "center" | "right" | null;
  title_font_size?: string | null;
  subtitle_font_size?: string | null;
  button_font_size?: string | null;
  button_padding?: string | null;
  button_min_height?: string | null;
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
  autoAdvance = false,
}: {
  banners: HeroBanner[];
  className?: string;
  intervalMs?: number;
  autoAdvance?: boolean;
}) {
  const [activeIndex, setActiveIndex] = React.useState(0);
  const [loadedSlides, setLoadedSlides] = React.useState<Set<number>>(
    () => new Set([0])
  );
  const total = banners.length;
  const defaultHeight = "min(600px, calc(100dvh - var(--header-offset, 5.5rem)))";

  React.useEffect(() => {
    if (!autoAdvance || total <= 1) return;
    const timer = window.setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % total);
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [autoAdvance, intervalMs, total]);

  React.useEffect(() => {
    if (activeIndex >= total) {
      setActiveIndex(0);
    }
  }, [activeIndex, total]);

  React.useEffect(() => {
    setLoadedSlides((prev) => {
      if (prev.has(activeIndex)) return prev;
      const next = new Set(prev);
      next.add(activeIndex);
      return next;
    });
  }, [activeIndex]);

  if (!total) return null;

  const primary = banners[0];
  const containerStyle: React.CSSProperties = {
    height: toCssValue(primary.style_height) || defaultHeight,
    width: toCssValue(primary.style_width),
    maxWidth: toCssValue(primary.style_max_width),
  };

  return (
    <div className={cn("w-full", className)} style={containerStyle}>
      <div className="relative h-full overflow-hidden">
        {banners.map((banner, index) => {
          const isActive = index === activeIndex;
          const hasLoadedImage = loadedSlides.has(index) || index === 0;
          const borderWidth = toCssValue(banner.style_border_width);
          const borderColor = banner.style_border_color || undefined;
          const resolvedBorderWidth = borderWidth || (borderColor ? "1px" : undefined);
          const contentVerticalPosition = banner.content_vertical_position || "bottom";
          const contentHorizontalAlignment = banner.content_horizontal_alignment || "left";
          const buttonAlignment = banner.button_alignment || contentHorizontalAlignment;
          const verticalPositionClass =
            contentVerticalPosition === "top"
              ? "justify-start"
              : contentVerticalPosition === "center"
                ? "justify-center"
                : "justify-end";
          const horizontalAlignmentClass =
            contentHorizontalAlignment === "center"
              ? "items-center text-center"
              : contentHorizontalAlignment === "right"
                ? "items-end text-right"
                : "items-start text-left";
          const buttonAlignmentClass =
            buttonAlignment === "center"
              ? "self-center"
              : buttonAlignment === "right"
                ? "self-end"
                : "self-start";
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
            borderWidth: resolvedBorderWidth,
            borderColor,
            borderStyle: resolvedBorderWidth || borderColor ? "solid" : undefined,
            backgroundColor: banner.style_background_color || undefined,
          };

          const textStyle: React.CSSProperties = {
            color: banner.text_color || undefined,
          };
          const titleStyle: React.CSSProperties = {
            fontSize: toCssValue(banner.title_font_size),
          };
          const subtitleStyle: React.CSSProperties = {
            fontSize: toCssValue(banner.subtitle_font_size),
          };
          const buttonStyle: React.CSSProperties = {
            fontSize: toCssValue(banner.button_font_size),
            padding: toCssValue(banner.button_padding),
            minHeight: toCssValue(banner.button_min_height),
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
              {hasLoadedImage ? (
                <Image
                  src={banner.image}
                  alt={banner.title}
                  fill
                  sizes="100vw"
                  quality={index === 0 ? 60 : 64}
                  priority={index === 0}
                  fetchPriority={index === 0 ? "high" : "low"}
                  loading={index === 0 ? "eager" : "lazy"}
                  decoding={index === 0 ? "sync" : "async"}
                  className="object-cover"
                />
              ) : (
                <div className="h-full w-full bg-muted" />
              )}
              <div
                className={cn(
                  "absolute inset-0 flex flex-col p-4 transition-opacity sm:p-6",
                  verticalPositionClass,
                  horizontalAlignmentClass,
                  overlayColor
                    ? ""
                    : "bg-gradient-to-t from-black/60 via-black/10 to-transparent"
                )}
                style={{ ...overlayStyle, ...textStyle }}
              >
                <h2 className="text-xl font-semibold leading-tight sm:text-2xl" style={titleStyle}>
                  {banner.title}
                </h2>
                {banner.subtitle ? (
                  <p className="mt-2 text-xs opacity-90 sm:text-sm" style={subtitleStyle}>
                    {banner.subtitle}
                  </p>
                ) : null}
                {banner.link_text ? (
                  <span
                    className={cn(
                      "mt-4 inline-flex w-fit items-center rounded-full border border-current/30 px-4 py-1.5 text-xs font-semibold uppercase tracking-[0.2em] sm:min-h-0",
                      buttonAlignmentClass
                    )}
                    style={buttonStyle}
                  >
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
        <div className="mt-3 hidden items-center justify-center gap-1.5 sm:flex">
          {banners.map((banner, index) => (
            <button
              key={banner.id}
              type="button"
              className={cn(
                "h-2 w-2 rounded-full border border-border",
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
