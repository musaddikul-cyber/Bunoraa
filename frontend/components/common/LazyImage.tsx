"use client";

import Image, { ImageProps } from "next/image";
import { getLazyImageProps } from "@/lib/lazyImage";

interface LazyImageProps extends Omit<ImageProps, "loading" | "decoding"> {
  /**
   * Optional custom loading component shown while image loads
   */
  loadingComponent?: React.ReactNode;
  /**
   * Enable blur placeholder (if available)
   */
  enableBlur?: boolean;
  /**
   * Custom blur placeholder
   */
  blurDataURL?: string;
  /**
   * Aspect ratio for the image container (e.g., "16/9")
   */
  aspectRatio?: string;
  /**
   * Show error placeholder on load failure
   */
  fallbackComponent?: React.ReactNode;
  /**
   * Optional wrapper class name for the outer container
   */
  wrapperClassName?: string;
}

/**
 * Minimal production-ready Image wrapper
 * - Defaults to native browser lazy loading for non-priority images
 * - Decodes lazily for deferred images
 * - Preserves aspect ratio and blur behavior when requested
 */
export function LazyImage({
  src,
  alt,
  loadingComponent,
  enableBlur = false,
  blurDataURL,
  priority = false,
  wrapperClassName,
  aspectRatio,
  fallbackComponent,
  placeholder,
  ...props
}: LazyImageProps) {
  const containerStyle = aspectRatio ? { aspectRatio, width: "100%" } : undefined;
  const shouldUseBlur = enableBlur && (blurDataURL || placeholder === "blur");

  return (
    <div className={wrapperClassName} style={containerStyle}>
      {src ? (
        <Image
          src={src}
          alt={alt}
          priority={priority}
          loading={priority ? "eager" : "lazy"}
          decoding={priority ? "sync" : "async"}
          placeholder={shouldUseBlur ? "blur" : undefined}
          blurDataURL={shouldUseBlur ? blurDataURL : undefined}
          {...props}
        />
      ) : (
        fallbackComponent || loadingComponent || <div className="bg-muted h-full w-full" />
      )}
    </div>
  );
}

