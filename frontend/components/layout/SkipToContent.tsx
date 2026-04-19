"use client";

/**
 * Skip to Content Link
 * 
 * Accessibility improvement: allows keyboard users to skip navigation
 * and jump directly to main content.
 */

import { useState, useCallback } from "react";

export function SkipToContent() {
  const [isVisible, setIsVisible] = useState(false);

  const handleFocus = useCallback(() => {
    setIsVisible(true);
  }, []);

  const handleBlur = useCallback(() => {
    setIsVisible(false);
  }, []);

  const handleClick = useCallback((e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    const mainContent = document.getElementById("main-content");
    if (mainContent) {
      mainContent.setAttribute("tabindex", "-1");
      mainContent.focus();
      mainContent.scrollIntoView({ behavior: "smooth" });
      // Remove tabindex after focus to maintain natural tab order
      setTimeout(() => {
        mainContent.removeAttribute("tabindex");
      }, 1000);
    }
  }, []);

  return (
    <a
      href="#main-content"
      onClick={handleClick}
      onFocus={handleFocus}
      onBlur={handleBlur}
      className={`
        fixed left-4 top-4 z-[9999] 
        transform transition-transform duration-200 ease-out
        bg-primary text-white 
        px-6 py-3 rounded-lg
        font-medium text-sm
        focus:outline-none focus:ring-4 focus:ring-primary/30
        ${isVisible ? "translate-y-0" : "-translate-y-[200%]"}
        hover:bg-primary/90 hover:shadow-lg
      `}
      aria-label="Skip to main content"
    >
      <span className="flex items-center gap-2">
        <svg
          aria-hidden="true"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={2}
          stroke="currentColor"
          className="w-4 h-4"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M19.5 13.5 12 21m0 0-7.5-7.5M12 21V3"
          />
        </svg>
        Skip to content
      </span>
    </a>
  );
}
