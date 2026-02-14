"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { NotificationBell } from "@/components/notifications/NotificationBell";
import { CartDrawer } from "@/components/cart/CartDrawer";
import { useCart } from "@/components/cart/useCart";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { useWishlist } from "@/components/wishlist/useWishlist";
import { useNotifications } from "@/components/notifications/useNotifications";
import { useToast } from "@/components/ui/ToastProvider";

function resolveBackendAdminUrl() {
  const apiBase = (process.env.NEXT_PUBLIC_API_BASE_URL || "").trim();
  if (!apiBase) return "/admin/";

  const stripApiSuffix = (value: string) =>
    value.replace(/\/api(?:\/v\d+)?\/?$/i, "");

  if (apiBase.startsWith("/")) {
    const basePath = stripApiSuffix(apiBase.replace(/\/+$/, ""));
    return `${basePath || ""}/admin/`;
  }

  try {
    const parsed = new URL(apiBase);
    const cleanPath = stripApiSuffix(parsed.pathname.replace(/\/+$/, ""));
    parsed.pathname = `${cleanPath || ""}/admin/`;
    parsed.search = "";
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return "/admin/";
  }
}

function HeartIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      className="h-5 w-5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M20.8 6.6a5.5 5.5 0 0 0-9.1-3.9L12 3l.3-.3a5.5 5.5 0 0 0-7.7 7.7L12 18.8l7.4-7.4a5.5 5.5 0 0 0 1.4-4.8z" />
    </svg>
  );
}

function CartIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      className="h-5 w-5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M6 7h14l-1.5 8.5a2 2 0 0 1-2 1.5H9a2 2 0 0 1-2-1.5L5 3H3" />
      <circle cx="9" cy="20" r="1.5" />
      <circle cx="17" cy="20" r="1.5" />
    </svg>
  );
}

function UserIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      className="h-5 w-5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="8" r="3.5" />
      <path d="M4 20a8 8 0 0 1 16 0" />
    </svg>
  );
}

export function HeaderClient() {
  const pathname = usePathname();
  const [mounted, setMounted] = React.useState(false);
  const [open, setOpen] = React.useState(false);
  const [menuOpen, setMenuOpen] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement | null>(null);
  const { push } = useToast();
  const { cartQuery } = useCart();
  const { hasToken, profileQuery, logout } = useAuthContext();
  const { wishlistQuery } = useWishlist({ enabled: mounted && hasToken });
  const { unreadCountQuery } = useNotifications();
  const count = cartQuery.data?.item_count ?? 0;
  const wishlistCount =
    wishlistQuery.data?.meta?.pagination?.count ??
    wishlistQuery.data?.data?.length ??
    0;
  const unreadCount = unreadCountQuery.data?.count ?? 0;
  const hasUnreadNotifications = unreadCount > 0;
  const hasProfileAvatar = Boolean(profileQuery.data?.avatar);
  const adminPanelHref = React.useMemo(() => resolveBackendAdminUrl(), []);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  React.useEffect(() => {
    if (!mounted) return;
    if (count <= 0) return;
    if (typeof window === "undefined") return;
    const key = "cart_prompt_shown";
    if (window.sessionStorage.getItem(key)) return;
    push("You have items waiting in your cart.", "info");
    window.sessionStorage.setItem(key, "true");
  }, [count, mounted, push]);

  React.useEffect(() => {
    if (!menuOpen) return;
    const handleClick = (event: MouseEvent) => {
      if (!menuRef.current) return;
      if (!menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [menuOpen]);

  React.useEffect(() => {
    setOpen(false);
    setMenuOpen(false);
  }, [pathname]);

  const iconButtonClass =
    "relative inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-border/80 bg-card/90 text-sm leading-none text-foreground shadow-soft transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:h-10 sm:w-10";

  return (
    <div className="flex items-center gap-2 sm:gap-3">
      <div className="hidden sm:block">
        <NotificationBell className={iconButtonClass} count={unreadCount} />
      </div>
      <Link
        href="/wishlist/"
        className={`hidden sm:inline-flex ${iconButtonClass}`}
        aria-label="Wishlist"
      >
        <HeartIcon />
        <span className="sr-only">Wishlist</span>
        {wishlistCount > 0 ? (
          <span className="absolute -right-1 -top-1 rounded-full bg-accent px-1.5 py-0.5 text-[11px] font-semibold text-white">
            {wishlistCount}
          </span>
        ) : null}
      </Link>
      <button
        type="button"
        className={iconButtonClass}
        onClick={() => setOpen((prev) => !prev)}
        aria-label="Cart"
      >
        <CartIcon />
        <span className="sr-only">Cart</span>
        {count > 0 ? (
          <span className="absolute -right-1 -top-1 rounded-full bg-primary px-1.5 py-0.5 text-[11px] font-semibold text-white">
            {count}
          </span>
        ) : null}
      </button>
      <div className="relative flex items-center" ref={menuRef}>
        <button
          type="button"
          className={`${iconButtonClass} ${hasProfileAvatar ? "overflow-hidden p-0" : ""}`}
          onClick={() => setMenuOpen((prev) => !prev)}
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          aria-label="Account menu"
        >
          {mounted && hasToken ? (
            hasProfileAvatar ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={profileQuery.data?.avatar || ""}
                alt={profileQuery.data?.first_name || "Profile"}
                className="h-full w-full object-cover"
              />
            ) : (
              <span className="relative flex h-7 w-7 items-center justify-center overflow-hidden rounded-full bg-muted text-[10px] font-semibold uppercase text-foreground/70">
                {profileQuery.data?.first_name?.[0] || "U"}
              </span>
            )
          ) : (
            <UserIcon />
          )}
          {hasToken && hasUnreadNotifications ? (
            <span
              className="absolute right-0.5 top-0.5 h-2.5 w-2.5 rounded-full bg-accent ring-2 ring-card"
              aria-hidden="true"
            />
          ) : null}
        </button>
        {menuOpen ? (
          mounted && hasToken ? (
            <div
              className="absolute right-0 top-[calc(100%+0.5rem)] z-50 w-56 max-w-[calc(100vw-2rem)] rounded-xl border border-border bg-card p-2 shadow-lg"
              role="menu"
            >
              <div className="border-b border-border px-3 py-2">
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/50">
                  Signed in
                </p>
                <p className="truncate text-sm font-semibold">
                  {profileQuery.data?.full_name ||
                    profileQuery.data?.first_name ||
                    "Account"}
                </p>
                {profileQuery.data?.email ? (
                  <p className="truncate text-xs text-foreground/60">
                    {profileQuery.data.email}
                  </p>
                ) : null}
              </div>
              <Link
                href="/account/dashboard/"
                className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
              >
                Dashboard
              </Link>
              <Link
                href="/account/profile/"
                className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
              >
                Profile
              </Link>
              <Link
                href="/account/orders/"
                className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
              >
                Orders
              </Link>
              <Link
                href="/notifications/"
                className="flex items-center justify-between gap-3 rounded-lg px-3 py-2 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 sm:hidden"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
              >
                <span className="truncate">Notifications</span>
                {unreadCount > 0 ? (
                  <span className="inline-flex min-w-[1.5rem] items-center justify-center rounded-full bg-muted px-2 py-0.5 text-xs font-semibold text-foreground/80">
                    {unreadCount}
                  </span>
                ) : null}
              </Link>
              {profileQuery.data?.is_superuser || profileQuery.data?.is_staff ? (
                <Link
                  href={adminPanelHref}
                  className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                  role="menuitem"
                  onClick={() => setMenuOpen(false)}
                >
                  Admin panel
                </Link>
              ) : null}
              <button
                className="mt-1 w-full truncate rounded-lg px-3 py-2 text-left text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                onClick={() => {
                  setMenuOpen(false);
                  logout();
                }}
                role="menuitem"
                type="button"
              >
                Logout
              </button>
            </div>
          ) : (
            <div
              className="absolute right-0 top-[calc(100%+0.5rem)] z-50 w-56 max-w-[calc(100vw-2rem)] rounded-xl border border-border bg-card p-2 shadow-lg"
              role="menu"
            >
              <div className="border-b border-border px-3 py-2">
                <p className="text-xs uppercase tracking-[0.2em] text-foreground/50">Account</p>
                <p className="text-sm font-semibold">Welcome</p>
              </div>
              <Link
                href="/account/login/"
                className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
              >
                Sign in
              </Link>
              <Link
                href="/account/register/"
                className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
              >
                Create account
              </Link>
              <div className="my-1 border-t border-border" role="separator" />
              <Link
                href="/faq/"
                className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
              >
                FAQ
              </Link>
              <Link
                href="/contact/"
                className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
              >
                Contact support
              </Link>
            </div>
          )
        ) : null}
      </div>
      <CartDrawer isOpen={open} onClose={() => setOpen(false)} />
    </div>
  );
}
