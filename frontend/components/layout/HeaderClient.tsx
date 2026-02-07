"use client";

import * as React from "react";
import Link from "next/link";
import { NotificationBell } from "@/components/notifications/NotificationBell";
import { CartDrawer } from "@/components/cart/CartDrawer";
import { useCart } from "@/components/cart/useCart";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { useWishlist } from "@/components/wishlist/useWishlist";

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

export function HeaderClient() {
  const [mounted, setMounted] = React.useState(false);
  const [open, setOpen] = React.useState(false);
  const [menuOpen, setMenuOpen] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement | null>(null);
  const { cartQuery } = useCart();
  const { hasToken, profileQuery, logout } = useAuthContext();
  const { wishlistQuery } = useWishlist({ enabled: mounted && hasToken });
  const count = cartQuery.data?.item_count ?? 0;
  const wishlistCount =
    wishlistQuery.data?.meta?.pagination?.count ??
    wishlistQuery.data?.data?.length ??
    0;

  React.useEffect(() => {
    setMounted(true);
  }, []);

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

  return (
    <div className="flex items-center gap-2 sm:gap-3">
      <div className="hidden sm:flex">
        <NotificationBell />
      </div>
      <Link
        href="/wishlist/"
        className="relative hidden p-2 text-sm sm:inline-flex"
        aria-label="Wishlist"
      >
        <HeartIcon />
        <span className="sr-only">Wishlist</span>
        {wishlistCount > 0 ? (
          <span className="absolute -right-2 -top-2 rounded-full bg-accent px-2 py-0.5 text-xs text-white">
            {wishlistCount}
          </span>
        ) : null}
      </Link>
      <button
        className="relative p-2 text-sm"
        onClick={() => setOpen((prev) => !prev)}
        aria-label="Cart"
      >
        <CartIcon />
        <span className="sr-only">Cart</span>
        {count > 0 ? (
          <span className="absolute right-0 -top-1 rounded-full bg-primary px-2 py-0.5 text-xs text-white">
            {count}
          </span>
        ) : null}
      </button>
      {!mounted ? (
        <Link href="/account/login/" className="text-sm">
          Sign in
        </Link>
      ) : hasToken ? (
        <div className="relative" ref={menuRef}>
          <button
            className="inline-flex items-center gap-2 rounded-full px-2 py-1 text-sm sm:px-3"
            onClick={() => setMenuOpen((prev) => !prev)}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
          >
            <span className="relative flex h-7 w-7 items-center justify-center overflow-hidden rounded-full bg-muted text-[10px] font-semibold uppercase text-foreground/70">
              {profileQuery.data?.avatar ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={profileQuery.data.avatar}
                  alt={profileQuery.data?.first_name || "Profile"}
                  className="h-full w-full object-cover"
                />
              ) : (
                (profileQuery.data?.first_name?.[0] || "U")
              )}
            </span>
          </button>
          {menuOpen ? (
            <div
              className="absolute right-0 top-full z-50 mt-2 w-56 max-w-[calc(100vw-2rem)] rounded-xl border border-border bg-card p-2 shadow-lg"
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
                className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
              >
                Dashboard
              </Link>
              <Link
                href="/account/profile/"
                className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
              >
                Profile
              </Link>
              <Link
                href="/account/orders/"
                className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
              >
                Orders
              </Link>
              {profileQuery.data?.is_superuser || profileQuery.data?.is_staff ? (
                <Link
                  href="/admin/"
                  className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted"
                  role="menuitem"
                  onClick={() => setMenuOpen(false)}
                >
                  Admin panel
                </Link>
              ) : null}
              <button
                className="mt-1 w-full truncate rounded-lg px-3 py-2 text-left text-sm hover:bg-muted"
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
          ) : null}
        </div>
      ) : (
        <Link href="/account/login/" className="text-sm">
          Sign in
        </Link>
      )}
      <CartDrawer isOpen={open} onClose={() => setOpen(false)} />
    </div>
  );
}
