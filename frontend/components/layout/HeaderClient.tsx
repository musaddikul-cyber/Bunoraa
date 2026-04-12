"use client";

import * as React from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { usePathname } from "next/navigation";
import { Handbag, Heart, UserRound } from "lucide-react";
import { NotificationBell } from "@/components/notifications/NotificationBell";
import { useCart } from "@/components/cart/useCart";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { useWishlist } from "@/components/wishlist/useWishlist";
import { useNotifications } from "@/components/notifications/useNotifications";
import { useToast } from "@/components/ui/ToastProvider";

const CartDrawer = dynamic(
  () => import("@/components/cart/CartDrawer").then((mod) => mod.CartDrawer),
  { ssr: false }
);

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

export function HeaderClient() {
  const pathname = usePathname();
  const [mounted, setMounted] = React.useState(false);
  const [open, setOpen] = React.useState(false);
  const [menuOpen, setMenuOpen] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement | null>(null);
  const { push } = useToast();
  const {
    hasToken,
    profileQuery,
    accounts,
    activeAccountId,
    switchAccount,
    logout,
  } = useAuthContext();
  const shouldLoadHeaderCounts = mounted && hasToken;
  const { cartQuery, cartSummaryQuery } = useCart({
    includeCart: open,
    includeSummary: shouldLoadHeaderCounts,
  });
  const { wishlistQuery } = useWishlist({ enabled: shouldLoadHeaderCounts });
  const { unreadCountQuery } = useNotifications(undefined, {
    includeList: false,
    includeUnread: true,
  });
  const count =
    cartSummaryQuery.data?.item_count ??
    cartQuery.data?.item_count ??
    0;
  const wishlistCount =
    wishlistQuery.data?.meta?.pagination?.count ??
    wishlistQuery.data?.data?.length ??
    0;
  const unreadCount = unreadCountQuery.data?.count ?? 0;
  const hasUnreadNotifications = unreadCount > 0;
  const hasProfileAvatar = Boolean(profileQuery.data?.avatar);
  const adminPanelHref = React.useMemo(() => resolveBackendAdminUrl(), []);
  const otherAccounts = React.useMemo(
    () => accounts.filter((account) => account.id !== activeAccountId),
    [accounts, activeAccountId]
  );
  const addAccountHref = React.useMemo(() => {
    const nextPath = pathname || "/account/profile/";
    return `/account/login/?next=${encodeURIComponent(nextPath)}&add_account=1`;
  }, [pathname]);

  const getAccountLabel = React.useCallback(
    (account: {
      email?: string;
      full_name?: string;
      first_name?: string;
      id: string;
    }) =>
      account.email ||
      account.full_name ||
      account.first_name ||
      `Account ${account.id.slice(0, 8)}`,
    []
  );

  React.useEffect(() => {
    setMounted(true);
  }, []);

  React.useEffect(() => {
    if (!mounted) return;
    if (count <= 0) return;
    if (typeof window === "undefined") return;
    const key = "cart_prompt_shown";
    if (window.sessionStorage.getItem(key)) return;
    push("You have items waiting in your bag.", "info");
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
    "relative inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-border/80 bg-card/90 text-sm leading-none text-foreground shadow-soft transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background";
  const iconTooltipClass =
    "pointer-events-none absolute left-1/2 top-full z-40 mt-2 hidden -translate-x-1/2 whitespace-nowrap rounded-md bg-foreground px-2 py-1 text-[11px] font-medium text-background opacity-0 shadow-soft transition-opacity duration-150 sm:block";

  return (
    <div className="flex items-center gap-2 sm:gap-3">
      <div className="group relative hidden sm:block">
        <NotificationBell className={iconButtonClass} count={unreadCount} />
        <span className={`${iconTooltipClass} group-hover:opacity-100 group-focus-within:opacity-100`} aria-hidden="true">
          Notifications
        </span>
      </div>
      <Link
        href="/wishlist/"
        prefetch={false}
        className={`group hidden sm:inline-flex ${iconButtonClass}`}
        aria-label="Wishlist"
      >
        <Heart className="h-5 w-5" strokeWidth={1.8} aria-hidden="true" />
        <span className="sr-only">Wishlist</span>
        <span className={`${iconTooltipClass} group-hover:opacity-100 group-focus-visible:opacity-100`} aria-hidden="true">
          Wishlist
        </span>
        {wishlistCount > 0 ? (
          <span className="absolute -right-1 -top-1 rounded-full bg-accent px-1.5 py-0.5 text-[11px] font-semibold text-white">
            {wishlistCount}
          </span>
        ) : null}
      </Link>
      <button
        type="button"
        className={`group ${iconButtonClass}`}
        onClick={() => setOpen((prev) => !prev)}
        aria-label="Bag"
      >
        <Handbag className="h-5 w-5" strokeWidth={1.8} aria-hidden="true" />
        <span className="sr-only">Bag</span>
        <span className={`${iconTooltipClass} group-hover:opacity-100 group-focus-visible:opacity-100`} aria-hidden="true">
          Bag
        </span>
        {count > 0 ? (
          <span className="absolute -right-1 -top-1 rounded-full bg-primary px-1.5 py-0.5 text-[11px] font-semibold text-white">
            {count}
          </span>
        ) : null}
      </button>
      <div className="group relative flex items-center" ref={menuRef}>
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
            <UserRound className="h-5 w-5" strokeWidth={1.8} aria-hidden="true" />
          )}
          {hasToken && hasUnreadNotifications ? (
            <span
              className="absolute right-0.5 top-0.5 h-2.5 w-2.5 rounded-full bg-accent ring-2 ring-card"
              aria-hidden="true"
            />
          ) : null}
        </button>
        {!menuOpen ? (
          <span className={`${iconTooltipClass} group-hover:opacity-100 group-focus-within:opacity-100`} aria-hidden="true">
            Account
          </span>
        ) : null}
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
              {otherAccounts.length ? (
                <div className="mt-1 border-t border-border pt-1">
                  <p className="px-3 py-1 text-[11px] uppercase tracking-[0.16em] text-foreground/50">
                    Switch Account
                  </p>
                  {otherAccounts.map((account) => (
                    <button
                      key={account.id}
                      type="button"
                      className="block w-full truncate rounded-lg px-3 py-2 text-left text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                      onClick={() => {
                        switchAccount(account.id);
                        setMenuOpen(false);
                      }}
                    >
                      {getAccountLabel(account)}
                    </button>
                  ))}
                </div>
              ) : null}
              {accounts.length < 5 ? (
                <Link
                  href={addAccountHref}
                  className="block truncate rounded-lg px-3 py-2 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                  role="menuitem"
                  onClick={() => setMenuOpen(false)}
                >
                  Add account
                </Link>
              ) : (
                <p className="px-3 py-2 text-xs text-foreground/60">Account limit reached (5)</p>
              )}
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
                  target="_blank"
                  rel="noopener noreferrer"
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
      {open ? <CartDrawer isOpen={open} onClose={() => setOpen(false)} /> : null}
    </div>
  );
}
