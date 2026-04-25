"use client";

import * as React from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { usePathname } from "next/navigation";
import {
  Handbag,
  Heart,
  UserRound,
  User,
  Package,
  Settings,
  LogOut,
  Shield,
  ExternalLink,
  UserPlus,
  ArrowRightLeft,
  Bell,
  HelpCircle,
  Mail,
  ChevronRight,
  CreditCard,
  MapPin,
  Gift,
} from "lucide-react";
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
              className="absolute right-0 top-[calc(100%+0.5rem)] z-50 w-72 max-w-[calc(100vw-2rem)] origin-top-right animate-in fade-in zoom-in-95 rounded-2xl border border-border bg-card shadow-2xl"
              role="menu"
            >
              {/* ── User info header ── */}
              <div className="flex items-center gap-3 rounded-t-2xl bg-muted/40 px-4 py-3.5">
                <div className="flex h-11 w-11 shrink-0 items-center justify-center overflow-hidden rounded-full bg-primary/15 text-sm font-bold text-primary ring-2 ring-primary/20">
                  {hasProfileAvatar ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={profileQuery.data?.avatar || ""}
                      alt=""
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <span className="uppercase">
                      {profileQuery.data?.first_name?.[0] || "U"}
                    </span>
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold text-foreground">
                    {profileQuery.data?.full_name ||
                      profileQuery.data?.first_name ||
                      "Account"}
                  </p>
                  {profileQuery.data?.email ? (
                    <p className="truncate text-xs text-foreground/50">
                      {profileQuery.data.email}
                    </p>
                  ) : null}
                </div>
              </div>

              <div className="p-1.5">
                {/* ── Primary navigation ── */}
                <div className="space-y-0.5">
                  <Link
                    href="/account/profile/"
                    className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                    role="menuitem"
                    onClick={() => setMenuOpen(false)}
                  >
                    <User className="h-4 w-4 shrink-0 opacity-60" />
                    <span className="flex-1">Profile</span>
                    <ChevronRight className="h-3.5 w-3.5 shrink-0 text-foreground/20" />
                  </Link>
                  <Link
                    href="/account/orders/"
                    className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                    role="menuitem"
                    onClick={() => setMenuOpen(false)}
                  >
                    <Package className="h-4 w-4 shrink-0 opacity-60" />
                    <span className="flex-1">Orders</span>
                    <ChevronRight className="h-3.5 w-3.5 shrink-0 text-foreground/20" />
                  </Link>
                  <Link
                    href="/account/addresses/"
                    className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                    role="menuitem"
                    onClick={() => setMenuOpen(false)}
                  >
                    <MapPin className="h-4 w-4 shrink-0 opacity-60" />
                    <span className="flex-1">Addresses</span>
                    <ChevronRight className="h-3.5 w-3.5 shrink-0 text-foreground/20" />
                  </Link>
                  <Link
                    href="/account/payments/"
                    className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                    role="menuitem"
                    onClick={() => setMenuOpen(false)}
                  >
                    <CreditCard className="h-4 w-4 shrink-0 opacity-60" />
                    <span className="flex-1">Payments</span>
                    <ChevronRight className="h-3.5 w-3.5 shrink-0 text-foreground/20" />
                  </Link>
                  <Link
                    href="/wishlist/"
                    className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 sm:hidden"
                    role="menuitem"
                    onClick={() => setMenuOpen(false)}
                  >
                    <Heart className="h-4 w-4 shrink-0 opacity-60" />
                    <span className="flex-1">Wishlist</span>
                    {wishlistCount > 0 ? (
                      <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary">
                        {wishlistCount}
                      </span>
                    ) : null}
                  </Link>
                  <Link
                    href="/account/referrals/"
                    className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                    role="menuitem"
                    onClick={() => setMenuOpen(false)}
                  >
                    <Gift className="h-4 w-4 shrink-0 opacity-60" />
                    <span className="flex-1">Referrals</span>
                    <ChevronRight className="h-3.5 w-3.5 shrink-0 text-foreground/20" />
                  </Link>
                </div>

                {/* ── Notifications (mobile) ── */}
                <Link
                  href="/notifications/"
                  className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 sm:hidden"
                  role="menuitem"
                  onClick={() => setMenuOpen(false)}
                >
                  <Bell className="h-4 w-4 shrink-0 opacity-60" />
                  <span className="flex-1">Notifications</span>
                  {unreadCount > 0 ? (
                    <span className="rounded-full bg-accent/15 px-2 py-0.5 text-[10px] font-semibold text-accent">
                      {unreadCount}
                    </span>
                  ) : null}
                </Link>

                <div className="my-1.5 border-t border-border" role="separator" />

                {/* ── Settings & preferences ── */}
                <Link
                  href="/account/preferences/"
                  className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                  role="menuitem"
                  onClick={() => setMenuOpen(false)}
                >
                  <Settings className="h-4 w-4 shrink-0 opacity-60" />
                  <span className="flex-1">Preferences</span>
                </Link>

                {/* ── Account switching ── */}
                {otherAccounts.length > 0 ? (
                  <div className="mt-1.5 border-t border-border pt-1.5">
                    <p className="px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-foreground/40">
                      Switch account
                    </p>
                    {otherAccounts.map((account) => (
                      <button
                        key={account.id}
                        type="button"
                        className="flex w-full items-center gap-3 rounded-xl px-3 py-2 text-left text-sm text-foreground/70 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                        onClick={() => {
                          switchAccount(account.id);
                          setMenuOpen(false);
                        }}
                      >
                        <ArrowRightLeft className="h-3.5 w-3.5 shrink-0 opacity-50" />
                        <span className="truncate">{getAccountLabel(account)}</span>
                      </button>
                    ))}
                  </div>
                ) : null}

                {accounts.length < 5 ? (
                  <Link
                    href={addAccountHref}
                    className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                    role="menuitem"
                    onClick={() => setMenuOpen(false)}
                  >
                    <UserPlus className="h-4 w-4 shrink-0 opacity-60" />
                    <span className="flex-1">Add account</span>
                  </Link>
                ) : null}

                {/* ── Admin panel ── */}
                {profileQuery.data?.is_superuser || profileQuery.data?.is_staff ? (
                  <>
                    <div className="my-1.5 border-t border-border" role="separator" />
                    <Link
                      href={adminPanelHref}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                      role="menuitem"
                      onClick={() => setMenuOpen(false)}
                    >
                      <Shield className="h-4 w-4 shrink-0 opacity-60" />
                      <span className="flex-1">Admin panel</span>
                      <ExternalLink className="h-3 w-3 shrink-0 text-foreground/30" />
                    </Link>
                  </>
                ) : null}

                {/* ── Logout ── */}
                <div className="my-1.5 border-t border-border" role="separator" />
                <button
                  className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-red-600/80 transition hover:bg-red-50 hover:text-red-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500/40 dark:text-red-400/80 dark:hover:bg-red-500/10 dark:hover:text-red-400"
                  onClick={() => {
                    setMenuOpen(false);
                    logout();
                  }}
                  role="menuitem"
                  type="button"
                >
                  <LogOut className="h-4 w-4 shrink-0" />
                  <span>Sign out</span>
                </button>
              </div>
            </div>
          ) : (
            <div
              className="absolute right-0 top-[calc(100%+0.5rem)] z-50 w-64 max-w-[calc(100vw-2rem)] origin-top-right animate-in fade-in zoom-in-95 rounded-2xl border border-border bg-card shadow-2xl"
              role="menu"
            >
              {/* ── Welcome header ── */}
              <div className="flex items-center gap-3 rounded-t-2xl bg-muted/40 px-4 py-3.5">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-foreground/5">
                  <UserRound className="h-5 w-5 text-foreground/40" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground">Welcome</p>
                  <p className="text-xs text-foreground/50">Sign in for personalized experience</p>
                </div>
              </div>

              <div className="p-1.5">
                <Link
                  href="/account/login/"
                  className="flex items-center justify-center gap-2 rounded-xl bg-primary px-3 py-2.5 text-sm font-semibold text-white transition hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                  role="menuitem"
                  onClick={() => setMenuOpen(false)}
                >
                  Sign in
                </Link>
                <Link
                  href="/account/register/"
                  className="mt-1 flex items-center justify-center gap-2 rounded-xl border border-border px-3 py-2.5 text-sm font-medium text-foreground transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                  role="menuitem"
                  onClick={() => setMenuOpen(false)}
                >
                  <UserPlus className="h-3.5 w-3.5" />
                  Create account
                </Link>

                <div className="my-2 border-t border-border" role="separator" />

                <Link
                  href="/faq/"
                  className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                  role="menuitem"
                  onClick={() => setMenuOpen(false)}
                >
                  <HelpCircle className="h-4 w-4 shrink-0 opacity-60" />
                  FAQ
                </Link>
                <Link
                  href="/contact/"
                  className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-foreground/80 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                  role="menuitem"
                  onClick={() => setMenuOpen(false)}
                >
                  <Mail className="h-4 w-4 shrink-0 opacity-60" />
                  Contact support
                </Link>
              </div>
            </div>
          )
        ) : null}
      </div>
      {open ? <CartDrawer isOpen={open} onClose={() => setOpen(false)} /> : null}
    </div>
  );
}
