"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { SearchBar } from "@/components/search/SearchBar";
import { useAuthContext } from "@/components/providers/AuthProvider";
import type { MenuPage } from "@/lib/types";
import { cn } from "@/lib/utils";
import { buildCategoryPath } from "@/lib/categoryPaths";
import {
  X,
  ChevronRight,
  ChevronDown,
  ShoppingBag,
  Heart,
  User,
  LogIn,
  UserPlus,
  Package,
  LayoutGrid,
  Layers,
  Tags,
  FileText,
  HelpCircle,
  Mail,
  ArrowRightLeft,
  UserPlus2,
  ClipboardList,
  Sparkles,
} from "lucide-react";

type Category = { id: string; name: string; slug: string };

export function MobileNav({
  categories,
  menuPages,
  hasBundles,
}: {
  categories: Category[];
  menuPages: MenuPage[];
  hasBundles: boolean;
}) {
  const pathname = usePathname();
  const { hasToken, profileQuery, accounts, activeAccountId, switchAccount } = useAuthContext();
  const [open, setOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement | null>(null);
  const closeButtonRef = React.useRef<HTMLButtonElement | null>(null);
  const wasOpenRef = React.useRef(false);

  const accountLabel =
    profileQuery.data?.full_name ||
    profileQuery.data?.first_name ||
    profileQuery.data?.email ||
    "";
  const otherAccounts = React.useMemo(
    () => accounts.filter((account) => account.id !== activeAccountId),
    [accounts, activeAccountId]
  );

  const normalizePath = React.useCallback((value: string) => {
    if (value.length > 1 && value.endsWith("/")) {
      return value.slice(0, -1);
    }
    return value;
  }, []);

  const isActiveLink = React.useCallback(
    (href: string) => {
      const current = normalizePath(pathname || "/");
      const target = normalizePath(href);
      if (target === "/") return current === "/";
      return current === target || current.startsWith(`${target}/`);
    },
    [pathname, normalizePath]
  );

  const navLinkClass = React.useCallback(
    (href: string) =>
      cn(
        "block rounded-xl border px-3 py-2.5 text-sm transition-colors",
        "border-transparent text-foreground/90 hover:border-border hover:bg-muted hover:text-foreground",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40",
        isActiveLink(href) && "border-primary/25 bg-primary/10 text-primary"
      ),
    [isActiveLink]
  );

  const accountItemClass =
    "block w-full rounded-xl border border-transparent px-3 py-2.5 text-left text-sm text-foreground/90 transition-colors hover:border-border hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40";

  const closeNav = React.useCallback(() => {
    setOpen(false);
  }, []);

  React.useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeNav();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
    };
  }, [open, closeNav]);

  React.useEffect(() => {
    if (!open) return;
    const originalOverflow = document.body.style.overflow;
    const originalTouchAction = document.body.style.touchAction;
    document.body.style.overflow = "hidden";
    document.body.style.touchAction = "none";
    return () => {
      document.body.style.overflow = originalOverflow;
      document.body.style.touchAction = originalTouchAction;
    };
  }, [open]);

  React.useEffect(() => {
    if (!open) return;
    closeButtonRef.current?.focus();
  }, [open]);

  React.useEffect(() => {
    if (open) {
      wasOpenRef.current = true;
      return;
    }
    if (!wasOpenRef.current) return;
    triggerRef.current?.focus();
    wasOpenRef.current = false;
  }, [open]);

  React.useEffect(() => {
    setOpen(false);
  }, [pathname]);

  /* ── Expandable sections state ── */
  const [expandedSections, setExpandedSections] = React.useState<Record<string, boolean>>({
    shop: true,
  });

  const toggleSection = (key: string) =>
    setExpandedSections((prev) => ({ ...prev, [key]: !prev[key] }));

  const isSectionOpen = (key: string) => Boolean(expandedSections[key]);

  /* ── Animated slide state ── */
  const [slideIn, setSlideIn] = React.useState(false);

  React.useEffect(() => {
    if (open) {
      requestAnimationFrame(() => setSlideIn(true));
    } else {
      setSlideIn(false);
    }
  }, [open]);

  /* ── User avatar ── */
  const initials = React.useMemo(() => {
    const name = profileQuery.data?.full_name || profileQuery.data?.first_name || "";
    if (!name) return "";
    const parts = name.trim().split(/\s+/);
    return parts.length >= 2
      ? (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
      : parts[0].slice(0, 2).toUpperCase();
  }, [profileQuery.data]);

  /* ── Section header component ── */
  const SectionHeader = ({
    sectionKey,
    icon: Icon,
    label,
  }: {
    sectionKey: string;
    icon: React.ElementType;
    label: string;
  }) => (
    <button
      type="button"
      className="flex w-full items-center justify-between rounded-lg px-1 py-1.5 text-left text-xs font-semibold uppercase tracking-[0.2em] text-foreground/50 transition hover:text-foreground/70"
      onClick={() => toggleSection(sectionKey)}
      aria-expanded={isSectionOpen(sectionKey)}
    >
      <span className="inline-flex items-center gap-2">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </span>
      {isSectionOpen(sectionKey) ? (
        <ChevronDown className="h-3.5 w-3.5" />
      ) : (
        <ChevronRight className="h-3.5 w-3.5" />
      )}
    </button>
  );

  /* ── Nav link with icon ── */
  const NavLink = ({
    href,
    icon: Icon,
    label,
    badge,
    highlight,
  }: {
    href: string;
    icon: React.ElementType;
    label: string;
    badge?: string;
    highlight?: boolean;
  }) => (
    <Link
      className={cn(
        "flex items-center gap-3 rounded-xl border px-3 py-2.5 text-sm transition-colors",
        "border-transparent text-foreground/80 hover:border-border hover:bg-muted hover:text-foreground",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40",
        isActiveLink(href) && "border-primary/25 bg-primary/10 text-primary",
        highlight &&
          !isActiveLink(href) &&
          "border-primary/30 bg-primary/5 text-primary font-medium"
      )}
      href={href}
      onClick={closeNav}
    >
      <Icon className="h-4 w-4 shrink-0 opacity-60" />
      <span className="flex-1">{label}</span>
      {badge ? (
        <span className="rounded-full bg-primary/15 px-2 py-0.5 text-[10px] font-semibold text-primary">
          {badge}
        </span>
      ) : (
        <ChevronRight className="h-3.5 w-3.5 shrink-0 text-foreground/25" />
      )}
    </Link>
  );

  return (
    <div className="lg:hidden">
      <button
        ref={triggerRef}
        type="button"
        className="inline-flex items-center justify-center rounded-full border border-border bg-card p-2 text-sm text-foreground shadow-soft transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls="mobile-navigation-panel"
        aria-label="Open menu"
      >
        <span className="flex flex-col gap-1" aria-hidden="true">
          <span className="h-0.5 w-5 rounded-full bg-foreground/90" />
          <span className="h-0.5 w-5 rounded-full bg-foreground/90" />
          <span className="h-0.5 w-5 rounded-full bg-foreground/90" />
        </span>
      </button>

      {open ? (
        <div
          className="fixed inset-0 z-50 h-[100svh] supports-[height:100dvh]:h-[100dvh]"
          role="dialog"
          aria-modal="true"
          aria-labelledby="mobile-navigation-title"
          onClick={(event) => {
            if (event.target === event.currentTarget) closeNav();
          }}
        >
          {/* Backdrop */}
          <div
            className={cn(
              "absolute inset-0 bg-foreground/35 backdrop-blur-sm transition-opacity duration-300",
              slideIn ? "opacity-100" : "opacity-0"
            )}
            onClick={closeNav}
          />

          {/* Panel */}
          <aside
            id="mobile-navigation-panel"
            className={cn(
              "absolute inset-y-0 left-0 flex h-[100svh] min-h-[100svh] w-full max-w-[22rem] flex-col border-r border-border bg-background text-foreground shadow-2xl transition-transform duration-300 ease-out supports-[height:100dvh]:h-[100dvh] supports-[height:100dvh]:min-h-[100dvh]",
              slideIn ? "translate-x-0" : "-translate-x-full"
            )}
            onClick={(event) => event.stopPropagation()}
          >
            {/* ── Header ── */}
            <div className="flex items-center justify-between border-b border-border px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-3">
              <p id="mobile-navigation-title" className="text-lg font-semibold tracking-tight">
                Menu
              </p>
              <button
                ref={closeButtonRef}
                type="button"
                className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-border text-foreground/60 transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                onClick={closeNav}
                aria-label="Close menu"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* ── User banner ── */}
            <div className="border-b border-border px-5 py-3">
              {hasToken ? (
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary/15 text-sm font-bold text-primary">
                    {initials || <User className="h-4 w-4" />}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-semibold text-foreground">{accountLabel}</p>
                    {profileQuery.data?.email ? (
                      <p className="truncate text-xs text-foreground/50">{profileQuery.data.email}</p>
                    ) : null}
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Link
                    href="/account/login/"
                    onClick={closeNav}
                    className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-white transition hover:bg-primary/90"
                  >
                    <LogIn className="h-3.5 w-3.5" />
                    Sign in
                  </Link>
                  <Link
                    href="/account/register/"
                    onClick={closeNav}
                    className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-sm font-medium text-foreground transition hover:bg-muted"
                  >
                    <UserPlus className="h-3.5 w-3.5" />
                    Register
                  </Link>
                </div>
              )}
            </div>

            {/* ── Search ── */}
            <div className="shrink-0 px-5 py-3">
              <SearchBar />
            </div>

            {/* ── Navigation ── */}
            <nav className="min-h-0 flex-1 space-y-1 overflow-y-auto px-4 pb-[max(1.5rem,env(safe-area-inset-bottom))] text-sm scrollbar-thin">
              {/* ── Shop section ── */}
              <SectionHeader sectionKey="shop" icon={ShoppingBag} label="Shop" />
              {isSectionOpen("shop") ? (
                <div className="space-y-1 pb-2">
                  <NavLink href="/products/" icon={Package} label="All Products" />
                  <NavLink href="/collections/" icon={Layers} label="Collections" />
                  {hasBundles ? <NavLink href="/bundles/" icon={LayoutGrid} label="Bundles" /> : null}
                  <NavLink href="/preorders/" icon={Sparkles} label="Preorders" highlight />
                  <NavLink href="/categories/" icon={Tags} label="All Categories" />

                  {/* Expandable categories */}
                  {categories.length > 0 ? (
                    <>
                      <button
                        type="button"
                        className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-xs font-medium text-foreground/50 transition hover:text-foreground/70"
                        onClick={() => toggleSection("categories")}
                      >
                        <span>Browse by category</span>
                        {isSectionOpen("categories") ? (
                          <ChevronDown className="h-3 w-3" />
                        ) : (
                          <ChevronRight className="h-3 w-3" />
                        )}
                      </button>
                      {isSectionOpen("categories") ? (
                        <div className="ml-2 space-y-0.5 border-l-2 border-border pl-3">
                          {categories.slice(0, 12).map((category) => (
                            <Link
                              key={category.id}
                              className={cn(
                                "block rounded-lg px-2.5 py-2 text-sm text-foreground/70 transition hover:bg-muted hover:text-foreground",
                                isActiveLink(buildCategoryPath(category.slug)) &&
                                  "bg-primary/10 text-primary font-medium"
                              )}
                              href={buildCategoryPath(category.slug)}
                              onClick={closeNav}
                            >
                              {category.name}
                            </Link>
                          ))}
                          {categories.length > 12 ? (
                            <Link
                              className="block rounded-lg px-2.5 py-2 text-xs font-medium text-primary transition hover:bg-primary/5"
                              href="/categories/"
                              onClick={closeNav}
                            >
                              View all {categories.length} categories →
                            </Link>
                          ) : null}
                        </div>
                      ) : null}
                    </>
                  ) : null}
                </div>
              ) : null}

              <div className="my-1 border-t border-border" />

              {/* ── Account section ── */}
              {hasToken ? (
                <>
                  <SectionHeader sectionKey="account" icon={User} label="Account" />
                  {isSectionOpen("account") ? (
                    <div className="space-y-1 pb-2">
                      <NavLink href="/account/profile/" icon={User} label="Profile" />
                      <NavLink href="/account/orders/" icon={ClipboardList} label="Orders" />
                      <NavLink href="/wishlist/" icon={Heart} label="Wishlist" />
                      <NavLink href="/cart/" icon={ShoppingBag} label="Bag" />

                      {/* Switch account */}
                      {otherAccounts.length > 0 ? (
                        <div className="mt-1 space-y-0.5 rounded-xl border border-dashed border-border p-2">
                          <p className="px-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-foreground/40">
                            Switch account
                          </p>
                          {otherAccounts.map((account) => (
                            <button
                              key={account.id}
                              type="button"
                              className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm text-foreground/70 transition hover:bg-muted hover:text-foreground"
                              onClick={() => {
                                switchAccount(account.id);
                                closeNav();
                              }}
                            >
                              <ArrowRightLeft className="h-3.5 w-3.5 shrink-0 opacity-50" />
                              <span className="truncate">
                                {account.email ||
                                  account.full_name ||
                                  account.first_name ||
                                  `Account ${account.id.slice(0, 8)}`}
                              </span>
                            </button>
                          ))}
                        </div>
                      ) : null}

                      {accounts.length < 5 ? (
                        <NavLink
                          href={`/account/login/?next=${encodeURIComponent(pathname || "/account/profile/")}&add_account=1`}
                          icon={UserPlus2}
                          label="Add account"
                        />
                      ) : null}
                    </div>
                  ) : null}
                </>
              ) : (
                <div className="space-y-1 pb-2">
                  <NavLink href="/wishlist/" icon={Heart} label="Wishlist" />
                  <NavLink href="/cart/" icon={ShoppingBag} label="Bag" />
                </div>
              )}

              {/* ── Pages section ── */}
              {menuPages.length > 0 ? (
                <>
                  <div className="my-1 border-t border-border" />
                  <SectionHeader sectionKey="pages" icon={FileText} label="Pages" />
                  {isSectionOpen("pages") ? (
                    <div className="space-y-1 pb-2">
                      {menuPages.slice(0, 8).map((page) => (
                        <NavLink
                          key={page.id}
                          href={`/pages/${page.slug}/`}
                          icon={FileText}
                          label={page.title}
                        />
                      ))}
                    </div>
                  ) : null}
                </>
              ) : null}

              {/* ── Support section ── */}
              <div className="my-1 border-t border-border" />
              <SectionHeader sectionKey="support" icon={HelpCircle} label="Support" />
              {isSectionOpen("support") ? (
                <div className="space-y-1 pb-4">
                  <NavLink href="/contact/" icon={Mail} label="Contact" />
                  <NavLink href="/faq/" icon={HelpCircle} label="FAQ" />
                </div>
              ) : null}
            </nav>
          </aside>
        </div>
      ) : null}
    </div>
  );
}
