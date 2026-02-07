import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { MenuPage } from "@/lib/types";
import { ThemeSwitcher } from "@/components/theme/ThemeProvider";
import { LocaleSwitcher } from "@/components/locale/LocaleSwitcher";

async function getFooterPages() {
  try {
    const response = await apiFetch<MenuPage[]>("/pages/pages/footer/", {
      next: { revalidate: 600 },
    });
    return response.data;
  } catch {
    return [];
  }
}

export async function Footer() {
  const pages = await getFooterPages();

  return (
    <footer className="border-t border-border bg-card">
      <div className="mx-auto w-full max-w-7xl px-6 py-10">
        <div className="grid gap-6 md:grid-cols-4">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Bunoraa
            </p>
            <p className="mt-2 text-sm text-foreground/70">
              Discover handcrafted fashion, home, and lifestyle essentials curated by Bunoraa artisans.
            </p>
          </div>
          <div>
            <p className="text-sm font-semibold">Explore</p>
            <ul className="mt-3 space-y-2 text-sm text-foreground/70">
              <li>
                <Link href="/products/">Products</Link>
              </li>
              <li>
                <Link href="/categories/">Categories</Link>
              </li>
              <li>
                <Link href="/collections/">Collections</Link>
              </li>
              <li>
                <Link href="/bundles/">Bundles</Link>
              </li>
            </ul>
          </div>
          <div>
            <p className="text-sm font-semibold">Pages</p>
            <ul className="mt-3 space-y-2 text-sm text-foreground/70">
              {pages.map((page) => (
                <li key={page.id}>
                  <Link href={`/pages/${page.slug}/`}>{page.title}</Link>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <p className="text-sm font-semibold">Support</p>
            <ul className="mt-3 space-y-2 text-sm text-foreground/70">
              <li>
                <Link href="/contact/">Contact</Link>
              </li>
              <li>
                <Link href="/faq/">FAQ</Link>
              </li>
              <li>
                <Link href="/pages/shipping/">Shipping</Link>
              </li>
              <li>
                <Link href="/pages/returns/">Returns</Link>
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-10 flex flex-col gap-4 border-t border-border pt-6 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-foreground/60">
            © {new Date().getFullYear()} Bunoraa. All rights reserved.
          </p>
          <div className="flex w-full flex-col gap-3 text-sm text-foreground/70 sm:w-auto sm:flex-row sm:items-center sm:justify-end">
            <ThemeSwitcher />
            <LocaleSwitcher />
          </div>
        </div>
      </div>
    </footer>
  );
}
