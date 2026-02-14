import Link from "next/link";
import { apiFetch } from "@/lib/api";
import type { MenuPage } from "@/lib/types";
import { HeaderClient } from "@/components/layout/HeaderClient";
import { SearchBar } from "@/components/search/SearchBar";
import { MobileNav } from "@/components/layout/MobileNav";
import { MobileHeaderVisibility } from "@/components/layout/MobileHeaderVisibility";
import { asArray } from "@/lib/array";

type Category = { id: string; name: string; slug: string };

async function getMenuPages() {
  try {
    const response = await apiFetch<MenuPage[]>("/pages/menu/", {
      next: { revalidate: 600 },
    });
    return asArray<MenuPage>(response.data);
  } catch {
    return [];
  }
}

async function getTopCategories() {
  try {
    const response = await apiFetch<Category[]>("/catalog/categories/", {
      params: { page_size: 8 },
      next: { revalidate: 600 },
    });
    return asArray<Category>(response.data);
  } catch {
    return [];
  }
}

export async function Header() {
  const [menuResult, categoryResult] = await Promise.allSettled([
    getMenuPages(),
    getTopCategories(),
  ]);
  const menuPages = menuResult.status === "fulfilled" ? menuResult.value : [];
  const categories = categoryResult.status === "fulfilled" ? categoryResult.value : [];

  return (
    <MobileHeaderVisibility>
      <header className="border-b border-border/80 bg-background/95 backdrop-blur-md supports-[backdrop-filter]:bg-background/88">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-6">
            <MobileNav categories={categories} menuPages={menuPages} />
            <Link href="/" className="text-xl font-bold sm:text-2xl lg:text-3xl">
              Bunoraa
            </Link>
            <nav className="hidden items-center gap-4 text-sm lg:flex">
              <div className="w-48">
                <SearchBar />
              </div>
              <Link
                className="group relative inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1.5 text-sm font-semibold text-primary transition duration-200 hover:-translate-y-0.5 hover:border-primary/50 hover:bg-primary hover:text-white hover:shadow-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                href="/preorders/"
              >
                Preorders
                <span className="hidden text-[10px] font-semibold uppercase tracking-[0.2em] text-primary/70 group-hover:text-white/80 xl:inline">
                  New
                </span>
              </Link>
              {categories.slice(0, 4).map((category) => (
                <Link
                  key={category.id}
                  className="text-foreground/70 hover:text-foreground"
                  href={`/categories/${category.slug}/`}
                >
                  {category.name}
                </Link>
              ))}
              {menuPages.slice(0, 3).map((page) => (
                <Link
                  key={page.id}
                  className="text-foreground/70 hover:text-foreground"
                  href={`/pages/${page.slug}/`}
                >
                  {page.title}
                </Link>
              ))}
            </nav>
          </div>
          <HeaderClient />
        </div>
      </header>
    </MobileHeaderVisibility>
  );
}
