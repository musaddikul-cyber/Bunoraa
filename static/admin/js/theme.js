'use strict';
{
    const THEME_STORAGE_KEY = "bunoraa-theme";
    const ADMIN_THEME_STORAGE_KEY = "admin-theme";
    const LEGACY_THEME_STORAGE_KEY = "theme";
    const THEMES = ["system", "light", "dark", "moonlight", "gray", "modern"];
    const DARK_THEMES = new Set(["dark", "moonlight"]);
    const LIGHT_THEMES = new Set(["light", "gray", "modern"]);

    function normalizeTheme(theme) {
        return THEMES.includes(theme) ? theme : "system";
    }

    function mapToDjangoTheme(theme) {
        if (theme === "system") return "auto";
        if (DARK_THEMES.has(theme)) return "dark";
        if (LIGHT_THEMES.has(theme)) return "light";
        return "auto";
    }

    function applyRootTheme(theme, djangoTheme) {
        const root = document.documentElement;
        const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;

        root.classList.remove(...THEMES, "dark");
        root.classList.add(theme);

        if (theme === "dark" || (theme === "system" && prefersDark)) {
            root.classList.add("dark");
        }

        root.dataset.adminTheme = theme;
        root.dataset.theme = djangoTheme;
        root.style.colorScheme = theme === "system" ? "light dark" : (DARK_THEMES.has(theme) ? "dark" : "light");
    }

    function setTheme(theme, persist = true) {
        theme = normalizeTheme(theme);
        const djangoTheme = mapToDjangoTheme(theme);

        applyRootTheme(theme, djangoTheme);

        if (!persist) {
            return theme;
        }

        try {
            localStorage.setItem(THEME_STORAGE_KEY, theme);
            localStorage.setItem(ADMIN_THEME_STORAGE_KEY, theme);
            localStorage.setItem(LEGACY_THEME_STORAGE_KEY, djangoTheme);
        } catch (error) {
            // Ignore storage failures and keep the theme applied for this view.
        }
        return theme;
    }

    function getStoredTheme() {
        try {
            const storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
            if (storedTheme) {
                return storedTheme;
            }

            const storedAdminTheme = localStorage.getItem(ADMIN_THEME_STORAGE_KEY);
            if (storedAdminTheme) {
                return storedAdminTheme;
            }

            const legacy = localStorage.getItem(LEGACY_THEME_STORAGE_KEY);
            if (legacy === "dark") return "dark";
            if (legacy === "light") return "light";
        } catch (error) {
            return "system";
        }

        return "system";
    }

    function cycleTheme() {
        const current = normalizeTheme(
            getStoredTheme() ||
            document.documentElement.dataset.adminTheme ||
            "system"
        );
        const next = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
        setTheme(next);
    }

    function initTheme() {
        return setTheme(getStoredTheme());
    }

    window.addEventListener('load', function() {
        const buttons = document.getElementsByClassName("theme-toggle");
        Array.from(buttons).forEach((btn) => {
            btn.addEventListener("click", cycleTheme);
        });
    });

    window.addEventListener("storage", function(event) {
        if (![THEME_STORAGE_KEY, ADMIN_THEME_STORAGE_KEY, LEGACY_THEME_STORAGE_KEY].includes(event.key)) {
            return;
        }
        setTheme(getStoredTheme(), false);
    });

    if (window.matchMedia) {
        const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
        const syncSystemTheme = function() {
            if (normalizeTheme(getStoredTheme()) === "system") {
                setTheme("system", false);
            }
        };

        if (typeof mediaQuery.addEventListener === "function") {
            mediaQuery.addEventListener("change", syncSystemTheme);
        } else if (typeof mediaQuery.addListener === "function") {
            mediaQuery.addListener(syncSystemTheme);
        }
    }

    initTheme();
}
