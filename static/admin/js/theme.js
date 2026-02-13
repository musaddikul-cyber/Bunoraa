'use strict';
{
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

    function setTheme(theme) {
        theme = normalizeTheme(theme);
        const djangoTheme = mapToDjangoTheme(theme);

        document.documentElement.dataset.adminTheme = theme;
        document.documentElement.dataset.theme = djangoTheme;

        localStorage.setItem("admin-theme", theme);
        localStorage.setItem("theme", djangoTheme);
    }

    function cycleTheme() {
        const current = normalizeTheme(
            localStorage.getItem("admin-theme") ||
            document.documentElement.dataset.adminTheme ||
            "system"
        );
        const next = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
        setTheme(next);
    }

    function initTheme() {
        const storedAdminTheme = localStorage.getItem("admin-theme");
        if (storedAdminTheme) {
            setTheme(storedAdminTheme);
            return;
        }

        const legacy = localStorage.getItem("theme");
        if (legacy === "dark") return setTheme("dark");
        if (legacy === "light") return setTheme("light");
        return setTheme("system");
    }

    window.addEventListener('load', function() {
        const buttons = document.getElementsByClassName("theme-toggle");
        Array.from(buttons).forEach((btn) => {
            btn.addEventListener("click", cycleTheme);
        });
    });

    initTheme();
}
