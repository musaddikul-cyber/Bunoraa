/**
 * Router Utility
 * @module utils/router
 */

const Router = (function() {
    'use strict';

    const routes = new Map();
    let currentRoute = null;
    let beforeHooks = [];
    let afterHooks = [];

    function parseRoute(path) {
        const [pathname, search] = path.split('?');
        const params = new URLSearchParams(search || '');
        
        return {
            pathname: pathname.replace(/\/$/, '') || '/',
            query: Object.fromEntries(params),
            hash: window.location.hash.slice(1)
        };
    }

    function matchRoute(pathname) {
        for (const [pattern, handler] of routes) {
            const regex = new RegExp('^' + pattern.replace(/:[^/]+/g, '([^/]+)') + '$');
            const match = pathname.match(regex);
            
            if (match) {
                const paramNames = (pattern.match(/:[^/]+/g) || []).map(p => p.slice(1));
                const params = {};
                paramNames.forEach((name, i) => {
                    params[name] = match[i + 1];
                });
                
                return { handler, params };
            }
        }
        return null;
    }

    function register(pattern, handler) {
        routes.set(pattern, handler);
    }

    function before(hook) {
        beforeHooks.push(hook);
        return () => {
            beforeHooks = beforeHooks.filter(h => h !== hook);
        };
    }

    function after(hook) {
        afterHooks.push(hook);
        return () => {
            afterHooks = afterHooks.filter(h => h !== hook);
        };
    }

    async function navigate(path, options = {}) {
        const route = parseRoute(path);
        const matched = matchRoute(route.pathname);

        for (const hook of beforeHooks) {
            const result = await hook(route, currentRoute);
            if (result === false) return false;
            if (typeof result === 'string') {
                return navigate(result, options);
            }
        }

        if (!options.replace) {
            history.pushState({ path }, '', path);
        } else {
            history.replaceState({ path }, '', path);
        }

        currentRoute = route;

        if (matched) {
            await matched.handler({ ...route, params: matched.params });
        }

        for (const hook of afterHooks) {
            await hook(route);
        }

        return true;
    }

    function back() {
        history.back();
    }

    function forward() {
        history.forward();
    }

    function getQuery(key = null) {
        const params = new URLSearchParams(window.location.search);
        if (key) return params.get(key);
        return Object.fromEntries(params);
    }

    function setQuery(params, options = {}) {
        const current = new URLSearchParams(window.location.search);
        
        Object.entries(params).forEach(([key, value]) => {
            if (value === null || value === undefined || value === '') {
                current.delete(key);
            } else {
                current.set(key, value);
            }
        });

        const search = current.toString();
        const path = window.location.pathname + (search ? '?' + search : '');
        
        if (options.replace) {
            history.replaceState({}, '', path);
        } else {
            history.pushState({}, '', path);
        }

        window.dispatchEvent(new CustomEvent('querychange', { detail: Object.fromEntries(current) }));
    }

    function getCurrentRoute() {
        return currentRoute || parseRoute(window.location.pathname + window.location.search);
    }

    function init() {
        window.addEventListener('popstate', (e) => {
            const path = e.state?.path || window.location.pathname + window.location.search;
            const route = parseRoute(path);
            const matched = matchRoute(route.pathname);

            currentRoute = route;

            if (matched) {
                matched.handler({ ...route, params: matched.params });
            }

            afterHooks.forEach(hook => hook(route));
        });
    }

    return {
        register,
        navigate,
        back,
        forward,
        before,
        after,
        getQuery,
        setQuery,
        getCurrentRoute,
        init
    };
})();

window.Router = Router;
