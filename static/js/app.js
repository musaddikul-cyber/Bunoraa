/**
 * Bunoraa - Main Application Entry Point
 * @module app
 */

import { initLazyHydration } from './utils/lazyHydrate.js';

const App = (function() {
    'use strict';

    // Page controllers will be loaded on-demand via dynamic import for smaller initial bundle
    const pageControllers = {};

    let currentPage = null;
    let currentController = null;

    async function loadPageController(page) {
        try {
            // Dynamic import with error handling and caching
            const mod = await import(/* webpackChunkName: "page-[request]" */ `./pages/${page}.js`);
            return mod.default || mod;
        } catch (e) {
            console.warn(`Page controller for ${page} not found:`, e);
            return null;
        }
    }

    function init() {
        detectCurrentPage();
        initGlobalComponents();
        initCurrentPage();
        initCartBadge();
        initWishlistBadge();
        initGlobalEventListeners();
        initMobileMenu();
        initLanguageSelector();
        // Currency selector disabled in single-currency mode

        // Mitigate unexpected initial jumps caused by JS focusing or layout shifts.
        // If the navigation is a fresh navigate (not back/forward) and there's no URL hash,
        // and the page is already scrolled near the bottom on initial load, reset to top.
        try {
            const navEntries = performance.getEntriesByType ? performance.getEntriesByType('navigation') : [];
            const nav = (navEntries && navEntries[0]) || null;
            if ((nav && nav.type === 'navigate') && !window.location.hash) {
                setTimeout(() => {
                    const doc = document.scrollingElement || document.documentElement;
                    if (!doc) return;
                    const scrolled = doc.scrollTop || window.pageYOffset || 0;
                    const maxScroll = Math.max(0, doc.scrollHeight - window.innerHeight);
                    // If already scrolled more than half the document height or very close to bottom, reset
                    if (scrolled > Math.max(100, maxScroll * 0.6)) {
                        window.scrollTo({ top: 0, behavior: 'auto' });
                    }
                }, 60);
            }
        } catch (e) {
            // ignore
        }
    }
    async function initWishlistBadge() {
        // Update wishlist badge on page load; sync wishlist button states. Fallback to cached value if API fails
        try {
            // If user is not authenticated, rely on cached/local storage value and clear active states
            if (!AuthApi.isAuthenticated()) {
                const raw = localStorage.getItem('wishlist');
                if (raw) {
                    const parsed = JSON.parse(raw);
                    WishlistApi.updateBadge(parsed);
                    const items = parsed.items || (parsed.data && parsed.data.items) || [];
                    syncWishlistButtons(items);
                } else {
                    // Clear any filled buttons for anonymous users
                    syncWishlistButtons([]);
                }
                return;
            }

            const response = await WishlistApi.getWishlist({ pageSize: 200 });
            const payload = response.data || {};
            const items = payload.items || payload.data || [];
            WishlistApi.updateBadge(payload);
            syncWishlistButtons(items);
        } catch (error) {
            try {
                const raw = localStorage.getItem('wishlist');
                if (raw) {
                    const parsed = JSON.parse(raw);
                    WishlistApi.updateBadge(parsed);
                    const items = parsed.items || (parsed.data && parsed.data.items) || [];
                    syncWishlistButtons(items);
                }
            } catch (e) {
                // ignore
            }
        }
    }

    // Keep last loaded wishlist items for re-sync (used by mutation observer)
    let _lastWishlistItems = [];

    // Set wishlist button states (filled/outlined) based on wishlist items array
    function syncWishlistButtons(items) {
        try {
            _lastWishlistItems = items || [];

            const mapById = {};
            const mapBySlug = {};

            (items || []).forEach(it => {
                // Support multiple shapes: item.product (uuid/string), item.product_id, item.product.slug or item.product_slug
                const pid = (it.product || it.product_id || (it.product && it.product.id) || null);
                const pslug = it.product_slug || (it.product && it.product.slug) || null;
                const id = it.id || it.pk || it.uuid || it.item || null;
                if (pid) mapById[String(pid)] = id || true;
                if (pslug) mapBySlug[String(pslug)] = id || true;
            });

            document.querySelectorAll('.wishlist-btn').forEach(btn => { 
                try {
                    const svg = btn.querySelector('svg');
                    const fillPath = svg?.querySelector('.heart-fill');
                    const productId = btn.dataset.productId || btn.closest('[data-product-id]')?.dataset.productId;
                    const productSlug = btn.dataset.productSlug || btn.closest('[data-product-slug]')?.dataset.productSlug;

                    let itemId = null;
                    if (productId && mapById.hasOwnProperty(String(productId))) itemId = mapById[String(productId)];
                    else if (productSlug && mapBySlug.hasOwnProperty(String(productSlug))) itemId = mapBySlug[String(productSlug)];

                    if (itemId) {
                        btn.dataset.wishlistItemId = itemId;
                        btn.classList.add('text-red-500');
                        btn.setAttribute('aria-pressed', 'true');
                        svg?.classList.add('fill-current');
                        if (fillPath) fillPath.style.opacity = '1';
                    } else {
                        btn.removeAttribute('data-wishlist-item-id');
                        btn.classList.remove('text-red-500');
                        btn.setAttribute('aria-pressed', 'false');
                        svg?.classList.remove('fill-current');
                        if (fillPath) fillPath.style.opacity = '0';
                    }
                } catch (e) {
                    // ignore per-button errors
                }
            });
        } catch (e) {
            // error logging removed
        }
    }

    // Observe DOM additions and re-sync wishlist buttons when new product cards appear
    (function() {
        if (typeof MutationObserver === 'undefined') return;
        let debounceTimer = null;
        const observer = new MutationObserver(function(mutations) {
            let found = false;
            for (const m of mutations) {
                if (m.addedNodes && m.addedNodes.length) {
                    for (const n of m.addedNodes) {
                        if (n.nodeType === 1 && (n.matches?.('.product-card') || n.querySelector?.('.product-card') || n.querySelector?.('.wishlist-btn'))) {
                            found = true; break;
                        }
                    }
                }
                if (found) break;
            }
            if (found) {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => {
                    try { syncWishlistButtons(_lastWishlistItems); } catch (e) {}
                }, 150);
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    })();

    async function initCartBadge() {
        const cartBadges = document.querySelectorAll('[data-cart-count]');
        if (!cartBadges.length) return;
        try {
            const response = await CartApi.getCart();
            const count = response.data?.item_count || 0;
            updateCartBadge(count);
        } catch (error) {
            // Fallback to cached cart in localStorage to show counts immediately
            try {
                const raw = localStorage.getItem('cart');
                if (raw) {
                    const parsed = JSON.parse(raw);
                    const count = parsed?.item_count || parsed?.items?.length || 0;
                    updateCartBadge(count);
                }
            } catch (e) {
                // error logging removed
            }
        }
    }

    function detectCurrentPage() {
        const path = window.location.pathname;
        const body = document.body;
        
        // Check for data-page attribute first
        if (body.dataset.page) {
            currentPage = body.dataset.page;
            return;
        }

        // Avoid initializing account controller for general /account(s)/* routes (login, register, etc.)
        // Only profile pages (e.g. /account/profile or /accounts/profile) should be considered account pages
        if ((path.startsWith('/accounts/') || path.startsWith('/account/')) && !(path.startsWith('/accounts/profile') || path.startsWith('/account/profile'))) {
            currentPage = null;
            return;
        }

        // Detect from URL
        if (path === '/' || path === '/home/') {
            currentPage = 'home';
        } else if (path === '/categories/' || path === '/products/') {
            currentPage = 'search';
        } else if (path.startsWith('/categories/') && path !== '/categories/') {
            currentPage = 'category';
        } else if (path.startsWith('/products/') && path !== '/products/') {
            currentPage = 'product';
        } else if (path === '/search/' || path.startsWith('/search')) {
            currentPage = 'search';
        } else if (path.startsWith('/cart')) {
            currentPage = 'cart';
        } else if (path.startsWith('/checkout')) {
            currentPage = 'checkout';
        } else if (path === '/account' || path.startsWith('/account/') || path.startsWith('/accounts/profile')) {
            currentPage = 'account';
        } else if (path.startsWith('/orders')) {
            currentPage = 'orders';
        } else if (path.startsWith('/wishlist')) {
            currentPage = 'wishlist';
        } else if (path.startsWith('/contact')) {
            currentPage = 'contact';
        }
    }

    function initGlobalComponents() {
        // Initialize global UI components
        // Modal, Toast, and Dropdown are lazy-initialized when needed
        
        // Initialize tabs if present on page
        if (typeof Tabs !== 'undefined' && document.querySelector('[data-tabs]')) {
            Tabs.init();
        }
        
        // Initialize any existing dropdowns on the page
        if (typeof Dropdown !== 'undefined') {
            document.querySelectorAll('[data-dropdown-trigger]').forEach(trigger => {
                const targetId = trigger.dataset.dropdownTarget;
                const target = document.getElementById(targetId);
                if (target) {
                    Dropdown.create(trigger, { content: target.innerHTML });
                }
            });
        }

        // Initialize lazy hydration for components marked with `data-hydrate`
        try { initLazyHydration(); } catch (e) { /* ignore */ }
    }

    async function initCurrentPage() {
        if (!currentPage) return;
        // Destroy previous controller if it exposes destroy()
        try {
            if (currentController && typeof currentController.destroy === 'function') {
                currentController.destroy();
            }
        } catch (e) { /* ignore */ }

        const controller = await loadPageController(currentPage);
        if (controller && typeof controller.init === 'function') {
            currentController = controller;
            try { await currentController.init(); } catch (e) { console.error('failed to init page controller', e); }
        }
    }

    // Register service worker (if supported)
    try {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/static/js/sw.js').catch(() => {});
        }
    } catch (e) { /* ignore */ }

    async function initCartBadge() {
        const cartBadges = document.querySelectorAll('[data-cart-count]');
        if (!cartBadges.length) return;
        try {
            const response = await CartApi.getCart();
            const count = response.data?.item_count || 0;
            // Persist latest cart count for offline/failure fallback
            try { localStorage.setItem('cart', JSON.stringify({ item_count: count, savedAt: Date.now() })); } catch(e) {}
            updateCartBadge(count);
        } catch (error) {
            // Fallback to cached cart in localStorage to show counts immediately
            try {
                const raw = localStorage.getItem('cart');
                if (raw) {
                    const parsed = JSON.parse(raw);
                    const count = parsed?.item_count || 0;
                    updateCartBadge(count);
                    return;
                }
            } catch (e) {
                console.error('Failed to get cart count fallback:', e);
            }
            // error logging removed
        }
    }

    async function findWishlistItemId(productId) {
        try {
            const response = await WishlistApi.getWishlist({ pageSize: 200 });
            const payload = response.data || {};
            const items = payload.items || [];
            const match = items.find(item => String(item.product) === String(productId));
            return match?.id || null;
        } catch (error) {
            return null;
        }
    }

    function updateCartBadge(count) {
        const cartBadges = document.querySelectorAll('[data-cart-count]');
        cartBadges.forEach(badge => {
            badge.textContent = count > 99 ? '99+' : count;
            badge.classList.toggle('hidden', count === 0);
        });
    }

    function initGlobalEventListeners() {
        // Listen for cart updates
        document.addEventListener('cart:updated', async () => {
            await initCartBadge();
        });

        // Listen for wishlist updates
        document.addEventListener('wishlist:updated', async () => {
            await initWishlistBadge();
        });

        // Listen for auth state changes
        document.addEventListener('auth:login', () => {
            updateAuthUI(true);
        });

        document.addEventListener('auth:logout', () => {
            updateAuthUI(false);
        });

        // Initialize wishlist icon fill state for existing buttons
        document.querySelectorAll('.wishlist-btn').forEach(btn => {
            try {
                const svg = btn.querySelector('svg');
                const fillPath = svg?.querySelector('.heart-fill');
                if (btn.classList.contains('text-red-500')) {
                    svg?.classList.add('fill-current');
                    if (fillPath) fillPath.style.opacity = '1';
                } else {
                    if (fillPath) fillPath.style.opacity = '0';
                }
            } catch (e) {
                // ignore
            }
        });

        // Global quick add to cart (data-attribute and class-based)
        document.addEventListener('click', async (e) => {
            const quickAddBtn = e.target.closest('[data-quick-add], [data-add-to-cart], .add-to-cart-btn');
            if (quickAddBtn) {
                e.preventDefault();
                const productId = quickAddBtn.dataset.productId || quickAddBtn.dataset.quickAdd || quickAddBtn.dataset.addToCart;
                if (!productId) return;
                
                quickAddBtn.disabled = true;
                const originalHtml = quickAddBtn.innerHTML;
                quickAddBtn.innerHTML = '<svg class="animate-spin h-4 w-4 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';

                try {
                    await CartApi.addItem(productId, 1);
                    Toast.success('Added to cart!');
                    document.dispatchEvent(new CustomEvent('cart:updated'));
                } catch (error) {
                    Toast.error(error.message || 'Failed to add to cart.');
                } finally {
                    quickAddBtn.disabled = false;
                    quickAddBtn.innerHTML = originalHtml;
                }
            }
        });

        // Global quick add to wishlist (data-attribute and class-based)
        document.addEventListener('click', async (e) => {
            const wishlistBtn = e.target.closest('[data-wishlist-toggle], .wishlist-btn');
            if (wishlistBtn) {
                e.preventDefault();

                if (!AuthApi.isAuthenticated()) {
                    Toast.warning('Please login to add items to your wishlist.');
                    window.location.href = '/account/login/?next=' + encodeURIComponent(window.location.pathname);
                    return;
                }

                const productId = wishlistBtn.dataset.wishlistToggle || wishlistBtn.dataset.productId || wishlistBtn.closest('[data-product-id]')?.dataset.productId;
                if (!productId) {
                        // missing productId
                }

                wishlistBtn.disabled = true;
                let wishlistItemId = wishlistBtn.dataset.wishlistItemId || '';

                if (!wishlistItemId && wishlistBtn.classList.contains('text-red-500')) {
                    wishlistItemId = await findWishlistItemId(productId) || '';
                }

                const isActive = wishlistBtn.classList.contains('text-red-500');
                const shouldRemove = isActive && wishlistItemId;

                try {
                    if (shouldRemove) {
                        const res = await WishlistApi.removeItem(wishlistItemId);
                        wishlistBtn.classList.remove('text-red-500');
                        wishlistBtn.setAttribute('aria-pressed', 'false');
                        wishlistBtn.querySelector('svg')?.classList.remove('fill-current');
                        const fillPath = wishlistBtn.querySelector('svg')?.querySelector('.heart-fill');
                        if (fillPath) fillPath.style.opacity = '0';
                        wishlistBtn.removeAttribute('data-wishlist-item-id');
                        Toast.success('Removed from wishlist.');
                    } else {
                        const response = await WishlistApi.addItem(productId);
                        const createdId = response.data?.id || response.data?.item?.id || await findWishlistItemId(productId);
                        if (createdId) {
                            wishlistBtn.dataset.wishlistItemId = createdId;
                        }
                        wishlistBtn.classList.add('text-red-500');
                        wishlistBtn.setAttribute('aria-pressed', 'true');
                        wishlistBtn.querySelector('svg')?.classList.add('fill-current');
                        const fillPath = wishlistBtn.querySelector('svg')?.querySelector('.heart-fill');
                        if (fillPath) fillPath.style.opacity = '1';
                        Toast.success(response.message || 'Added to wishlist!');
                    }
                } catch (error) {
                    console.error('wishlist:error', error);
                    Toast.error(error.message || 'Failed to update wishlist.');
                } finally {
                    wishlistBtn.disabled = false;
                }
            }
        });

        // Quick view handler
        document.addEventListener('click', (e) => {
            const quickViewBtn = e.target.closest('[data-quick-view], .quick-view-btn');
            if (quickViewBtn) {
                e.preventDefault();
                const productId = quickViewBtn.dataset.quickView || quickViewBtn.dataset.productId;
                const productSlug = quickViewBtn.dataset.productSlug;
                
                if (productSlug) {
                    window.location.href = `/products/${productSlug}/`;
                } else if (productId) {
                    // If we have Modal component, show quick view modal
                    if (typeof Modal !== 'undefined' && Modal.showQuickView) {
                        Modal.showQuickView(productId);
                    } else {
                        window.location.href = `/products/${productId}/`;
                    }
                }
            }
        });

        // Handle logout
        document.addEventListener('click', async (e) => {
            const logoutBtn = e.target.closest('[data-logout]');
            if (logoutBtn) {
                e.preventDefault();
                
                try {
                    await AuthApi.logout();
                    Toast.success('Logged out successfully.');
                    document.dispatchEvent(new CustomEvent('auth:logout'));
                    window.location.href = '/';
                } catch (error) {
                    Toast.error('Failed to logout.');
                }
            }
        });

        // Handle back to top
        const backToTopBtn = document.getElementById('back-to-top');
        if (backToTopBtn) {
            window.addEventListener('scroll', Debounce.throttle(() => {
                if (window.scrollY > 500) {
                    backToTopBtn.classList.remove('opacity-0', 'pointer-events-none');
                } else {
                    backToTopBtn.classList.add('opacity-0', 'pointer-events-none');
                }
            }, 100));

            backToTopBtn.addEventListener('click', () => {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
        }
    }

    function updateAuthUI(isLoggedIn) {
        const authElements = document.querySelectorAll('[data-auth-state]');
        
        authElements.forEach(el => {
            const state = el.dataset.authState;
            if (state === 'logged-in') {
                el.classList.toggle('hidden', !isLoggedIn);
            } else if (state === 'logged-out') {
                el.classList.toggle('hidden', isLoggedIn);
            }
        });
    }

    function initMobileMenu() {
        const menuBtn = document.getElementById('mobile-menu-btn');
        const closeBtn = document.getElementById('close-mobile-menu');
        const mobileMenu = document.getElementById('mobile-menu');
        const overlay = document.getElementById('mobile-menu-overlay');

        function openMenu() {
            mobileMenu?.classList.remove('translate-x-full');
            overlay?.classList.remove('hidden');
            document.body.classList.add('overflow-hidden');
        }

        function closeMenu() {
            mobileMenu?.classList.add('translate-x-full');
            overlay?.classList.add('hidden');
            document.body.classList.remove('overflow-hidden');
        }

        menuBtn?.addEventListener('click', openMenu);
        closeBtn?.addEventListener('click', closeMenu);
        overlay?.addEventListener('click', closeMenu);
    }



    function initLanguageSelector() {
        const languageBtn = document.querySelector('[data-language-selector]');
        const languageDropdown = document.getElementById('language-dropdown');

        if (languageBtn && languageDropdown) {
            Dropdown.create(languageBtn, languageDropdown);

            languageDropdown.querySelectorAll('[data-language]').forEach(item => {
                item.addEventListener('click', async () => {
                    const lang = item.dataset.language;
                    
                    try {
                        await LocalizationApi.setLanguage(lang);
                        Storage.set('language', lang);
                        window.location.reload();
                    } catch (error) {
                        Toast.error('Failed to change language.');
                    }
                });
            });
        }
    }

    function destroy() {
        if (currentController && typeof currentController.destroy === 'function') {
            currentController.destroy();
        }
        currentPage = null;
        currentController = null;
    }

    return {
        init,
        destroy,
        getCurrentPage: () => currentPage,
        updateCartBadge
    };
})();

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', App.init);
} else {
    App.init();
}

window.App = App;
