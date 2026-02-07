/**
 * Bunoraa ML Recommendations Library
 * 
 * Client-side library for fetching and displaying ML-powered recommendations.
 * Integrates with the ML API to provide:
 * - Personalized product recommendations
 * - Similar products
 * - Frequently bought together
 * - Trending/popular products
 * - Cart recommendations
 * 
 * Usage:
 *   <script src="/static/js/ml-recommendations.js"></script>
 *   BunoraaML.init();
 *   BunoraaML.loadRecommendations('#container', { type: 'personalized', limit: 8 });
 */

(function(window, document) {
    'use strict';

    // Configuration
    const CONFIG = {
        apiBase: '/api/ml',
        cacheTime: 5 * 60 * 1000, // 5 minutes
        retryAttempts: 2,
        retryDelay: 1000,
        animationDuration: 300,
    };

    // Cache for recommendations
    const cache = new Map();

    // ================================
    // API Functions
    // ================================

    async function fetchWithRetry(url, options = {}, attempts = CONFIG.retryAttempts) {
        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest',
                    ...options.headers,
                },
                credentials: 'same-origin',
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            if (attempts > 1) {
                await new Promise(resolve => setTimeout(resolve, CONFIG.retryDelay));
                return fetchWithRetry(url, options, attempts - 1);
            }
            throw error;
        }
    }

    function getCacheKey(type, params) {
        return `${type}:${JSON.stringify(params)}`;
    }

    function getFromCache(key) {
        const item = cache.get(key);
        if (item && Date.now() - item.timestamp < CONFIG.cacheTime) {
            return item.data;
        }
        cache.delete(key);
        return null;
    }

    function setCache(key, data) {
        cache.set(key, { data, timestamp: Date.now() });
    }

    // ================================
    // Recommendation Fetchers
    // ================================

    async function getPersonalizedRecommendations(limit = 10) {
        const cacheKey = getCacheKey('personalized', { limit });
        const cached = getFromCache(cacheKey);
        if (cached) return cached;

        const data = await fetchWithRetry(`${CONFIG.apiBase}/recommendations/?limit=${limit}`);
        setCache(cacheKey, data.recommendations || []);
        return data.recommendations || [];
    }

    async function getSimilarProducts(productId, limit = 8) {
        const cacheKey = getCacheKey('similar', { productId, limit });
        const cached = getFromCache(cacheKey);
        if (cached) return cached;

        const data = await fetchWithRetry(`${CONFIG.apiBase}/recommendations/similar/${productId}/?limit=${limit}`);
        setCache(cacheKey, data.similar_products || []);
        return data.similar_products || [];
    }

    async function getFrequentlyBoughtTogether(productId, limit = 5) {
        const cacheKey = getCacheKey('fbt', { productId, limit });
        const cached = getFromCache(cacheKey);
        if (cached) return cached;

        const data = await fetchWithRetry(`${CONFIG.apiBase}/recommendations/fbt/${productId}/?limit=${limit}`);
        setCache(cacheKey, data.frequently_bought_together || []);
        return data.frequently_bought_together || [];
    }

    async function getPopularProducts(limit = 10, categoryId = null) {
        const cacheKey = getCacheKey('popular', { limit, categoryId });
        const cached = getFromCache(cacheKey);
        if (cached) return cached;

        let url = `${CONFIG.apiBase}/recommendations/popular/?limit=${limit}`;
        if (categoryId) url += `&category_id=${categoryId}`;

        const data = await fetchWithRetry(url);
        setCache(cacheKey, data.products || []);
        return data.products || [];
    }

    async function getCartRecommendations(productIds, limit = 5) {
        const cacheKey = getCacheKey('cart', { productIds, limit });
        const cached = getFromCache(cacheKey);
        if (cached) return cached;

        const data = await fetchWithRetry(`${CONFIG.apiBase}/recommendations/cart/`, {
            method: 'POST',
            body: JSON.stringify({ product_ids: productIds, limit }),
        });
        setCache(cacheKey, data.recommendations || []);
        return data.recommendations || [];
    }

    async function getTrendingProducts(limit = 10, days = 7) {
        const cacheKey = getCacheKey('trending', { limit, days });
        const cached = getFromCache(cacheKey);
        if (cached) return cached;

        const data = await fetchWithRetry(`${CONFIG.apiBase}/recommendations/popular/?limit=${limit}&days=${days}`);
        setCache(cacheKey, data.products || []);
        return data.products || [];
    }

    async function getVisuallySimilarProducts(productId, limit = 8) {
        const cacheKey = getCacheKey('visual', { productId, limit });
        const cached = getFromCache(cacheKey);
        if (cached) return cached;

        const data = await fetchWithRetry(`${CONFIG.apiBase}/recommendations/visual-similar/${productId}/?limit=${limit}`);
        setCache(cacheKey, data.visually_similar_products || []);
        return data.visually_similar_products || [];
    }

    // ================================
    // UI Components
    // ================================

    function createProductCard(product, options = {}) {
        const {
            showPrice = true,
            showRating = true,
            showQuickView = false,
            cardClass = '',
            imageSize = 'medium',
        } = options;

        const imageSizes = {
            small: 'w-32 h-32',
            medium: 'w-full aspect-square',
            large: 'w-full h-80',
        };

        const discount = product.compare_price && product.compare_price > product.price
            ? Math.round((1 - product.price / product.compare_price) * 100)
            : 0;

        return `
            <div class="product-card group relative bg-white dark:bg-stone-900 rounded-2xl shadow-sm hover:shadow-lg transition-all duration-300 overflow-hidden ${cardClass}"
                 data-product-id="${product.id}"
                 data-product-name="${product.name}"
                 data-ml-recommendation="true">
                
                <!-- Image -->
                <a href="${product.url || `/products/${product.slug}/`}" class="block ${imageSizes[imageSize]} overflow-hidden">
                    <img src="${product.image || product.primary_image || '/static/images/placeholder.jpg'}"
                         alt="${product.name}"
                         loading="lazy"
                         class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500">
                    
                    ${discount > 0 ? `
                        <span class="absolute top-3 left-3 bg-red-500 text-white text-xs font-bold px-2 py-1 rounded-full">
                            -${discount}%
                        </span>
                    ` : ''}
                    
                    ${product.is_new ? `
                        <span class="absolute top-3 right-3 bg-emerald-500 text-white text-xs font-bold px-2 py-1 rounded-full">
                            NEW
                        </span>
                    ` : ''}
                </a>
                
                <!-- Quick Actions -->
                <div class="absolute top-3 right-3 flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <button onclick="BunoraaML.addToWishlist(${product.id})"
                            class="p-2 bg-white dark:bg-stone-800 rounded-full shadow-md hover:bg-red-50 dark:hover:bg-red-900 transition-colors"
                            aria-label="Add to wishlist">
                        <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
                        </svg>
                    </button>
                    ${showQuickView ? `
                        <button onclick="BunoraaML.quickView(${product.id})"
                                class="p-2 bg-white dark:bg-stone-800 rounded-full shadow-md hover:bg-primary-50 dark:hover:bg-amber-900 transition-colors"
                                aria-label="Quick view">
                            <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
                            </svg>
                        </button>
                    ` : ''}
                </div>
                
                <!-- Content -->
                <div class="p-4">
                    ${product.category_name ? `
                        <p class="text-xs text-stone-500 dark:text-stone-400 mb-1">${product.category_name}</p>
                    ` : ''}
                    
                    <a href="${product.url || `/products/${product.slug}/`}" class="block">
                        <h3 class="font-medium text-stone-900 dark:text-white mb-2 line-clamp-2 hover:text-primary-600 dark:hover:text-amber-400 transition-colors">
                            ${product.name}
                        </h3>
                    </a>
                    
                    ${showRating && product.rating ? `
                        <div class="flex items-center gap-1 mb-2">
                            <div class="flex text-yellow-400">
                                ${Array(5).fill(0).map((_, i) => `
                                    <svg class="w-4 h-4 ${i < Math.round(product.rating) ? 'fill-current' : 'text-stone-300 dark:text-stone-600'}" viewBox="0 0 20 20">
                                        <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z"/>
                                    </svg>
                                `).join('')}
                            </div>
                            <span class="text-xs text-stone-500 dark:text-stone-400">(${product.reviews_count || 0})</span>
                        </div>
                    ` : ''}
                    
                    ${showPrice ? `
                        <div class="flex items-center gap-2">
                            <span class="text-lg font-bold text-stone-900 dark:text-white">
                                ৳${product.price?.toLocaleString() || '0'}
                            </span>
                            ${product.compare_price && product.compare_price > product.price ? `
                                <span class="text-sm text-stone-400 line-through">
                                    ৳${product.compare_price.toLocaleString()}
                                </span>
                            ` : ''}
                        </div>
                    ` : ''}
                    
                    <!-- Add to Cart -->
                    <button onclick="BunoraaML.addToCart(${product.id})"
                            class="mt-3 w-full py-2 px-4 bg-primary-600 dark:bg-amber-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors opacity-0 group-hover:opacity-100">
                        Add to Cart
                    </button>
                </div>
                
                ${product.ml_score ? `
                    <div class="absolute bottom-2 right-2 text-xs text-stone-400 dark:text-stone-500" title="ML Score: ${(product.ml_score * 100).toFixed(1)}%">
                        <svg class="w-4 h-4 inline" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z"/>
                        </svg>
                    </div>
                ` : ''}
            </div>
        `;
    }

    function createProductGrid(products, options = {}) {
        const {
            columns = { sm: 2, md: 3, lg: 4, xl: 5 },
            gap = 6,
        } = options;

        if (!products || products.length === 0) {
            return `
                <div class="text-center py-12 text-stone-500 dark:text-stone-400">
                    <p>No recommendations available yet.</p>
                </div>
            `;
        }

        const gridClass = `grid grid-cols-2 sm:grid-cols-${columns.sm} md:grid-cols-${columns.md} lg:grid-cols-${columns.lg} xl:grid-cols-${columns.xl} gap-${gap}`;

        return `
            <div class="${gridClass}">
                ${products.map((product, index) => 
                    createProductCard(product, { ...options, position: index })
                ).join('')}
            </div>
        `;
    }

    function createCarousel(products, options = {}) {
        const { id = 'ml-carousel', showArrows = true, autoplay = false } = options;

        if (!products || products.length === 0) {
            return '';
        }

        return `
            <div id="${id}" class="relative">
                <div class="overflow-x-auto scrollbar-hide scroll-smooth snap-x snap-mandatory pb-4">
                    <div class="flex gap-4" style="width: max-content;">
                        ${products.map((product, index) => `
                            <div class="snap-start" style="width: 220px;">
                                ${createProductCard(product, { ...options, imageSize: 'small' })}
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                ${showArrows ? `
                    <button onclick="BunoraaML.scrollCarousel('${id}', -1)"
                            class="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-4 p-2 bg-white dark:bg-stone-800 rounded-full shadow-lg hover:bg-stone-50 dark:hover:bg-stone-700 transition-colors z-10">
                        <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                        </svg>
                    </button>
                    <button onclick="BunoraaML.scrollCarousel('${id}', 1)"
                            class="absolute right-0 top-1/2 -translate-y-1/2 translate-x-4 p-2 bg-white dark:bg-stone-800 rounded-full shadow-lg hover:bg-stone-50 dark:hover:bg-stone-700 transition-colors z-10">
                        <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                        </svg>
                    </button>
                ` : ''}
            </div>
        `;
    }

    function createSectionHeader(title, subtitle = '', viewAllUrl = '') {
        return `
            <div class="flex items-center justify-between mb-8">
                <div>
                    <h2 class="text-2xl lg:text-3xl font-display font-bold text-stone-900 dark:text-white mb-2">
                        ${title}
                    </h2>
                    ${subtitle ? `<p class="text-stone-600 dark:text-stone-400">${subtitle}</p>` : ''}
                </div>
                ${viewAllUrl ? `
                    <a href="${viewAllUrl}" class="hidden md:flex items-center text-primary-600 dark:text-amber-400 font-medium hover:text-primary-700 dark:hover:text-amber-300 transition-colors">
                        View All
                        <svg class="w-5 h-5 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/>
                        </svg>
                    </a>
                ` : ''}
            </div>
        `;
    }

    function showLoading(container) {
        const skeletons = Array(4).fill(0).map(() => `
            <div class="animate-pulse">
                <div class="bg-stone-200 dark:bg-stone-700 rounded-2xl aspect-square mb-4"></div>
                <div class="bg-stone-200 dark:bg-stone-700 h-4 rounded mb-2 w-3/4"></div>
                <div class="bg-stone-200 dark:bg-stone-700 h-4 rounded w-1/2"></div>
            </div>
        `).join('');

        container.innerHTML = `<div class="grid grid-cols-2 md:grid-cols-4 gap-6">${skeletons}</div>`;
    }

    // ================================
    // Main API
    // ================================

    async function loadRecommendations(containerSelector, options = {}) {
        const container = typeof containerSelector === 'string' 
            ? document.querySelector(containerSelector) 
            : containerSelector;

        if (!container) {
            console.warn('BunoraaML: Container not found:', containerSelector);
            return;
        }

        const {
            type = 'personalized',
            productId = null,
            categoryId = null,
            limit = 8,
            layout = 'grid',
            title = null,
            subtitle = null,
            viewAllUrl = null,
            showHeader = false,
        } = options;

        // Show loading state
        showLoading(container);

        try {
            let products = [];

            switch (type) {
                case 'personalized':
                    products = await getPersonalizedRecommendations(limit);
                    break;
                case 'similar':
                    if (!productId) throw new Error('productId required for similar products');
                    products = await getSimilarProducts(productId, limit);
                    break;
                case 'visual':
                    if (!productId) throw new Error('productId required for visually similar products');
                    products = await getVisuallySimilarProducts(productId, limit);
                    break;
                case 'fbt':
                case 'frequently-bought-together':
                    if (!productId) throw new Error('productId required for FBT');
                    products = await getFrequentlyBoughtTogether(productId, limit);
                    break;
                case 'popular':
                case 'trending':
                    products = await getPopularProducts(limit, categoryId);
                    break;
                case 'cart':
                    const cartIds = getCartProductIds();
                    if (cartIds.length === 0) {
                        products = await getPopularProducts(limit);
                    } else {
                        products = await getCartRecommendations(cartIds, limit);
                    }
                    break;
                default:
                    products = await getPersonalizedRecommendations(limit);
            }

            // Build HTML
            let html = '';

            if (showHeader && title) {
                html += createSectionHeader(title, subtitle, viewAllUrl);
            }

            if (layout === 'carousel') {
                html += createCarousel(products, options);
            } else {
                html += createProductGrid(products, options);
            }

            // Animate in
            container.style.opacity = '0';
            container.innerHTML = html;
            
            requestAnimationFrame(() => {
                container.style.transition = `opacity ${CONFIG.animationDuration}ms ease-in-out`;
                container.style.opacity = '1';
            });

            // Track impression
            trackRecommendationImpression(type, products);

        } catch (error) {
            console.error('BunoraaML: Error loading recommendations:', error);
            container.innerHTML = `
                <div class="text-center py-8 text-stone-500 dark:text-stone-400">
                    <p>Could not load recommendations. Please try again later.</p>
                </div>
            `;
        }
    }

    function getCartProductIds() {
        try {
            const cart = JSON.parse(localStorage.getItem('cart') || '{}');
            return Object.keys(cart.items || {}).map(id => parseInt(id));
        } catch {
            return [];
        }
    }

    function trackRecommendationImpression(type, products) {
        if (window.BunoraaTracking && products.length > 0) {
            window.dispatchEvent(new CustomEvent('bunoraa:ml_impression', {
                detail: {
                    recommendation_type: type,
                    product_ids: products.map(p => p.id),
                    count: products.length,
                }
            }));
        }
    }

    // ================================
    // User Actions
    // ================================

    function addToCart(productId, quantity = 1) {
        // Dispatch event for main app to handle
        window.dispatchEvent(new CustomEvent('bunoraa:add_to_cart', {
            detail: { product_id: productId, quantity, source: 'ml_recommendation' }
        }));
    }

    function addToWishlist(productId) {
        window.dispatchEvent(new CustomEvent('bunoraa:add_to_wishlist', {
            detail: { product_id: productId, source: 'ml_recommendation' }
        }));
    }

    function quickView(productId) {
        window.dispatchEvent(new CustomEvent('bunoraa:quick_view', {
            detail: { product_id: productId, source: 'ml_recommendation' }
        }));
    }

    function scrollCarousel(carouselId, direction) {
        const carousel = document.querySelector(`#${carouselId} .overflow-x-auto`);
        if (carousel) {
            const scrollAmount = 240 * direction;
            carousel.scrollBy({ left: scrollAmount, behavior: 'smooth' });
        }
    }

    // ================================
    // Auto-initialization
    // ================================

    function init() {
        // Find all elements with data-ml-recommendations attribute
        document.querySelectorAll('[data-ml-recommendations]').forEach(el => {
            const options = {
                type: el.dataset.mlRecommendations || 'personalized',
                productId: el.dataset.productId ? parseInt(el.dataset.productId) : null,
                categoryId: el.dataset.categoryId ? parseInt(el.dataset.categoryId) : null,
                limit: el.dataset.limit ? parseInt(el.dataset.limit) : 8,
                layout: el.dataset.layout || 'grid',
                title: el.dataset.title || null,
                subtitle: el.dataset.subtitle || null,
                showHeader: el.dataset.showHeader === 'true',
            };

            loadRecommendations(el, options);
        });

        // Auto-load on product detail pages
        const productDetail = document.querySelector('[data-product-id]');
        if (productDetail) {
            const productId = parseInt(productDetail.dataset.productId);
            
            // Load similar products
            const similarContainer = document.querySelector('#ml-similar-products, #similar-products');
            if (similarContainer) {
                loadRecommendations(similarContainer, {
                    type: 'similar',
                    productId,
                    limit: 8,
                    title: 'You May Also Like',
                    subtitle: 'Based on this product',
                    showHeader: true,
                });
            }

            // Load frequently bought together
            const fbtContainer = document.querySelector('#ml-fbt, #frequently-bought-together');
            if (fbtContainer) {
                loadRecommendations(fbtContainer, {
                    type: 'fbt',
                    productId,
                    limit: 5,
                    layout: 'carousel',
                    title: 'Frequently Bought Together',
                    showHeader: true,
                });
            }
        }
    }

    // ================================
    // Public API
    // ================================

    window.BunoraaML = {
        init,
        loadRecommendations,
        getPersonalizedRecommendations,
        getSimilarProducts,
        getFrequentlyBoughtTogether,
        getPopularProducts,
        getCartRecommendations,
        getTrendingProducts,
        addToCart,
        addToWishlist,
        quickView,
        scrollCarousel,
        createProductCard,
        createProductGrid,
        createCarousel,
        clearCache: () => cache.clear(),
    };

    // Auto-init on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})(window, document);
