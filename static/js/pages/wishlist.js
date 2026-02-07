/**
 * Wishlist Page - Comprehensive Version
 * @module pages/wishlist
 * Features: Priority levels, price tracking, reminders, target prices, bulk operations
 */

const WishlistPage = (function() {
    'use strict';

    let currentPage = 1;
    let currentSort = 'added_desc';
    let currentFilter = 'all';

    // Get currency configuration dynamically
    function getCurrencySymbol() {
        return window.BUNORAA_CURRENCY?.symbol || '৳';
    }

    // Configuration
    const CONFIG = {
        get CURRENCY_SYMBOL() { return getCurrencySymbol(); },
        PRIORITY_LEVELS: {
            low: { label: 'Low', color: 'gray', icon: '○' },
            normal: { label: 'Normal', color: 'blue', icon: '◐' },
            high: { label: 'High', color: 'amber', icon: '●' },
            urgent: { label: 'Urgent', color: 'red', icon: '★' }
        }
    };

    async function init() {
        if (!AuthGuard.protectPage()) return;

        await loadWishlist();
        bindGlobalEvents();
    }

    function resolveProductImage(item = {}) {
        const product = item.product || item || {};
        const candidates = [
            item.product_image,
            product.product_image,
            product.primary_image,
            product.image,
            Array.isArray(product.images) ? product.images[0] : null,
            product.image_url,
            product.thumbnail
        ];

        const pick = (val) => {
            if (!val) return '';
            if (typeof val === 'string') return val;
            if (typeof val === 'object') {
                if (typeof val.image === 'string' && val.image) return val.image;
                if (val.image && typeof val.image === 'object') {
                    if (typeof val.image.url === 'string' && val.image.url) return val.image.url;
                    if (typeof val.image.src === 'string' && val.image.src) return val.image.src;
                }
                if (typeof val.url === 'string' && val.url) return val.url;
                if (typeof val.src === 'string' && val.src) return val.src;
            }
            return '';
        };

        for (const candidate of candidates) {
            const url = pick(candidate);
            if (url) return url;
        }

        return '';
    }

    function resolvePrices(item = {}) {
        const product = item.product || item || {};
        
        const pickNumber = (val) => {
            if (val === null || val === undefined) return null;
            const num = Number(val);
            return Number.isFinite(num) ? num : null;
        };

        const priceCandidates = [
            item.product_price,
            product.price,
            item.price,
            item.current_price,
            item.price_at_add
        ];
        
        let price = null;
        for (const p of priceCandidates) {
            price = pickNumber(p);
            if (price !== null) break;
        }

        const saleCandidates = [
            item.product_sale_price,
            product.sale_price,
            item.sale_price
        ];
        
        let salePrice = null;
        for (const s of saleCandidates) {
            salePrice = pickNumber(s);
            if (salePrice !== null) break;
        }

        // Price tracking
        const lowestPrice = pickNumber(item.lowest_price_seen);
        const highestPrice = pickNumber(item.highest_price_seen);
        const targetPrice = pickNumber(item.target_price);
        const priceAtAdd = pickNumber(item.price_at_add);

        return {
            price: price !== null ? price : 0,
            salePrice: salePrice !== null ? salePrice : null,
            lowestPrice,
            highestPrice,
            targetPrice,
            priceAtAdd
        };
    }

    async function loadWishlist() {
        const container = document.getElementById('wishlist-container');
        if (!container) return;

        Loader.show(container, 'skeleton');

        try {
            const response = await WishlistApi.getWishlist({ page: currentPage, sort: currentSort });
            let items = [];
            let meta = {};

            if (Array.isArray(response)) {
                items = response;
            } else if (response && typeof response === 'object') {
                items = response.data || response.results || response.items || [];
                if (!Array.isArray(items) && response.data && typeof response.data === 'object') {
                    items = response.data.items || response.data.results || [];
                    meta = response.data.meta || response.meta || {};
                } else {
                    meta = response.meta || {};
                }
            }

            if (!Array.isArray(items)) {
                items = items && typeof items === 'object' ? [items] : [];
            }

            // Apply client-side filter
            let filteredItems = items;
            if (currentFilter === 'on_sale') {
                filteredItems = items.filter(item => {
                    const prices = resolvePrices(item);
                    return prices.salePrice && prices.salePrice < prices.price;
                });
            } else if (currentFilter === 'in_stock') {
                filteredItems = items.filter(item => item.is_in_stock !== false);
            } else if (currentFilter === 'price_drop') {
                filteredItems = items.filter(item => {
                    const prices = resolvePrices(item);
                    return prices.priceAtAdd && prices.price < prices.priceAtAdd;
                });
            } else if (currentFilter === 'at_target') {
                filteredItems = items.filter(item => {
                    const prices = resolvePrices(item);
                    return prices.targetPrice && prices.price <= prices.targetPrice;
                });
            }

            renderWishlist(filteredItems, items, meta);
        } catch (error) {
            const msg = (error && (error.message || error.detail)) || 'Failed to load wishlist.';

            if (error && error.status === 401) {
                AuthGuard.redirectToLogin();
                return;
            }

            container.innerHTML = `<p class="text-red-500 text-center py-8">${Templates.escapeHtml(msg)}</p>`;
        }
    }

    function renderWishlist(filteredItems, allItems, meta) {
        const container = document.getElementById('wishlist-container');
        if (!container) return;

        // Calculate stats
        const totalItems = allItems.length;
        const onSaleCount = allItems.filter(item => {
            const p = resolvePrices(item);
            return p.salePrice && p.salePrice < p.price;
        }).length;
        const priceDropCount = allItems.filter(item => {
            const p = resolvePrices(item);
            return p.priceAtAdd && p.price < p.priceAtAdd;
        }).length;
        const atTargetCount = allItems.filter(item => {
            const p = resolvePrices(item);
            return p.targetPrice && p.price <= p.targetPrice;
        }).length;

        if (totalItems === 0) {
            container.innerHTML = `
                <div class="text-center py-16">
                    <svg class="w-24 h-24 text-gray-300 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
                    </svg>
                    <h2 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">Your wishlist is empty</h2>
                    <p class="text-gray-600 dark:text-gray-400 mb-8">Start adding items you love to your wishlist.</p>
                    <a href="/products/" class="inline-flex items-center px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 transition-colors">
                        Browse Products
                    </a>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <!-- Header with Stats -->
            <div class="mb-6">
                <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                    <div>
                        <h1 class="text-2xl font-bold text-gray-900 dark:text-white">My Wishlist</h1>
                        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">${totalItems} items saved</p>
                    </div>
                    <div class="flex flex-wrap gap-2">
                        <button id="add-all-to-cart-btn" class="px-4 py-2 bg-primary-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 transition-colors flex items-center">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path>
                            </svg>
                            Add All to Cart
                        </button>
                        <button id="share-wishlist-btn" class="px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 text-sm font-medium rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors flex items-center">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"></path>
                            </svg>
                            Share
                        </button>
                        <button id="clear-wishlist-btn" class="px-4 py-2 text-red-600 dark:text-red-400 text-sm font-medium hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors">
                            Clear All
                        </button>
                    </div>
                </div>
                
                <!-- Quick Stats -->
                <div class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div class="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                        <div class="text-2xl font-bold text-gray-900 dark:text-white">${totalItems}</div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">Total Items</div>
                    </div>
                    <div class="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700 ${onSaleCount > 0 ? 'ring-2 ring-green-500' : ''}">
                        <div class="text-2xl font-bold text-green-600 dark:text-green-400">${onSaleCount}</div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">On Sale</div>
                    </div>
                    <div class="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700 ${priceDropCount > 0 ? 'ring-2 ring-blue-500' : ''}">
                        <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">${priceDropCount}</div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">Price Dropped</div>
                    </div>
                    <div class="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700 ${atTargetCount > 0 ? 'ring-2 ring-amber-500' : ''}">
                        <div class="text-2xl font-bold text-amber-600 dark:text-amber-400">${atTargetCount}</div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">At Target Price</div>
                    </div>
                </div>
            </div>
            
            <!-- Filters and Sort -->
            <div class="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <div class="flex flex-wrap gap-2">
                    <button class="filter-btn px-3 py-1.5 text-sm rounded-lg transition-colors ${currentFilter === 'all' ? 'bg-primary-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}" data-filter="all">All</button>
                    <button class="filter-btn px-3 py-1.5 text-sm rounded-lg transition-colors ${currentFilter === 'on_sale' ? 'bg-primary-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}" data-filter="on_sale">On Sale</button>
                    <button class="filter-btn px-3 py-1.5 text-sm rounded-lg transition-colors ${currentFilter === 'in_stock' ? 'bg-primary-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}" data-filter="in_stock">In Stock</button>
                    <button class="filter-btn px-3 py-1.5 text-sm rounded-lg transition-colors ${currentFilter === 'price_drop' ? 'bg-primary-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}" data-filter="price_drop">Price Drop</button>
                    <button class="filter-btn px-3 py-1.5 text-sm rounded-lg transition-colors ${currentFilter === 'at_target' ? 'bg-primary-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}" data-filter="at_target">At Target</button>
                </div>
                <div class="flex items-center gap-2">
                    <label class="text-sm text-gray-500 dark:text-gray-400">Sort:</label>
                    <select id="wishlist-sort" class="text-sm border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 focus:ring-primary-500 focus:border-primary-500">
                        <option value="added_desc" ${currentSort === 'added_desc' ? 'selected' : ''}>Newest First</option>
                        <option value="added_asc" ${currentSort === 'added_asc' ? 'selected' : ''}>Oldest First</option>
                        <option value="price_asc" ${currentSort === 'price_asc' ? 'selected' : ''}>Price: Low to High</option>
                        <option value="price_desc" ${currentSort === 'price_desc' ? 'selected' : ''}>Price: High to Low</option>
                        <option value="priority" ${currentSort === 'priority' ? 'selected' : ''}>Priority</option>
                        <option value="name" ${currentSort === 'name' ? 'selected' : ''}>Name A-Z</option>
                    </select>
                </div>
            </div>
            
            <!-- Items Grid -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                ${filteredItems.map(item => renderWishlistItem(item)).join('')}
            </div>
            
            ${filteredItems.length === 0 && totalItems > 0 ? `
                <div class="text-center py-12">
                    <p class="text-gray-500 dark:text-gray-400">No items match the selected filter.</p>
                    <button class="mt-4 text-primary-600 hover:underline" onclick="document.querySelector('[data-filter=all]').click()">Show all items</button>
                </div>
            ` : ''}
            
            ${meta.total_pages > 1 ? `<div id="wishlist-pagination" class="mt-8"></div>` : ''}
        `;

        // Mount pagination
        if (meta && meta.total_pages > 1) {
            const mount = document.getElementById('wishlist-pagination');
            if (mount && window.Pagination) {
                const paginator = new window.Pagination({
                    totalPages: meta.total_pages,
                    currentPage: meta.current_page || currentPage,
                    className: 'justify-center',
                    onChange: (page) => {
                        currentPage = page;
                        loadWishlist();
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                    }
                });
                mount.innerHTML = '';
                mount.appendChild(paginator.create());
            }
        }

        bindEvents();
    }

    function renderWishlistItem(item) {
        try {
            const product = item.product || item || {};
            const productName = item.product_name || product.name || '';
            const productSlug = item.product_slug || product.slug || '';
            const inStock = item.is_in_stock !== undefined ? item.is_in_stock : (product.is_in_stock !== undefined ? product.is_in_stock : (product.stock_quantity > 0));
            const imageUrl = resolveProductImage(item || {});
            const requiresVariants = !!item.product_has_variants;

            let priceObj = { price: 0, salePrice: null, lowestPrice: null, highestPrice: null, targetPrice: null, priceAtAdd: null };
            try { priceObj = resolvePrices(item || {}); } catch { priceObj = { price: 0, salePrice: null }; }
            const { price, salePrice, lowestPrice, highestPrice, targetPrice, priceAtAdd } = priceObj;
            
            // Calculate price change
            const currentPrice = salePrice || price;
            const hasPriceDrop = priceAtAdd && currentPrice < priceAtAdd;
            const priceChangePercent = priceAtAdd ? Math.round(((currentPrice - priceAtAdd) / priceAtAdd) * 100) : 0;
            const isAtTarget = targetPrice && currentPrice <= targetPrice;
            const isOnSale = salePrice && salePrice < price;
            
            // Priority
            const priority = item.priority || 'normal';
            const priorityConfig = CONFIG.PRIORITY_LEVELS[priority] || CONFIG.PRIORITY_LEVELS.normal;

            const safeEscape = (s) => {
                try { return Templates.escapeHtml(s || ''); } catch { return String(s || ''); }
            };

            const safePriceRender = (p) => {
                try { return Price.render({ price: p.price, salePrice: p.salePrice }); } catch { return `<span class="font-bold">${CONFIG.CURRENCY_SYMBOL}${p.price || 0}</span>`; }
            };

            const formatPrice = (p) => {
                try { return Templates.formatPrice(p); } catch { return `${CONFIG.CURRENCY_SYMBOL}${p}`; }
            };

            const aspectCss = (product && product.aspect && (product.aspect.css || (product.aspect.width && product.aspect.height ? `${product.aspect.width}/${product.aspect.height}` : null))) || '1/1';

            return `
                <div class="wishlist-item relative bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 overflow-hidden group" 
                     data-item-id="${item && item.id ? item.id : ''}" 
                     data-product-id="${(product && product.id) ? product.id : (item && item.product) ? item.product : ''}" 
                     data-product-slug="${safeEscape(productSlug)}" 
                     data-product-has-variants="${requiresVariants}"
                     data-priority="${priority}">
                    
                    <!-- Image Section -->
                    <div class="relative" style="aspect-ratio: ${aspectCss};">
                        <!-- Badges -->
                        <div class="absolute top-2 left-2 z-10 flex flex-col gap-1">
                            ${isOnSale ? `
                                <div class="bg-red-500 text-white text-xs font-bold px-2 py-1 rounded">
                                    -${Math.round((1 - salePrice / price) * 100)}%
                                </div>
                            ` : ''}
                            ${hasPriceDrop ? `
                                <div class="bg-blue-500 text-white text-xs font-bold px-2 py-1 rounded flex items-center">
                                    <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
                                    </svg>
                                    ${Math.abs(priceChangePercent)}% drop
                                </div>
                            ` : ''}
                            ${isAtTarget ? `
                                <div class="bg-amber-500 text-white text-xs font-bold px-2 py-1 rounded flex items-center">
                                    <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path>
                                    </svg>
                                    Target!
                                </div>
                            ` : ''}
                            ${!inStock ? `
                                <div class="bg-gray-800 text-white text-xs font-bold px-2 py-1 rounded">
                                    Out of Stock
                                </div>
                            ` : ''}
                        </div>
                        
                        <!-- Priority Indicator -->
                        <div class="absolute top-2 right-12 z-10">
                            <button class="priority-btn w-8 h-8 rounded-full bg-white dark:bg-gray-700 shadow-md flex items-center justify-center text-${priorityConfig.color}-500 hover:scale-110 transition-transform" title="Priority: ${priorityConfig.label}" data-item-id="${item.id}">
                                <span class="text-sm">${priorityConfig.icon}</span>
                            </button>
                        </div>
                        
                        <!-- Remove Button -->
                        <button class="remove-btn absolute top-2 right-2 z-20 w-8 h-8 bg-gray-900/80 text-white rounded-full shadow-lg flex items-center justify-center hover:bg-red-600 transition-colors opacity-0 group-hover:opacity-100" aria-label="Remove from wishlist">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                            </svg>
                        </button>
                        
                        <!-- Product Image -->
                        <a href="/products/${safeEscape(productSlug)}/">
                            ${imageUrl ? `
                                <img 
                                    src="${imageUrl}" 
                                    alt="${safeEscape(productName)}"
                                    class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                    loading="lazy"
                                >
                            ` : `
                                <div class="w-full h-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-gray-400 dark:text-gray-500 text-xs uppercase tracking-wide">No Image</div>
                            `}
                        </a>
                    </div>
                    
                    <!-- Content Section -->
                    <div class="p-4">
                        ${product && product.category ? `
                            <a href="/categories/${safeEscape(product.category.slug)}/" class="text-xs text-gray-500 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400">
                                ${safeEscape(product.category.name)}
                            </a>
                        ` : ''}
                        <h3 class="font-medium text-gray-900 dark:text-white mt-1 line-clamp-2">
                            <a href="/products/${safeEscape(productSlug)}/" class="hover:text-primary-600 dark:hover:text-primary-400">
                                ${safeEscape(productName)}
                            </a>
                        </h3>
                        
                        <!-- Price Section -->
                        <div class="mt-2">
                            ${safePriceRender({ price, salePrice })}
                        </div>
                        
                        <!-- Price History -->
                        ${lowestPrice || targetPrice ? `
                            <div class="mt-2 text-xs space-y-1">
                                ${lowestPrice ? `
                                    <div class="flex items-center justify-between text-gray-500 dark:text-gray-400">
                                        <span>Lowest:</span>
                                        <span class="font-medium text-green-600 dark:text-green-400">${formatPrice(lowestPrice)}</span>
                                    </div>
                                ` : ''}
                                ${targetPrice ? `
                                    <div class="flex items-center justify-between text-gray-500 dark:text-gray-400">
                                        <span>Target:</span>
                                        <span class="font-medium text-amber-600 dark:text-amber-400">${formatPrice(targetPrice)}</span>
                                    </div>
                                ` : ''}
                            </div>
                        ` : ''}
                        
                        <!-- Rating -->
                        ${product && product.average_rating ? `
                            <div class="flex items-center gap-1 mt-2">
                                ${Templates.renderStars(product.average_rating)}
                                <span class="text-xs text-gray-500 dark:text-gray-400">(${product.review_count || 0})</span>
                            </div>
                        ` : ''}
                        
                        <!-- Actions -->
                        <div class="mt-4 flex gap-2">
                            <button 
                                class="add-to-cart-btn flex-1 px-3 py-2 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-sm flex items-center justify-center"
                                ${!inStock ? 'disabled' : ''}
                            >
                                <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path>
                                </svg>
                                ${requiresVariants ? 'Options' : (inStock ? 'Add' : 'Sold Out')}
                            </button>
                            <button class="set-target-btn px-3 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-sm" title="Set target price" data-item-id="${item.id}" data-current-price="${currentPrice}">
                                <svg class="w-4 h-4" fill="${targetPrice ? 'currentColor' : 'none'}" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"></path>
                                </svg>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Added Date -->
                    ${item && item.added_at ? `
                        <div class="px-4 pb-3 border-t border-gray-100 dark:border-gray-700 pt-3">
                            <p class="text-xs text-gray-400 dark:text-gray-500">Added ${Templates.formatDate(item.added_at)}</p>
                        </div>
                    ` : ''}
                </div>
            `;
        } catch (err) {
            console.error('Failed to render wishlist item:', err);
            return '<div class="p-4 bg-white dark:bg-gray-800 rounded shadow text-gray-500 dark:text-gray-400">Failed to render item</div>';
        }
    }

    function bindGlobalEvents() {
        // Sort change
        document.getElementById('wishlist-sort')?.addEventListener('change', (e) => {
            currentSort = e.target.value;
            loadWishlist();
        });

        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                currentFilter = btn.dataset.filter;
                loadWishlist();
            });
        });
    }

    function bindEvents() {
        const clearAllBtn = document.getElementById('clear-wishlist-btn');
        const addAllBtn = document.getElementById('add-all-to-cart-btn');
        const shareBtn = document.getElementById('share-wishlist-btn');
        const wishlistItems = document.querySelectorAll('.wishlist-item');
        const sortSelect = document.getElementById('wishlist-sort');
        const filterBtns = document.querySelectorAll('.filter-btn');

        // Sort change
        sortSelect?.addEventListener('change', (e) => {
            currentSort = e.target.value;
            loadWishlist();
        });

        // Filter buttons
        filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                currentFilter = btn.dataset.filter;
                loadWishlist();
            });
        });

        // Clear all
        clearAllBtn?.addEventListener('click', async () => {
            const confirmed = await Modal.confirm({
                title: 'Clear Wishlist',
                message: 'Are you sure you want to remove all items from your wishlist?',
                confirmText: 'Clear All',
                cancelText: 'Cancel'
            });

            if (confirmed) {
                try {
                    await WishlistApi.clear();
                    Toast.success('Wishlist cleared.');
                    await loadWishlist();
                } catch (error) {
                    Toast.error(error.message || 'Failed to clear wishlist.');
                }
            }
        });

        // Add all to cart
        addAllBtn?.addEventListener('click', async () => {
            const btn = addAllBtn;
            btn.disabled = true;
            btn.innerHTML = '<svg class="animate-spin w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Adding...';
            
            try {
                // Get all items without variants
                const items = document.querySelectorAll('.wishlist-item:not([data-product-has-variants="true"])');
                let added = 0;
                let failed = 0;
                
                for (const item of items) {
                    const productId = item.dataset.productId;
                    if (!productId) continue;
                    
                    try {
                        await CartApi.addItem(productId, 1);
                        added++;
                    } catch (err) {
                        failed++;
                    }
                }
                
                if (added > 0) {
                    Toast.success(`Added ${added} items to cart!`);
                    document.dispatchEvent(new CustomEvent('cart:updated'));
                }
                if (failed > 0) {
                    Toast.warning(`${failed} items could not be added (may require variant selection).`);
                }
            } catch (error) {
                Toast.error(error.message || 'Failed to add items to cart.');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>Add All to Cart';
            }
        });

        // Share wishlist
        shareBtn?.addEventListener('click', async () => {
            try {
                const shareUrl = `${window.location.origin}/wishlist/share/`;
                
                if (navigator.share) {
                    await navigator.share({
                        title: 'My Wishlist',
                        text: 'Check out my wishlist!',
                        url: shareUrl
                    });
                } else {
                    await navigator.clipboard.writeText(shareUrl);
                    Toast.success('Wishlist link copied to clipboard!');
                }
            } catch (error) {
                if (error.name !== 'AbortError') {
                    Toast.error('Failed to share wishlist.');
                }
            }
        });

        // Item-specific events
        wishlistItems.forEach(item => {
            const itemId = item.dataset.itemId;
            const productId = item.dataset.productId;
            const productSlug = item.dataset.productSlug;

            // Remove button
            item.querySelector('.remove-btn')?.addEventListener('click', async () => {
                try {
                    await WishlistApi.removeItem(itemId);
                    Toast.success('Removed from wishlist.');
                    item.remove();

                    const remaining = document.querySelectorAll('.wishlist-item');
                    if (remaining.length === 0) {
                        await loadWishlist();
                    }
                } catch (error) {
                    Toast.error(error.message || 'Failed to remove item.');
                }
            });

            // Priority button
            item.querySelector('.priority-btn')?.addEventListener('click', async () => {
                const priorities = ['low', 'normal', 'high', 'urgent'];
                const currentPriority = item.dataset.priority || 'normal';
                const currentIndex = priorities.indexOf(currentPriority);
                const nextPriority = priorities[(currentIndex + 1) % priorities.length];
                
                try {
                    // Update via API if available
                    if (WishlistApi.updateItem) {
                        await WishlistApi.updateItem(itemId, { priority: nextPriority });
                    }
                    
                    // Update UI
                    item.dataset.priority = nextPriority;
                    const btn = item.querySelector('.priority-btn');
                    const config = CONFIG.PRIORITY_LEVELS[nextPriority];
                    btn.title = `Priority: ${config.label}`;
                    btn.innerHTML = `<span class="text-sm">${config.icon}</span>`;
                    btn.className = `priority-btn w-8 h-8 rounded-full bg-white dark:bg-gray-700 shadow-md flex items-center justify-center text-${config.color}-500 hover:scale-110 transition-transform`;
                    
                    Toast.success(`Priority set to ${config.label}`);
                } catch (error) {
                    Toast.error('Failed to update priority.');
                }
            });

            // Set target price button
            item.querySelector('.set-target-btn')?.addEventListener('click', async () => {
                const currentPrice = parseFloat(item.querySelector('.set-target-btn').dataset.currentPrice) || 0;
                
                const content = `
                    <div class="space-y-4">
                        <p class="text-sm text-gray-600 dark:text-gray-400">Set a target price and we'll notify you when the item drops to or below this price.</p>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Current Price</label>
                            <div class="text-lg font-bold text-gray-900 dark:text-white">${CONFIG.CURRENCY_SYMBOL}${currentPrice.toLocaleString()}</div>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Target Price</label>
                            <div class="flex items-center">
                                <span class="text-gray-500 mr-2">${CONFIG.CURRENCY_SYMBOL}</span>
                                <input type="number" id="target-price-input" value="${Math.round(currentPrice * 0.9)}" min="1" max="${currentPrice}" class="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500">
                            </div>
                            <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">Suggested: ${CONFIG.CURRENCY_SYMBOL}${Math.round(currentPrice * 0.9).toLocaleString()} (10% off)</p>
                        </div>
                    </div>
                `;
                
                const confirmed = await Modal.open({
                    title: 'Set Target Price',
                    content,
                    confirmText: 'Set Alert',
                    cancelText: 'Cancel',
                    onConfirm: async () => {
                        const targetPrice = parseFloat(document.getElementById('target-price-input').value);
                        
                        if (!targetPrice || targetPrice <= 0) {
                            Toast.error('Please enter a valid target price.');
                            return false;
                        }
                        
                        try {
                            if (WishlistApi.updateItem) {
                                await WishlistApi.updateItem(itemId, { target_price: targetPrice });
                            }
                            Toast.success(`Price alert set for ${CONFIG.CURRENCY_SYMBOL}${targetPrice.toLocaleString()}`);
                            await loadWishlist();
                            return true;
                        } catch (error) {
                            Toast.error('Failed to set price alert.');
                            return false;
                        }
                    }
                });
            });

            // Add to cart button
            item.querySelector('.add-to-cart-btn')?.addEventListener('click', async (e) => {
                const btn = e.target.closest('.add-to-cart-btn');
                if (btn.disabled) return;

                btn.disabled = true;
                const originalText = btn.innerHTML;
                btn.innerHTML = '<svg class="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';

                const requiresVariants = item.dataset.productHasVariants === 'true' || item.dataset.productHasVariants === 'True' || item.dataset.productHasVariants === '1';

                if (requiresVariants) {
                    showVariantPicker(item);
                    btn.disabled = false;
                    btn.innerHTML = originalText;
                    return;
                }

                try {
                    await CartApi.addItem(productId, 1);
                    Toast.success('Added to cart!');
                    document.dispatchEvent(new CustomEvent('cart:updated'));
                } catch (error) {
                    const hasVariantError = Boolean(
                        error && (
                            (error.errors && error.errors.variant_id) ||
                            (error.message && typeof error.message === 'string' && error.message.toLowerCase().includes('variant'))
                        )
                    );

                    if (hasVariantError) {
                        Toast.info('This product requires selecting a variant.');
                        if (productSlug) {
                            window.location.href = `/products/${productSlug}/`;
                            return;
                        }
                    }

                    Toast.error(error.message || 'Failed to add to cart.');
                } finally {
                    btn.disabled = false;
                    btn.innerHTML = originalText;
                }
            });
        });
    }

    function renderModalVariants(variants) {
        const grouped = {};
        variants.forEach(variant => {
            if (!grouped[variant.attribute_name]) grouped[variant.attribute_name] = [];
            grouped[variant.attribute_name].push(variant);
        });

        return Object.entries(grouped).map(([attrName, options]) => `
            <div class="mt-4">
                <label class="text-sm font-medium text-gray-700">${Templates.escapeHtml(attrName)}:</label>
                <div class="flex flex-wrap gap-2 mt-2" id="wishlist-variant-group-${Templates.slugify(attrName)}">
                    ${options.map((opt, index) => `
                        <button type="button" class="wishlist-modal-variant-btn px-3 py-2 border rounded-lg text-sm transition-colors ${index === 0 ? 'border-primary-500 bg-primary-50 text-primary-700' : 'border-gray-300 hover:border-gray-400'}" data-variant-id="${opt.id}" data-price="${opt.price_converted ?? opt.price ?? ''}" data-stock="${opt.stock_quantity || 0}">
                            ${Templates.escapeHtml(opt.value)}
                            ${(opt.price_converted ?? opt.price) ? `<span class="text-xs text-gray-500"> (${Templates.formatPrice(opt.price_converted ?? opt.price)})</span>` : ''}
                        </button>
                    `).join('')}
                </div>
            </div>
        `).join('');
    }

    async function showVariantPicker(item) {
        // item: wishlist item object or DOM element dataset; prefer slug
        const slug = item.product_slug || item.dataset?.productSlug || '';
        const id = item.product || item.dataset?.productId || '';

        try {
            // Support environments where ProductsApi may not be available (fallback to ApiClient)
            let res;
            if (typeof ProductsApi !== 'undefined' && ProductsApi.getProduct) {
                res = await ProductsApi.getProduct(slug || id);
            } else {
                const currency = (window.BUNORAA_CURRENCY && window.BUNORAA_CURRENCY.code) || undefined;
                res = await ApiClient.get(`/catalog/products/${slug || id}/`, { currency });
            }

            if (!res || !res.success || !res.data) {
                const msg = res && res.message ? res.message : 'Failed to load product variants.';
                Toast.error(msg);
                return;
            }

            const product = res.data;
            const variants = product.variants || [];
            if (!variants.length) {
                // fallback to product page
                window.location.href = `/products/${product.slug || slug || id}/`;
                return;
            }

            const firstImage = product.images?.[0]?.image || product.primary_image || product.image || '';

            const content = `
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="col-span-1">
                        ${firstImage ? `<img src="${firstImage}" class="w-full h-48 object-cover rounded" alt="${Templates.escapeHtml(product.name)}">` : `<div class="w-full h-48 bg-gray-100 rounded"></div>`}
                    </div>
                    <div class="col-span-2">
                        <h3 class="text-lg font-semibold">${Templates.escapeHtml(product.name)}</h3>
                        <div id="wishlist-variant-price" class="mt-2 text-lg font-bold">${Templates.formatPrice(variants?.[0]?.price_converted ?? variants?.[0]?.price ?? product.price)}</div>
                        <div id="wishlist-variant-options" class="mt-4">
                            ${renderModalVariants(variants)}
                        </div>
                        <div class="mt-4 flex items-center gap-2">
                            <label class="text-sm text-gray-700">Qty</label>
                            <input id="wishlist-variant-qty" type="number" value="1" min="1" class="w-20 px-3 py-2 border rounded" />
                        </div>
                    </div>
                </div>
            `;

            const confirmed = await Modal.open({
                title: 'Select Variant',
                content,
                confirmText: 'Add to Cart',
                cancelText: 'Cancel',
                size: 'md',
                onConfirm: async () => {
                    // find selected variant
                    const active = document.querySelector('.wishlist-modal-variant-btn.border-primary-500');
                    const btn = active || document.querySelector('.wishlist-modal-variant-btn');
                    if (!btn) {
                        Toast.error('Please select a variant.');
                        return false;
                    }
                    const variantId = btn.dataset.variantId;
                    const qty = parseInt(document.getElementById('wishlist-variant-qty')?.value) || 1;

                    try {
                        await CartApi.addItem(product.id, qty, variantId);
                        Toast.success('Added to cart!');
                        document.dispatchEvent(new CustomEvent('cart:updated'));
                        return true;
                    } catch (err) {
                        Toast.error(err.message || 'Failed to add to cart.');
                        return false;
                    }
                }
            });

            // bind variant buttons to update selection and price after modal is rendered
            setTimeout(() => {
                const variantBtns = document.querySelectorAll('.wishlist-modal-variant-btn');
                variantBtns.forEach(btn => {
                    btn.addEventListener('click', () => {
                        variantBtns.forEach(b => b.classList.remove('border-primary-500', 'bg-primary-50', 'text-primary-700'));
                        btn.classList.add('border-primary-500', 'bg-primary-50', 'text-primary-700');

                        const price = btn.dataset.price;
                        if (price !== undefined) {
                            const priceEl = document.getElementById('wishlist-variant-price');
                            if (priceEl) priceEl.textContent = Templates.formatPrice(price);
                        }
                    });
                });
                // preselect first variant
                const first = document.querySelector('.wishlist-modal-variant-btn');
                if (first) first.click();
            }, 20);

        } catch (error) {
            Toast.error('Failed to load variants.');
        }
    }

    function destroy() {
        currentPage = 1;
    }

    return {
        init,
        destroy
    };
})();

window.WishlistPage = WishlistPage;
export default WishlistPage;
