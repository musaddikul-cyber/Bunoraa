/**
 * Category Page - Enhanced with Advanced E-commerce Features
 * @module pages/category
 */

const CategoryPage = (function() {
    'use strict';

    let currentFilters = {};
    let currentPage = 1;
    let currentCategory = null;
    let abortController = null;
    let initialized = false;
    let isLoadingMore = false;
    let hasMoreProducts = true;
    let allProducts = [];
    let compareList = [];
    const MAX_COMPARE = 4;

    async function init() {
        // Prevent multiple initializations
        if (initialized) return;
        initialized = true;

        const categorySlug = getCategorySlugFromUrl();
        
        // If no category slug, this is the categories list page - don't try to load a specific category
        if (!categorySlug) {
            // Just initialize any list-specific functionality if needed
            return;
        }

        // Check if content is already server-rendered
        const headerContainer = document.getElementById('category-header');
        if (headerContainer && headerContainer.querySelector('h1')) {
            // Server-rendered content exists - just bind event handlers
            initFilterHandlers();
            initSortHandler();
            initViewToggle();
            initEnhancedFeatures();
            return;
        }

        currentFilters = getFiltersFromUrl();
        currentPage = parseInt(new URLSearchParams(window.location.search).get('page')) || 1;

        await loadCategory(categorySlug);
        initFilterHandlers();
        initSortHandler();
        initViewToggle();
        initEnhancedFeatures();
    }

    // ============================================
    // ENHANCED FEATURES INITIALIZATION
    // ============================================
    function initEnhancedFeatures() {
        initInfiniteScroll();
        initCompareProducts();
        initQuickView();
        initPriceRangeSlider();
        initActiveFiltersDisplay();
        initProductCountDisplay();
    }

    // ============================================
    // ENHANCED FEATURE: Infinite Scroll
    // ============================================
    function initInfiniteScroll() {
        const loadMoreTrigger = document.getElementById('load-more-trigger');
        if (!loadMoreTrigger) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !isLoadingMore && hasMoreProducts) {
                    loadMoreProducts();
                }
            });
        }, {
            rootMargin: '200px 0px',
            threshold: 0.01
        });

        observer.observe(loadMoreTrigger);
    }

    async function loadMoreProducts() {
        if (isLoadingMore || !hasMoreProducts || !currentCategory) return;
        
        isLoadingMore = true;
        currentPage++;
        
        const loadingIndicator = document.getElementById('loading-more-indicator');
        if (loadingIndicator) {
            loadingIndicator.classList.remove('hidden');
        }

        try {
            const params = {
                category: currentCategory.id,
                page: currentPage,
                limit: 12,
                ...currentFilters
            };

            const response = await ProductsApi.getAll(params);
            const products = response.data || [];
            const meta = response.meta || {};

            if (products.length === 0) {
                hasMoreProducts = false;
            } else {
                allProducts = [...allProducts, ...products];
                appendProducts(products);
                hasMoreProducts = currentPage < (meta.total_pages || 1);
            }

            updateUrl();
        } catch (error) {
            console.error('Failed to load more products:', error);
        } finally {
            isLoadingMore = false;
            if (loadingIndicator) {
                loadingIndicator.classList.add('hidden');
            }
        }
    }

    function appendProducts(products) {
        const container = document.getElementById('products-grid');
        if (!container) return;

        const viewMode = Storage.get('productViewMode') || 'grid';
        
        products.forEach(product => {
            const productHtml = ProductCard.render(product, { 
                layout: viewMode,
                showCompare: true,
                showQuickView: true
            });
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = productHtml;
            const productEl = tempDiv.firstElementChild;
            productEl.classList.add('animate-fadeInUp');
            container.appendChild(productEl);
        });

        ProductCard.bindEvents(container);
    }

    // ============================================
    // ENHANCED FEATURE: Compare Products
    // ============================================
    function initCompareProducts() {
        // Load compare list from storage
        compareList = JSON.parse(localStorage.getItem('compareProducts') || '[]');
        updateCompareBar();

        // Listen for compare toggle events
        document.addEventListener('click', (e) => {
            const compareBtn = e.target.closest('[data-compare]');
            if (!compareBtn) return;

            e.preventDefault();
            const productId = parseInt(compareBtn.dataset.compare);
            toggleCompare(productId);
        });
    }

    function toggleCompare(productId) {
        const index = compareList.findIndex(p => p.id === productId);
        
        if (index > -1) {
            compareList.splice(index, 1);
            Toast.info('Removed from compare');
        } else {
            if (compareList.length >= MAX_COMPARE) {
                Toast.warning(`You can compare up to ${MAX_COMPARE} products`);
                return;
            }
            
            const product = allProducts.find(p => p.id === productId);
            if (product) {
                compareList.push({
                    id: product.id,
                    name: product.name,
                    image: product.primary_image || product.image,
                    price: product.price,
                    sale_price: product.sale_price
                });
                Toast.success('Added to compare');
            }
        }

        localStorage.setItem('compareProducts', JSON.stringify(compareList));
        updateCompareBar();
        updateCompareButtons();
    }

    function updateCompareBar() {
        let compareBar = document.getElementById('compare-bar');
        
        if (compareList.length === 0) {
            compareBar?.remove();
            return;
        }

        if (!compareBar) {
            compareBar = document.createElement('div');
            compareBar.id = 'compare-bar';
            compareBar.className = 'fixed bottom-0 left-0 right-0 bg-white dark:bg-stone-800 border-t border-stone-200 dark:border-stone-700 shadow-2xl z-40 transform transition-transform duration-300';
            document.body.appendChild(compareBar);
        }

        compareBar.innerHTML = `
            <div class="container mx-auto px-4 py-4">
                <div class="flex items-center justify-between gap-4">
                    <div class="flex items-center gap-3 overflow-x-auto">
                        <span class="text-sm font-medium text-stone-600 dark:text-stone-400 whitespace-nowrap">Compare (${compareList.length}/${MAX_COMPARE}):</span>
                        ${compareList.map(product => `
                            <div class="relative flex-shrink-0 group">
                                <img src="${product.image || '/static/images/placeholder.jpg'}" alt="${Templates.escapeHtml(product.name)}" class="w-14 h-14 object-cover rounded-lg border border-stone-200 dark:border-stone-600">
                                <button data-remove-compare="${product.id}" class="absolute -top-2 -right-2 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                    <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                                </button>
                            </div>
                        `).join('')}
                    </div>
                    <div class="flex items-center gap-2">
                        <button id="compare-now-btn" class="px-4 py-2 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-lg hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed" ${compareList.length < 2 ? 'disabled' : ''}>
                            Compare Now
                        </button>
                        <button id="clear-compare-btn" class="px-4 py-2 text-stone-600 dark:text-stone-400 hover:text-stone-800 dark:hover:text-stone-200 transition-colors">
                            Clear All
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Bind compare bar events
        compareBar.querySelectorAll('[data-remove-compare]').forEach(btn => {
            btn.addEventListener('click', () => {
                const productId = parseInt(btn.dataset.removeCompare);
                toggleCompare(productId);
            });
        });

        document.getElementById('compare-now-btn')?.addEventListener('click', openCompareModal);
        document.getElementById('clear-compare-btn')?.addEventListener('click', clearCompare);
    }

    function updateCompareButtons() {
        document.querySelectorAll('[data-compare]').forEach(btn => {
            const productId = parseInt(btn.dataset.compare);
            const isInCompare = compareList.some(p => p.id === productId);
            
            if (isInCompare) {
                btn.classList.add('bg-primary-100', 'text-primary-600');
                btn.classList.remove('bg-stone-100', 'text-stone-600');
            } else {
                btn.classList.remove('bg-primary-100', 'text-primary-600');
                btn.classList.add('bg-stone-100', 'text-stone-600');
            }
        });
    }

    function clearCompare() {
        compareList = [];
        localStorage.removeItem('compareProducts');
        updateCompareBar();
        updateCompareButtons();
        Toast.info('Compare list cleared');
    }

    async function openCompareModal() {
        if (compareList.length < 2) return;

        const modal = document.createElement('div');
        modal.id = 'compare-modal';
        modal.className = 'fixed inset-0 z-50 overflow-auto';
        modal.innerHTML = `
            <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('compare-modal').remove()"></div>
            <div class="relative min-h-full flex items-center justify-center p-4">
                <div class="bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-auto">
                    <div class="sticky top-0 bg-white dark:bg-stone-800 border-b border-stone-200 dark:border-stone-700 p-4 flex items-center justify-between z-10">
                        <h2 class="text-xl font-bold text-stone-900 dark:text-white">Compare Products</h2>
                        <button onclick="document.getElementById('compare-modal').remove()" class="w-10 h-10 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors">
                            <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                        </button>
                    </div>
                    <div class="p-4 overflow-x-auto">
                        <table class="w-full min-w-[600px]">
                            <thead>
                                <tr>
                                    <th class="text-left p-3 text-sm font-medium text-stone-500 dark:text-stone-400 w-32">Feature</th>
                                    ${compareList.map(p => `
                                        <th class="p-3 text-center">
                                            <div class="flex flex-col items-center">
                                                <img src="${p.image || '/static/images/placeholder.jpg'}" alt="${Templates.escapeHtml(p.name)}" class="w-24 h-24 object-cover rounded-xl mb-2">
                                                <span class="text-sm font-semibold text-stone-900 dark:text-white">${Templates.escapeHtml(p.name)}</span>
                                            </div>
                                        </th>
                                    `).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                <tr class="border-t border-stone-100 dark:border-stone-700">
                                    <td class="p-3 text-sm font-medium text-stone-600 dark:text-stone-400">Price</td>
                                    ${compareList.map(p => `
                                        <td class="p-3 text-center">
                                            ${p.sale_price ? `
                                                <span class="text-lg font-bold text-primary-600 dark:text-amber-400">${Templates.formatPrice(p.sale_price)}</span>
                                                <span class="text-sm text-stone-400 line-through ml-1">${Templates.formatPrice(p.price)}</span>
                                            ` : `
                                                <span class="text-lg font-bold text-stone-900 dark:text-white">${Templates.formatPrice(p.price)}</span>
                                            `}
                                        </td>
                                    `).join('')}
                                </tr>
                                <tr class="border-t border-stone-100 dark:border-stone-700">
                                    <td class="p-3 text-sm font-medium text-stone-600 dark:text-stone-400">Actions</td>
                                    ${compareList.map(p => `
                                        <td class="p-3 text-center">
                                            <button onclick="CartApi.addItem(${p.id}, 1).then(() => Toast.success('Added to cart'))" class="px-4 py-2 bg-primary-600 dark:bg-amber-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                                                Add to Cart
                                            </button>
                                        </td>
                                    `).join('')}
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
    }

    // ============================================
    // ENHANCED FEATURE: Quick View Modal
    // ============================================
    function initQuickView() {
        document.addEventListener('click', async (e) => {
            const quickViewBtn = e.target.closest('[data-quick-view]');
            if (!quickViewBtn) return;

            const productId = quickViewBtn.dataset.quickView;
            if (!productId) return;

            e.preventDefault();
            await showQuickView(productId);
        });
    }

    async function showQuickView(productId) {
        const modal = document.createElement('div');
        modal.id = 'quick-view-modal';
        modal.className = 'fixed inset-0 z-50 flex items-center justify-center p-4';
        modal.innerHTML = `
            <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('quick-view-modal').remove()"></div>
            <div class="relative bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-auto">
                <div class="p-8 flex items-center justify-center min-h-[400px]">
                    <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 dark:border-amber-400"></div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        try {
            const response = await ProductsApi.getProduct(productId);
            const product = response.data || response;

            const modalContent = modal.querySelector('.relative');
            modalContent.innerHTML = `
                <button class="absolute top-4 right-4 z-10 w-10 h-10 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors" onclick="document.getElementById('quick-view-modal').remove()">
                    <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                </button>
                <div class="grid md:grid-cols-2 gap-8 p-8">
                    <div>
                        <div class="aspect-square rounded-xl overflow-hidden bg-stone-100 dark:bg-stone-700 mb-4">
                            <img src="${product.primary_image || product.image || '/static/images/placeholder.jpg'}" alt="${Templates.escapeHtml(product.name)}" class="w-full h-full object-cover" id="quick-view-main-image">
                        </div>
                        ${product.images && product.images.length > 1 ? `
                            <div class="flex gap-2 overflow-x-auto pb-2">
                                ${product.images.slice(0, 5).map((img, i) => `
                                    <button class="w-16 h-16 flex-shrink-0 rounded-lg overflow-hidden border-2 ${i === 0 ? 'border-primary-600 dark:border-amber-400' : 'border-transparent'} hover:border-primary-400 transition-colors" onclick="document.getElementById('quick-view-main-image').src='${img.image || img}'">
                                        <img src="${img.thumbnail || img.image || img}" alt="" class="w-full h-full object-cover">
                                    </button>
                                `).join('')}
                            </div>
                        ` : ''}
                    </div>
                    <div class="flex flex-col">
                        <h2 class="text-2xl font-bold text-stone-900 dark:text-white mb-2">${Templates.escapeHtml(product.name)}</h2>
                        <div class="flex items-center gap-2 mb-4">
                            <div class="flex text-amber-400">
                                ${'<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.178c.969 0 1.371 1.24.588 1.81l-3.385 2.46a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.385-2.46a1 1 0 00-1.175 0l-3.385 2.46c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118l-3.385-2.46c-.783-.57-.38-1.81.588-1.81h4.178a1 1 0 00.95-.69l1.286-3.967z"/></svg>'.repeat(Math.round(product.rating || 4))}
                            </div>
                            <span class="text-sm text-stone-500 dark:text-stone-400">(${product.review_count || 0} reviews)</span>
                            ${product.stock_quantity <= 5 && product.stock_quantity > 0 ? `
                                <span class="text-sm text-orange-600 dark:text-orange-400 font-medium">Only ${product.stock_quantity} left!</span>
                            ` : ''}
                        </div>
                        <div class="mb-6">
                            ${product.sale_price || product.discounted_price ? `
                                <span class="text-3xl font-bold text-primary-600 dark:text-amber-400">${Templates.formatPrice(product.sale_price || product.discounted_price)}</span>
                                <span class="text-lg text-stone-400 line-through ml-2">${Templates.formatPrice(product.price)}</span>
                                <span class="ml-2 px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 text-sm font-medium rounded">Save ${Math.round((1 - (product.sale_price || product.discounted_price) / product.price) * 100)}%</span>
                            ` : `
                                <span class="text-3xl font-bold text-stone-900 dark:text-white">${Templates.formatPrice(product.price)}</span>
                            `}
                        </div>
                        <p class="text-stone-600 dark:text-stone-400 mb-6 line-clamp-3">${Templates.escapeHtml(product.short_description || product.description || '')}</p>
                        
                        <!-- Quantity Selector -->
                        <div class="flex items-center gap-4 mb-6">
                            <span class="text-sm font-medium text-stone-700 dark:text-stone-300">Quantity:</span>
                            <div class="flex items-center border border-stone-300 dark:border-stone-600 rounded-lg">
                                <button id="qv-qty-minus" class="w-10 h-10 flex items-center justify-center text-stone-600 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-700 transition-colors">−</button>
                                <input type="number" id="qv-qty-input" value="1" min="1" max="${product.stock_quantity || 99}" class="w-16 h-10 text-center border-x border-stone-300 dark:border-stone-600 bg-transparent text-stone-900 dark:text-white">
                                <button id="qv-qty-plus" class="w-10 h-10 flex items-center justify-center text-stone-600 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-700 transition-colors">+</button>
                            </div>
                        </div>

                        <div class="mt-auto space-y-3">
                            <button id="qv-add-to-cart" class="w-full py-3 px-6 bg-primary-600 dark:bg-amber-600 hover:bg-primary-700 dark:hover:bg-amber-700 text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"/></svg>
                                Add to Cart
                            </button>
                            <div class="grid grid-cols-2 gap-3">
                                <button onclick="WishlistApi.add(${product.id}).then(() => Toast.success('Added to wishlist'))" class="py-3 px-6 border-2 border-stone-200 dark:border-stone-600 text-stone-700 dark:text-stone-300 font-semibold rounded-xl hover:bg-stone-50 dark:hover:bg-stone-700 transition-colors flex items-center justify-center gap-2">
                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/></svg>
                                    Wishlist
                                </button>
                                <a href="/products/${product.slug || product.id}/" class="py-3 px-6 border-2 border-stone-200 dark:border-stone-600 text-stone-700 dark:text-stone-300 font-semibold rounded-xl text-center hover:bg-stone-50 dark:hover:bg-stone-700 transition-colors">
                                    Full Details
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Bind quantity controls
            const qtyInput = document.getElementById('qv-qty-input');
            const qtyMinus = document.getElementById('qv-qty-minus');
            const qtyPlus = document.getElementById('qv-qty-plus');
            const addToCartBtn = document.getElementById('qv-add-to-cart');

            qtyMinus?.addEventListener('click', () => {
                const val = parseInt(qtyInput.value) || 1;
                if (val > 1) qtyInput.value = val - 1;
            });

            qtyPlus?.addEventListener('click', () => {
                const val = parseInt(qtyInput.value) || 1;
                const max = parseInt(qtyInput.max) || 99;
                if (val < max) qtyInput.value = val + 1;
            });

            addToCartBtn?.addEventListener('click', async () => {
                const qty = parseInt(qtyInput.value) || 1;
                addToCartBtn.disabled = true;
                addToCartBtn.innerHTML = '<svg class="animate-spin h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';
                
                try {
                    await CartApi.addItem(product.id, qty);
                    Toast.success('Added to cart');
                    modal.remove();
                } catch (error) {
                    Toast.error('Failed to add to cart');
                } finally {
                    addToCartBtn.disabled = false;
                    addToCartBtn.innerHTML = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"/></svg> Add to Cart';
                }
            });
        } catch (error) {
            console.error('Failed to load product:', error);
            modal.remove();
            Toast.error('Failed to load product details');
        }
    }

    // ============================================
    // ENHANCED FEATURE: Price Range Slider
    // ============================================
    function initPriceRangeSlider() {
        const sliderContainer = document.getElementById('price-range-slider');
        if (!sliderContainer) return;

        // This creates an enhanced dual-thumb slider
        const minInput = document.getElementById('filter-min-price');
        const maxInput = document.getElementById('filter-max-price');
        
        if (!minInput || !maxInput) return;

        // Add input event listeners for real-time feedback
        [minInput, maxInput].forEach(input => {
            input.addEventListener('input', () => {
                updatePriceRangeDisplay();
            });
        });
    }

    function updatePriceRangeDisplay() {
        const display = document.getElementById('price-range-display');
        const minVal = document.getElementById('filter-min-price')?.value || 0;
        const maxVal = document.getElementById('filter-max-price')?.value || '∞';
        
        if (display) {
            display.textContent = `$${minVal} - $${maxVal}`;
        }
    }

    // ============================================
    // ENHANCED FEATURE: Active Filters Display
    // ============================================
    function initActiveFiltersDisplay() {
        updateActiveFiltersDisplay();
    }

    function updateActiveFiltersDisplay() {
        const container = document.getElementById('active-filters');
        if (!container) return;

        const filters = [];
        
        if (currentFilters.min_price) {
            filters.push({ key: 'min_price', label: `Min: $${currentFilters.min_price}` });
        }
        if (currentFilters.max_price) {
            filters.push({ key: 'max_price', label: `Max: $${currentFilters.max_price}` });
        }
        if (currentFilters.in_stock) {
            filters.push({ key: 'in_stock', label: 'In Stock' });
        }
        if (currentFilters.on_sale) {
            filters.push({ key: 'on_sale', label: 'On Sale' });
        }
        if (currentFilters.ordering) {
            const orderLabels = {
                'price': 'Price: Low to High',
                '-price': 'Price: High to Low',
                '-created_at': 'Newest First',
                'name': 'A-Z',
                '-popularity': 'Most Popular'
            };
            filters.push({ key: 'ordering', label: orderLabels[currentFilters.ordering] || currentFilters.ordering });
        }

        if (filters.length === 0) {
            container.innerHTML = '';
            return;
        }

        container.innerHTML = `
            <div class="flex flex-wrap items-center gap-2 mb-4">
                <span class="text-sm text-stone-500 dark:text-stone-400">Active filters:</span>
                ${filters.map(f => `
                    <button data-remove-filter="${f.key}" class="inline-flex items-center gap-1 px-3 py-1 bg-primary-100 dark:bg-amber-900/30 text-primary-700 dark:text-amber-400 rounded-full text-sm hover:bg-primary-200 dark:hover:bg-amber-900/50 transition-colors">
                        ${f.label}
                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                `).join('')}
                <button id="clear-all-active-filters" class="text-sm text-stone-500 dark:text-stone-400 hover:text-stone-700 dark:hover:text-stone-300 underline">Clear all</button>
            </div>
        `;

        container.querySelectorAll('[data-remove-filter]').forEach(btn => {
            btn.addEventListener('click', () => {
                const key = btn.dataset.removeFilter;
                delete currentFilters[key];
                applyFilters();
            });
        });

        document.getElementById('clear-all-active-filters')?.addEventListener('click', () => {
            currentFilters = {};
            applyFilters();
        });
    }

    // ============================================
    // ENHANCED FEATURE: Product Count Display
    // ============================================
    function initProductCountDisplay() {
        updateProductCount();
    }

    function updateProductCount(total = null) {
        const countEl = document.getElementById('product-count');
        if (!countEl) return;
        
        if (total !== null) {
            countEl.textContent = `${total} products`;
        }
    }

    function getCategorySlugFromUrl() {
        const path = window.location.pathname;
        const match = path.match(/\/categories\/([^\/]+)/);
        return match ? match[1] : null;
    }

    function getFiltersFromUrl() {
        const params = new URLSearchParams(window.location.search);
        const filters = {};

        if (params.get('min_price')) filters.min_price = params.get('min_price');
        if (params.get('max_price')) filters.max_price = params.get('max_price');
        if (params.get('ordering')) filters.ordering = params.get('ordering');
        if (params.get('in_stock')) filters.in_stock = params.get('in_stock') === 'true';
        if (params.get('on_sale')) filters.on_sale = params.get('on_sale') === 'true';
        
        const attributes = params.getAll('attr');
        if (attributes.length) filters.attributes = attributes;

        return filters;
    }

    async function loadCategory(slug) {
        const headerContainer = document.getElementById('category-header');
        const productsContainer = document.getElementById('category-products');
        const filtersContainer = document.getElementById('category-filters');

        if (headerContainer) Loader.show(headerContainer, 'skeleton');
        if (productsContainer) Loader.show(productsContainer, 'skeleton');

        try {
            const categoryResponse = await CategoriesApi.getCategory(slug);
            currentCategory = categoryResponse.data || categoryResponse;

            if (!currentCategory) {
                window.location.href = '/404/';
                return;
            }

            renderCategoryHeader(currentCategory);
            await loadBreadcrumbs(currentCategory);
            await loadFilters(currentCategory);
            await loadProducts();
            await loadSubcategories(currentCategory);
        } catch (error) {
            console.error('Failed to load category:', error);
            if (headerContainer) {
                headerContainer.innerHTML = '<p class="text-red-500">Failed to load category.</p>';
            }
        }
    }

    function renderCategoryHeader(category) {
        const container = document.getElementById('category-header');
        if (!container) return;

        document.title = `${category.name} | Bunoraa`;

        container.innerHTML = `
            <div class="relative py-8 md:py-12">
                ${category.image ? `
                    <div class="absolute inset-0 overflow-hidden rounded-2xl">
                        <img src="${category.image}" alt="" class="w-full h-full object-cover opacity-20">
                        <div class="absolute inset-0 bg-gradient-to-r from-white via-white/95 to-white/80"></div>
                    </div>
                ` : ''}
                <div class="relative">
                    <h1 class="text-3xl md:text-4xl font-bold text-gray-900 mb-2">${Templates.escapeHtml(category.name)}</h1>
                    ${category.description ? `
                        <p class="text-gray-600 max-w-2xl">${Templates.escapeHtml(category.description)}</p>
                    ` : ''}
                    ${category.product_count ? `
                        <p class="mt-4 text-sm text-gray-500">${category.product_count} products</p>
                    ` : ''}
                </div>
            </div>
        `;
    }

    async function loadBreadcrumbs(category) {
        const container = document.getElementById('breadcrumbs');
        if (!container) return;

        try {
            const response = await CategoriesApi.getBreadcrumbs(category.id);
            const breadcrumbs = response.data || [];
            
            const items = [
                { label: 'Home', url: '/' },
                { label: 'Categories', url: '/categories/' },
                ...breadcrumbs.map(item => ({
                    label: item.name,
                    url: `/categories/${item.slug}/`
                }))
            ];

            container.innerHTML = Breadcrumb.render(items);
        } catch (error) {
            console.error('Failed to load breadcrumbs:', error);
        }
    }

    async function loadFilters(category) {
        const container = document.getElementById('category-filters');
        if (!container) return;

        try {
            const response = await ProductsApi.getFilterOptions({ category: category.id });
            const filterOptions = response.data || {};

            container.innerHTML = `
                <div class="space-y-6">
                    <!-- Price Range -->
                    <div class="border-b border-gray-200 pb-6">
                        <h3 class="text-sm font-semibold text-gray-900 mb-4">Price Range</h3>
                        <div class="flex items-center gap-2">
                            <input 
                                type="number" 
                                id="filter-min-price" 
                                placeholder="Min"
                                value="${currentFilters.min_price || ''}"
                                class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-primary-500 focus:border-primary-500"
                            >
                            <span class="text-gray-400">-</span>
                            <input 
                                type="number" 
                                id="filter-max-price" 
                                placeholder="Max"
                                value="${currentFilters.max_price || ''}"
                                class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-primary-500 focus:border-primary-500"
                            >
                        </div>
                        <button id="apply-price-filter" class="mt-3 w-full px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium rounded-lg transition-colors">
                            Apply
                        </button>
                    </div>

                    <!-- Availability -->
                    <div class="border-b border-gray-200 pb-6">
                        <h3 class="text-sm font-semibold text-gray-900 mb-4">Availability</h3>
                        <div class="space-y-2">
                            <label class="flex items-center">
                                <input 
                                    type="checkbox" 
                                    id="filter-in-stock"
                                    ${currentFilters.in_stock ? 'checked' : ''}
                                    class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                                >
                                <span class="ml-2 text-sm text-gray-600">In Stock</span>
                            </label>
                            <label class="flex items-center">
                                <input 
                                    type="checkbox" 
                                    id="filter-on-sale"
                                    ${currentFilters.on_sale ? 'checked' : ''}
                                    class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                                >
                                <span class="ml-2 text-sm text-gray-600">On Sale</span>
                            </label>
                        </div>
                    </div>

                    ${filterOptions.attributes && filterOptions.attributes.length ? `
                        ${filterOptions.attributes.map(attr => `
                            <div class="border-b border-gray-200 pb-6">
                                <h3 class="text-sm font-semibold text-gray-900 mb-4">${Templates.escapeHtml(attr.name)}</h3>
                                <div class="space-y-2 max-h-48 overflow-y-auto">
                                    ${attr.values.map(value => `
                                        <label class="flex items-center">
                                            <input 
                                                type="checkbox" 
                                                name="attr-${attr.slug}"
                                                value="${Templates.escapeHtml(value.value)}"
                                                ${currentFilters.attributes?.includes(`${attr.slug}:${value.value}`) ? 'checked' : ''}
                                                class="filter-attribute w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                                                data-attribute="${attr.slug}"
                                            >
                                            <span class="ml-2 text-sm text-gray-600">${Templates.escapeHtml(value.value)}</span>
                                            ${value.count ? `<span class="ml-auto text-xs text-gray-400">(${value.count})</span>` : ''}
                                        </label>
                                    `).join('')}
                                </div>
                            </div>
                        `).join('')}
                    ` : ''}

                    <!-- Clear Filters -->
                    <button id="clear-filters" class="w-full px-4 py-2 border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors">
                        Clear All Filters
                    </button>
                </div>
            `;

            bindFilterEvents();
        } catch (error) {
            console.error('Failed to load filters:', error);
            container.innerHTML = '';
        }
    }

    function bindFilterEvents() {
        const applyPriceBtn = document.getElementById('apply-price-filter');
        const inStockCheckbox = document.getElementById('filter-in-stock');
        const onSaleCheckbox = document.getElementById('filter-on-sale');
        const clearBtn = document.getElementById('clear-filters');
        const attributeCheckboxes = document.querySelectorAll('.filter-attribute');

        applyPriceBtn?.addEventListener('click', () => {
            const minPrice = document.getElementById('filter-min-price')?.value;
            const maxPrice = document.getElementById('filter-max-price')?.value;
            
            if (minPrice) currentFilters.min_price = minPrice;
            else delete currentFilters.min_price;
            
            if (maxPrice) currentFilters.max_price = maxPrice;
            else delete currentFilters.max_price;

            applyFilters();
        });

        inStockCheckbox?.addEventListener('change', (e) => {
            if (e.target.checked) currentFilters.in_stock = true;
            else delete currentFilters.in_stock;
            applyFilters();
        });

        onSaleCheckbox?.addEventListener('change', (e) => {
            if (e.target.checked) currentFilters.on_sale = true;
            else delete currentFilters.on_sale;
            applyFilters();
        });

        attributeCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                updateAttributeFilters();
                applyFilters();
            });
        });

        clearBtn?.addEventListener('click', () => {
            currentFilters = {};
            currentPage = 1;
            applyFilters();
        });
    }

    function updateAttributeFilters() {
        const checkboxes = document.querySelectorAll('.filter-attribute:checked');
        const attributes = [];
        
        checkboxes.forEach(cb => {
            attributes.push(`${cb.dataset.attribute}:${cb.value}`);
        });

        if (attributes.length) {
            currentFilters.attributes = attributes;
        } else {
            delete currentFilters.attributes;
        }
    }

    function applyFilters() {
        currentPage = 1;
        updateUrl();
        loadProducts();
    }

    function updateUrl() {
        const params = new URLSearchParams();
        
        if (currentFilters.min_price) params.set('min_price', currentFilters.min_price);
        if (currentFilters.max_price) params.set('max_price', currentFilters.max_price);
        if (currentFilters.ordering) params.set('ordering', currentFilters.ordering);
        if (currentFilters.in_stock) params.set('in_stock', 'true');
        if (currentFilters.on_sale) params.set('on_sale', 'true');
        if (currentFilters.attributes) {
            currentFilters.attributes.forEach(attr => params.append('attr', attr));
        }
        if (currentPage > 1) params.set('page', currentPage);

        const newUrl = `${window.location.pathname}${params.toString() ? '?' + params.toString() : ''}`;
        window.history.pushState({}, '', newUrl);
    }

    async function loadProducts() {
        const container = document.getElementById('category-products');
        if (!container || !currentCategory) return;

        if (abortController) {
            abortController.abort();
        }
        abortController = new AbortController();

        Loader.show(container, 'skeleton');

        try {
            const params = {
                category: currentCategory.id,
                page: currentPage,
                limit: 12,
                ...currentFilters
            };

            if (currentFilters.attributes) {
                delete params.attributes;
                currentFilters.attributes.forEach(attr => {
                    const [key, value] = attr.split(':');
                    params[`attr_${key}`] = value;
                });
            }

            const response = await ProductsApi.getAll(params);
            const products = response.data || [];
            const meta = response.meta || {};

            // Track products for compare feature
            allProducts = products;
            hasMoreProducts = currentPage < (meta.total_pages || 1);

            renderProducts(products, meta);
            updateActiveFiltersDisplay();
            updateProductCount(meta.total || products.length);
        } catch (error) {
            if (error.name === 'AbortError') return;
            console.error('Failed to load products:', error);
            container.innerHTML = '<p class="text-red-500 text-center py-8">Failed to load products. Please try again.</p>';
        }
    }

    function renderProducts(products, meta) {
        const container = document.getElementById('category-products');
        if (!container) return;

        const viewMode = Storage.get('productViewMode') || 'grid';
        const gridClass = viewMode === 'list' 
            ? 'space-y-4' 
            : 'grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6';

        if (products.length === 0) {
            container.innerHTML = `
                <div class="text-center py-12">
                    <svg class="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">No products found</h3>
                    <p class="text-gray-500 dark:text-stone-400 mb-4">Try adjusting your filters or browse other categories.</p>
                    <button id="clear-filters-empty" class="px-4 py-2 bg-primary-600 dark:bg-amber-600 text-white rounded-lg hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                        Clear Filters
                    </button>
                </div>
            `;

            document.getElementById('clear-filters-empty')?.addEventListener('click', () => {
                currentFilters = {};
                currentPage = 1;
                applyFilters();
            });
            return;
        }

        container.innerHTML = `
            <div id="active-filters" class="mb-4"></div>
            <div id="products-grid" class="${gridClass}">
                ${products.map(product => ProductCard.render(product, { 
                    layout: viewMode,
                    showCompare: true,
                    showQuickView: true
                })).join('')}
            </div>
            
            <!-- Infinite Scroll Trigger -->
            <div id="load-more-trigger" class="h-20 flex items-center justify-center">
                <div id="loading-more-indicator" class="hidden">
                    <svg class="animate-spin h-8 w-8 text-primary-600 dark:text-amber-400" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                </div>
            </div>
            
            ${meta.total_pages > 1 ? `
                <div id="products-pagination" class="mt-8"></div>
            ` : ''}
        `;

        ProductCard.bindEvents(container);
        updateActiveFiltersDisplay();
        updateCompareButtons();

        // Initialize infinite scroll observer
        initInfiniteScroll();

        if (meta.total_pages > 1) {
            const paginationContainer = document.getElementById('products-pagination');
            paginationContainer.innerHTML = Pagination.render({
                currentPage: meta.current_page || currentPage,
                totalPages: meta.total_pages,
                totalItems: meta.total
            });

            paginationContainer.addEventListener('click', (e) => {
                const pageBtn = e.target.closest('[data-page]');
                if (pageBtn) {
                    currentPage = parseInt(pageBtn.dataset.page);
                    hasMoreProducts = true;
                    updateUrl();
                    loadProducts();
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                }
            });
        }
    }

    async function loadSubcategories(category) {
        const container = document.getElementById('subcategories');
        if (!container) return;

        try {
            const response = await CategoriesApi.getSubcategories(category.id);
            const subcategories = response.data || [];

            if (subcategories.length === 0) {
                container.innerHTML = '';
                return;
            }

            container.innerHTML = `
                <div class="mb-8">
                    <h2 class="text-lg font-semibold text-gray-900 mb-4">Browse Subcategories</h2>
                    <div class="flex flex-wrap gap-2">
                        ${subcategories.map(sub => `
                            <a href="/categories/${sub.slug}/" class="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full text-sm transition-colors">
                                ${Templates.escapeHtml(sub.name)}
                                ${sub.product_count ? `<span class="text-gray-400 ml-1">(${sub.product_count})</span>` : ''}
                            </a>
                        `).join('')}
                    </div>
                </div>
            `;
        } catch (error) {
            console.error('Failed to load subcategories:', error);
            container.innerHTML = '';
        }
    }

    function initFilterHandlers() {
        const mobileFilterBtn = document.getElementById('mobile-filter-btn');
        const filterSidebar = document.getElementById('filter-sidebar');
        const closeFilterBtn = document.getElementById('close-filter-btn');

        mobileFilterBtn?.addEventListener('click', () => {
            filterSidebar?.classList.remove('hidden');
            document.body.classList.add('overflow-hidden');
        });

        closeFilterBtn?.addEventListener('click', () => {
            filterSidebar?.classList.add('hidden');
            document.body.classList.remove('overflow-hidden');
        });
    }

    function initSortHandler() {
        const sortSelect = document.getElementById('sort-select');
        if (!sortSelect) return;

        sortSelect.value = currentFilters.ordering || '';

        sortSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                currentFilters.ordering = e.target.value;
            } else {
                delete currentFilters.ordering;
            }
            applyFilters();
        });
    }

    function initViewToggle() {
        const gridBtn = document.getElementById('view-grid');
        const listBtn = document.getElementById('view-list');
        const currentView = Storage.get('productViewMode') || 'grid';

        if (currentView === 'list') {
            gridBtn?.classList.remove('bg-gray-200');
            listBtn?.classList.add('bg-gray-200');
        }

        gridBtn?.addEventListener('click', () => {
            Storage.set('productViewMode', 'grid');
            gridBtn.classList.add('bg-gray-200');
            listBtn?.classList.remove('bg-gray-200');
            loadProducts();
        });

        listBtn?.addEventListener('click', () => {
            Storage.set('productViewMode', 'list');
            listBtn.classList.add('bg-gray-200');
            gridBtn?.classList.remove('bg-gray-200');
            loadProducts();
        });
    }

    function destroy() {
        if (abortController) {
            abortController.abort();
            abortController = null;
        }
        currentFilters = {};
        currentPage = 1;
        currentCategory = null;
        initialized = false;
        isLoadingMore = false;
        hasMoreProducts = true;
        allProducts = [];
        // Remove compare bar
        document.getElementById('compare-bar')?.remove();
        // Remove quick view modal
        document.getElementById('quick-view-modal')?.remove();
        // Remove compare modal
        document.getElementById('compare-modal')?.remove();
    }

    return {
        init,
        destroy,
        // Expose for external use
        toggleCompare,
        clearCompare
    };
})();

window.CategoryPage = CategoryPage;
export default CategoryPage;
