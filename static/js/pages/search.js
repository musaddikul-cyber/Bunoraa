/**
 * Search Page
 * @module pages/search
 */

const SearchPage = (function() {
    'use strict';

    let currentQuery = '';
    let currentPage = 1;
    let currentFilters = {};
    let abortController = null;
    let initialized = false;

    async function init() {
        // Prevent multiple initializations
        if (initialized) return;
        initialized = true;

        // Check if content is already server-rendered
        const resultsContainer = document.getElementById('search-results') || document.getElementById('products-grid');
        const hasServerContent = resultsContainer && resultsContainer.querySelector('.product-card, [data-product-id]');
        
        if (hasServerContent) {
            // Server-rendered content exists - just bind event handlers
            initSearchForm();
            initFilters();
            initSorting();
            initViewToggle();
            return;
        }

        currentQuery = getQueryFromUrl();
        currentFilters = getFiltersFromUrl();
        currentPage = parseInt(new URLSearchParams(window.location.search).get('page')) || 1;

        // Load products whether there's a search query or not
        await loadProducts();

        initSearchForm();
        initFilters();
        initSorting();
        initViewToggle();
    }

    function initViewToggle() {
        const gridBtn = document.getElementById('view-grid');
        const listBtn = document.getElementById('view-list');
        
        gridBtn?.addEventListener('click', () => {
            gridBtn.classList.add('bg-primary-100', 'text-primary-700');
            gridBtn.classList.remove('text-gray-400');
            listBtn?.classList.remove('bg-primary-100', 'text-primary-700');
            listBtn?.classList.add('text-gray-400');
            Storage?.set('productViewMode', 'grid');
            loadProducts();
        });
        
        listBtn?.addEventListener('click', () => {
            listBtn.classList.add('bg-primary-100', 'text-primary-700');
            listBtn.classList.remove('text-gray-400');
            gridBtn?.classList.remove('bg-primary-100', 'text-primary-700');
            gridBtn?.classList.add('text-gray-400');
            Storage?.set('productViewMode', 'list');
            loadProducts();
        });
    }

    async function loadProducts() {
        const resultsContainer = document.getElementById('search-results') || document.getElementById('products-grid');
        const resultsCount = document.getElementById('results-count') || document.getElementById('product-count');
        
        if (!resultsContainer) return;

        if (abortController) {
            abortController.abort();
        }
        abortController = new AbortController();

        Loader.show(resultsContainer, 'skeleton');

        try {
            const params = {
                page: currentPage,
                pageSize: 12,
                ...currentFilters
            };
            
            if (currentQuery) {
                params.search = currentQuery;
            }

            // Check if we're on categories page
            const path = window.location.pathname;
            if (path === '/categories/') {
                // Load categories instead of products
                await loadCategories();
                return;
            }

            const response = await ProductsApi.getProducts(params);
            const products = Array.isArray(response) ? response : (response.data || response.results || []);
            const meta = response.meta || {};

            if (resultsCount) {
                if (currentQuery) {
                    resultsCount.textContent = `${meta.total || products.length} results for "${Templates.escapeHtml(currentQuery)}"`;
                } else {
                    resultsCount.textContent = `${meta.total || products.length} products`;
                }
            }

            renderResults(products, meta);
            await loadFilterOptions();
        } catch (error) {
            if (error.name === 'AbortError') return;
            console.error('Failed to load products:', error);
            resultsContainer.innerHTML = '<p class="text-red-500 text-center py-8">Failed to load products. Please try again.</p>';
        }
    }

    async function loadCategories() {
        const resultsContainer = document.getElementById('search-results') || document.getElementById('products-grid');
        const resultsCount = document.getElementById('results-count') || document.getElementById('product-count');
        const pageTitle = document.getElementById('page-title');
        
        if (!resultsContainer) return;

        try {
            const response = await CategoriesApi.getCategories({ limit: 50 });
            const categories = Array.isArray(response) ? response : (response.data || response.results || []);

            if (pageTitle) {
                pageTitle.textContent = 'All Categories';
            }

            if (resultsCount) {
                resultsCount.textContent = `${categories.length} categories`;
            }

            if (categories.length === 0) {
                resultsContainer.innerHTML = `
                    <div class="text-center py-16">
                        <svg class="w-24 h-24 text-gray-300 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
                        </svg>
                        <h2 class="text-2xl font-bold text-gray-900 mb-2">No categories found</h2>
                        <p class="text-gray-600">Check back later for new categories.</p>
                    </div>
                `;
                return;
            }

            resultsContainer.innerHTML = `
                <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
                    ${categories.map(cat => `
                        <a href="/categories/${cat.slug}/" class="group bg-white rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 overflow-hidden">
                            <div class="relative overflow-hidden" style="aspect-ratio: ${product?.aspect?.css || '1/1'};">
                                ${cat.image ? `
                                    <img src="${cat.image}" alt="${Templates.escapeHtml(cat.name)}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300">
                                ` : `
                                    <div class="w-full h-full flex items-center justify-center">
                                        <svg class="w-16 h-16 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
                                        </svg>
                                    </div>
                                `}
                            </div>
                            <div class="p-4 text-center">
                                <h3 class="font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">${Templates.escapeHtml(cat.name)}</h3>
                                ${cat.product_count ? `<p class="text-sm text-gray-500 mt-1">${cat.product_count} products</p>` : ''}
                            </div>
                        </a>
                    `).join('')}
                </div>
            `;
        } catch (error) {
            console.error('Failed to load categories:', error);
            resultsContainer.innerHTML = '<p class="text-red-500 text-center py-8">Failed to load categories. Please try again.</p>';
        }
    }

    function getQueryFromUrl() {
        const params = new URLSearchParams(window.location.search);
        return params.get('q') || '';
    }

    function getFiltersFromUrl() {
        const params = new URLSearchParams(window.location.search);
        const filters = {};

        if (params.get('category')) filters.category = params.get('category');
        if (params.get('min_price')) filters.minPrice = params.get('min_price');
        if (params.get('max_price')) filters.maxPrice = params.get('max_price');
        if (params.get('ordering')) filters.ordering = params.get('ordering');
        if (params.get('in_stock')) filters.inStock = params.get('in_stock') === 'true';
        if (params.get('sale')) filters.onSale = params.get('sale') === 'true';
        if (params.get('featured')) filters.featured = params.get('featured') === 'true';

        return filters;
    }

    function initSearchForm() {
        const searchForm = document.getElementById('search-form');
        const searchInput = document.getElementById('search-input');

        if (searchInput) {
            searchInput.value = currentQuery;
        }

        searchForm?.addEventListener('submit', (e) => {
            e.preventDefault();
            const query = searchInput?.value.trim();
            
            if (query) {
                currentQuery = query;
                currentPage = 1;
                updateUrl();
                performSearch();
            }
        });

        // Live search suggestions
        const suggestionsContainer = document.getElementById('search-suggestions');
        let debounceTimer = null;

        searchInput?.addEventListener('input', (e) => {
            const query = e.target.value.trim();
            
            clearTimeout(debounceTimer);

            if (query.length < 2) {
                if (suggestionsContainer) {
                    suggestionsContainer.innerHTML = '';
                    suggestionsContainer.classList.add('hidden');
                }
                return;
            }

            debounceTimer = setTimeout(async () => {
                try {
                    const response = await ProductsApi.search({ q: query, limit: 5 });
                    const products = response.data || [];

                    if (suggestionsContainer && products.length > 0) {
                        suggestionsContainer.innerHTML = `
                            <div class="py-2">
                                ${products.map(product => `
                                    <a href="/products/${product.slug}/" class="flex items-center gap-3 px-4 py-2 hover:bg-gray-50">
                                        ${product.image ? `<img src="${product.image}" alt="" class="w-10 h-10 object-cover rounded" onerror="this.style.display='none'">` : `<div class="w-10 h-10 bg-gray-100 rounded flex items-center justify-center"><svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg></div>`}
                                        <div>
                                            <p class="text-sm font-medium text-gray-900">${Templates.escapeHtml(product.name)}</p>
                                            <p class="text-sm text-primary-600">${Templates.formatPrice(product.current_price ?? product.price_converted ?? product.price)}</p>
                                        </div>
                                    </a>
                                `).join('')}
                            </div>
                        `;
                        suggestionsContainer.classList.remove('hidden');
                    }
                } catch (error) {
                    console.error('Search suggestions failed:', error);
                }
            }, 300);
        });

        // Hide suggestions on blur
        searchInput?.addEventListener('blur', () => {
            setTimeout(() => {
                if (suggestionsContainer) {
                    suggestionsContainer.classList.add('hidden');
                }
            }, 200);
        });
    }

    async function performSearch() {
        // Delegate to loadProducts which handles both search and browse
        await loadProducts();
    }

    function renderResults(products, meta) {
    }

    function renderResults(products, meta) {
        const container = document.getElementById('search-results');
        if (!container) return;

        if (products.length === 0) {
            container.innerHTML = `
                <div class="text-center py-16">
                    <svg class="w-24 h-24 text-gray-300 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                    </svg>
                    <h2 class="text-2xl font-bold text-gray-900 mb-2">No results found</h2>
                    <p class="text-gray-600 mb-4">We couldn't find any products matching "${Templates.escapeHtml(currentQuery)}"</p>
                    <p class="text-gray-500 text-sm">Try different keywords or browse our categories</p>
                </div>
            `;
            return;
        }

        const viewMode = Storage.get('productViewMode') || 'grid';
        const gridClass = viewMode === 'list' 
            ? 'space-y-4' 
            : 'grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6';

        container.innerHTML = `
            <div class="${gridClass}">
                ${products.map(product => ProductCard.render(product, { layout: viewMode })).join('')}
            </div>
            ${meta.total_pages > 1 ? `
                <div id="search-pagination" class="mt-8">${Pagination.render({
                    currentPage: meta.current_page || currentPage,
                    totalPages: meta.total_pages,
                    totalItems: meta.total
                })}</div>
            ` : ''}
        `;

        ProductCard.bindEvents(container);

        const paginationContainer = document.getElementById('search-pagination');
        paginationContainer?.addEventListener('click', (e) => {
            const pageBtn = e.target.closest('[data-page]');
            if (pageBtn) {
                currentPage = parseInt(pageBtn.dataset.page);
                updateUrl();
                performSearch();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });
    }

    async function loadFilterOptions() {
        const categoriesContainer = document.getElementById('filter-categories');
        if (!categoriesContainer) return;

        try {
            const response = await CategoriesAPI.getAll({ has_products: true, limit: 20 });
            const categories = response.data || [];

            categoriesContainer.innerHTML = `
                <div class="space-y-2">
                    <label class="flex items-center">
                        <input type="radio" name="category" value="" ${!currentFilters.category ? 'checked' : ''} class="text-primary-600 focus:ring-primary-500">
                        <span class="ml-2 text-sm text-gray-600">All Categories</span>
                    </label>
                    ${categories.map(cat => `
                        <label class="flex items-center">
                            <input type="radio" name="category" value="${cat.id}" ${currentFilters.category === String(cat.id) ? 'checked' : ''} class="text-primary-600 focus:ring-primary-500">
                            <span class="ml-2 text-sm text-gray-600">${Templates.escapeHtml(cat.name)}</span>
                        </label>
                    `).join('')}
                </div>
            `;

            bindCategoryFilter();
        } catch (error) {
            // error logging removed
        }
    }

    function bindCategoryFilter() {
        const categoryRadios = document.querySelectorAll('input[name="category"]');
        categoryRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.value) {
                    currentFilters.category = e.target.value;
                } else {
                    delete currentFilters.category;
                }
                currentPage = 1;
                updateUrl();
                performSearch();
            });
        });
    }

    function initFilters() {
        const applyPriceBtn = document.getElementById('apply-price-filter');
        const inStockCheckbox = document.getElementById('filter-in-stock');
        const clearFiltersBtn = document.getElementById('clear-filters');

        applyPriceBtn?.addEventListener('click', () => {
            const minPrice = document.getElementById('filter-min-price')?.value;
            const maxPrice = document.getElementById('filter-max-price')?.value;

            if (minPrice) currentFilters.min_price = minPrice;
            else delete currentFilters.min_price;
            
            if (maxPrice) currentFilters.max_price = maxPrice;
            else delete currentFilters.max_price;

            currentPage = 1;
            updateUrl();
            performSearch();
        });

        inStockCheckbox?.addEventListener('change', (e) => {
            if (e.target.checked) {
                currentFilters.in_stock = true;
            } else {
                delete currentFilters.in_stock;
            }
            currentPage = 1;
            updateUrl();
            performSearch();
        });

        clearFiltersBtn?.addEventListener('click', () => {
            currentFilters = {};
            currentPage = 1;
            
            document.querySelectorAll('input[name="category"]').forEach(r => {
                r.checked = r.value === '';
            });
            
            const minInput = document.getElementById('filter-min-price');
            const maxInput = document.getElementById('filter-max-price');
            if (minInput) minInput.value = '';
            if (maxInput) maxInput.value = '';
            
            if (inStockCheckbox) inStockCheckbox.checked = false;

            updateUrl();
            performSearch();
        });

        // Initialize filter values
        const minInput = document.getElementById('filter-min-price');
        const maxInput = document.getElementById('filter-max-price');
        if (minInput && currentFilters.min_price) minInput.value = currentFilters.min_price;
        if (maxInput && currentFilters.max_price) maxInput.value = currentFilters.max_price;
        if (inStockCheckbox && currentFilters.in_stock) inStockCheckbox.checked = true;
    }

    function initSorting() {
        const sortSelect = document.getElementById('sort-select');
        if (!sortSelect) return;

        sortSelect.value = currentFilters.ordering || '';

        sortSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                currentFilters.ordering = e.target.value;
            } else {
                delete currentFilters.ordering;
            }
            currentPage = 1;
            updateUrl();
            performSearch();
        });
    }

    function updateUrl() {
        const params = new URLSearchParams();
        
        if (currentQuery) params.set('q', currentQuery);
        if (currentFilters.category) params.set('category', currentFilters.category);
        if (currentFilters.min_price) params.set('min_price', currentFilters.min_price);
        if (currentFilters.max_price) params.set('max_price', currentFilters.max_price);
        if (currentFilters.ordering) params.set('ordering', currentFilters.ordering);
        if (currentFilters.in_stock) params.set('in_stock', 'true');
        if (currentPage > 1) params.set('page', currentPage);

        const newUrl = `${window.location.pathname}?${params.toString()}`;
        window.history.pushState({}, '', newUrl);
    }

    function destroy() {
        if (abortController) {
            abortController.abort();
            abortController = null;
        }
        currentQuery = '';
        currentPage = 1;
        currentFilters = {};
        initialized = false;
    }

    return {
        init,
        destroy
    };
})();

window.SearchPage = SearchPage;
export default SearchPage;
