/**
 * Bunoraa Frontend Core Module
 * Handles lazy loading, infinite scroll, dynamic content, and performance optimizations
 */

(function(window, document) {
    'use strict';

    // Configuration
    const CONFIG = {
        API_BASE: '/api/v1',
        LAZY_LOAD_THRESHOLD: '100px',
        INFINITE_SCROLL_THRESHOLD: 300,
        DEBOUNCE_DELAY: 150,
        CACHE_DURATION: 5 * 60 * 1000, // 5 minutes
        DEFAULT_CURRENCY: 'BDT',
        DEFAULT_LANGUAGE: 'bn'
    };

    // Cache for API responses
    const responseCache = new Map();

    // ============================================
    // Utility Functions
    // ============================================

    /**
     * Debounce function to limit function calls
     */
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Throttle function for scroll events
     */
    function throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    /**
     * Format currency based on locale
     */
    function formatCurrency(amount, currency = CONFIG.DEFAULT_CURRENCY) {
        // Use dynamic currency from window.BUNORAA_CURRENCY if available
        const currencyMeta = window.BUNORAA_CURRENCY || {};
        const activeCurrency = currency || currencyMeta.code || CONFIG.DEFAULT_CURRENCY;
        
        const currencySymbols = {
            'BDT': currencyMeta.symbol || '৳',
            'USD': '$',
            'EUR': '€',
            'INR': '₹'
        };

        const symbol = currencySymbols[activeCurrency] || currencyMeta.symbol || activeCurrency;
        const locale = currencyMeta.locale || (activeCurrency === 'BDT' ? 'bn-BD' : 'en-US');
        
        return `${symbol}${Number(amount).toLocaleString(locale)}`;
    }

    /**
     * Safe JSON parse with fallback
     */
    function safeJSONParse(str, fallback = null) {
        try {
            return JSON.parse(str);
        } catch (e) {
            return fallback;
        }
    }

    /**
     * Get CSRF token from cookie
     */
    function getCSRFToken() {
        const name = 'csrftoken';
        const cookies = document.cookie.split(';');
        for (let cookie of cookies) {
            cookie = cookie.trim();
            if (cookie.startsWith(name + '=')) {
                return cookie.substring(name.length + 1);
            }
        }
        return '';
    }

    // ============================================
    // API Client
    // ============================================

    const API = {
        /**
         * Make an API request with caching
         */
        async request(endpoint, options = {}) {
            const url = endpoint.startsWith('http') ? endpoint : `${CONFIG.API_BASE}${endpoint}`;
            const cacheKey = `${options.method || 'GET'}_${url}_${JSON.stringify(options.body || {})}`;

            // Check cache for GET requests
            if (!options.method || options.method === 'GET') {
                const cached = responseCache.get(cacheKey);
                if (cached && Date.now() - cached.timestamp < CONFIG.CACHE_DURATION) {
                    return cached.data;
                }
            }

            const headers = {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken(),
                ...options.headers
            };

            try {
                const response = await fetch(url, {
                    ...options,
                    headers,
                    credentials: 'same-origin'
                });

                if (!response.ok) {
                    throw new Error(`API Error: ${response.status}`);
                }

                const data = await response.json();

                // Cache GET responses
                if (!options.method || options.method === 'GET') {
                    responseCache.set(cacheKey, {
                        data,
                        timestamp: Date.now()
                    });
                }

                return data;
            } catch (error) {
                console.error('API Request failed:', error);
                throw error;
            }
        },

        get(endpoint) {
            return this.request(endpoint, { method: 'GET' });
        },

        post(endpoint, data) {
            return this.request(endpoint, {
                method: 'POST',
                body: JSON.stringify(data)
            });
        },

        put(endpoint, data) {
            return this.request(endpoint, {
                method: 'PUT',
                body: JSON.stringify(data)
            });
        },

        delete(endpoint) {
            return this.request(endpoint, { method: 'DELETE' });
        }
    };

    // ============================================
    // Lazy Loading Module
    // ============================================

    const LazyLoader = {
        observer: null,
        priorityObserver: null,
        videoObserver: null,
        initialized: false,

        init() {
            if (this.initialized) return;
            this.initialized = true;

            this.ensureLazyImages();

            if ('IntersectionObserver' in window) {
                // Main lazy load observer with default threshold
                this.observer = new IntersectionObserver(
                    this.handleIntersection.bind(this),
                    {
                        rootMargin: CONFIG.LAZY_LOAD_THRESHOLD,
                        threshold: 0.01
                    }
                );

                // Priority observer for above-the-fold content
                this.priorityObserver = new IntersectionObserver(
                    this.handlePriorityIntersection.bind(this),
                    {
                        rootMargin: '0px',
                        threshold: 0.01
                    }
                );

                // Video observer with larger threshold
                this.videoObserver = new IntersectionObserver(
                    this.handleVideoIntersection.bind(this),
                    {
                        rootMargin: '50px 0px',
                        threshold: 0.25
                    }
                );

                this.observeElements();
                this.observeVideos();
                this.setupMutationObserver();
            } else {
                // Fallback for older browsers
                this.loadAllImages();
            }
        },

        ensureLazyImages(root = document) {
            root.querySelectorAll('img').forEach(img => {
                img.loading = 'lazy';
            });
        },

        observeElements() {
            // Lazy load images
            document.querySelectorAll('img[data-src], img[loading="lazy"]').forEach(img => {
                if (img.dataset.src) {
                    // Check if element is above the fold
                    if (this.isAboveFold(img)) {
                        this.priorityObserver.observe(img);
                    } else {
                        this.observer.observe(img);
                    }
                }
            });

            // Lazy load iframes
            document.querySelectorAll('iframe[data-src]').forEach(iframe => {
                this.observer.observe(iframe);
            });

            // Lazy load background images
            document.querySelectorAll('[data-bg]').forEach(el => {
                this.observer.observe(el);
            });

            // Lazy load components
            document.querySelectorAll('[data-lazy-component]').forEach(el => {
                this.observer.observe(el);
            });

            // Lazy load sections
            document.querySelectorAll('[data-lazy-section]').forEach(el => {
                this.observer.observe(el);
            });

            // Lazy load animations
            document.querySelectorAll('[data-animate-on-scroll]').forEach(el => {
                this.observer.observe(el);
            });
        },

        observeVideos() {
            // Observe videos for autoplay when in view
            document.querySelectorAll('video[data-autoplay-on-view], video[data-lazy-video]').forEach(video => {
                this.videoObserver.observe(video);
            });
        },

        isAboveFold(el) {
            const rect = el.getBoundingClientRect();
            return rect.top < window.innerHeight && rect.bottom > 0;
        },

        setupMutationObserver() {
            // Watch for dynamically added content
            const mutationObserver = new MutationObserver((mutations) => {
                let hasNewElements = false;
                mutations.forEach((mutation) => {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            this.ensureLazyImages(node);
                            // Check if node itself needs lazy loading
                            if (node.matches && (
                                node.matches('img[data-src]') ||
                                node.matches('[data-bg]') ||
                                node.matches('[data-lazy-component]') ||
                                node.matches('[data-lazy-section]') ||
                                node.matches('[data-animate-on-scroll]')
                            )) {
                                hasNewElements = true;
                            }
                            // Check children
                            if (node.querySelectorAll) {
                                const lazyElements = node.querySelectorAll(
                                    'img[data-src], [data-bg], [data-lazy-component], [data-lazy-section], [data-animate-on-scroll], video[data-autoplay-on-view]'
                                );
                                if (lazyElements.length > 0) {
                                    hasNewElements = true;
                                }
                            }
                        }
                    });
                });
                if (hasNewElements) {
                    this.observeElements();
                    this.observeVideos();
                }
            });

            mutationObserver.observe(document.body, {
                childList: true,
                subtree: true
            });
        },

        handlePriorityIntersection(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const el = entry.target;
                    if (el.tagName === 'IMG') {
                        this.loadImage(el, true); // priority load
                    }
                    this.priorityObserver.unobserve(el);
                }
            });
        },

        handleIntersection(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const el = entry.target;

                    if (el.tagName === 'IMG') {
                        this.loadImage(el);
                    } else if (el.tagName === 'IFRAME') {
                        this.loadIframe(el);
                    } else if (el.dataset.bg) {
                        this.loadBackgroundImage(el);
                    } else if (el.dataset.lazyComponent) {
                        this.loadComponent(el);
                    } else if (el.dataset.lazySection) {
                        this.loadSection(el);
                    } else if (el.dataset.animateOnScroll) {
                        this.animateElement(el);
                    }

                    this.observer.unobserve(el);
                }
            });
        },

        handleVideoIntersection(entries) {
            entries.forEach(entry => {
                const video = entry.target;
                
                if (entry.isIntersecting) {
                    // Load video source if lazy
                    if (video.dataset.lazySrc) {
                        video.src = video.dataset.lazySrc;
                        video.removeAttribute('data-lazy-src');
                        video.load();
                    }
                    
                    // Autoplay when in view
                    if (video.dataset.autoplayOnView !== undefined) {
                        video.play().catch(() => {
                            // Autoplay blocked, ignore
                        });
                    }
                } else {
                    // Pause when out of view
                    if (video.dataset.autoplayOnView !== undefined && !video.paused) {
                        video.pause();
                    }
                }
            });
        },

        loadImage(img, priority = false) {
            const src = img.dataset.src;
            const srcset = img.dataset.srcset;

            if (!src) return;

            img.classList.add('loading');

            // Set priority hints for above-the-fold images
            if (priority && 'fetchPriority' in HTMLImageElement.prototype) {
                img.fetchPriority = 'high';
            }

            img.onload = () => {
                img.classList.remove('loading');
                img.classList.add('loaded');
                img.removeAttribute('data-src');
                img.removeAttribute('data-srcset');
                img.dispatchEvent(new CustomEvent('lazyloaded', { bubbles: true }));
            };
            img.onerror = () => {
                img.classList.remove('loading');
                img.classList.add('error');
                img.style.visibility = 'hidden';
                console.warn('Failed to load image:', src);
            };

            img.src = src;
            if (srcset) img.srcset = srcset;
        },

        loadIframe(iframe) {
            const src = iframe.dataset.src;
            if (!src) return;
            
            iframe.classList.add('loading');
            iframe.onload = () => {
                iframe.classList.remove('loading');
                iframe.classList.add('loaded');
            };
            iframe.src = src;
            iframe.removeAttribute('data-src');
        },

        loadBackgroundImage(el) {
            const bg = el.dataset.bg;
            if (!bg) return;

            el.style.backgroundImage = `url(${bg})`;
            el.classList.add('bg-loaded');
            el.removeAttribute('data-bg');
        },

        async loadComponent(el) {
            const component = el.dataset.lazyComponent;
            el.classList.add('loading');

            try {
                const response = await API.get(`/components/${component}/`);
                el.innerHTML = response.html;
                el.classList.remove('loading');
                el.classList.add('loaded');

                // Initialize any scripts in the component
                this.executeComponentScripts(el);

                // Re-observe any lazy elements in the new content
                this.observeElements();
            } catch (error) {
                el.classList.remove('loading');
                el.classList.add('error');
                console.error('Component load failed:', component, error);
            }
        },

        async loadSection(el) {
            const sectionUrl = el.dataset.lazySection;
            el.classList.add('loading');

            try {
                const response = await fetch(sectionUrl);
                if (!response.ok) throw new Error('Failed to load section');
                
                const html = await response.text();
                el.innerHTML = html;
                el.classList.remove('loading');
                el.classList.add('loaded');
                el.removeAttribute('data-lazy-section');

                // Initialize any scripts in the section
                this.executeComponentScripts(el);

                // Re-observe any lazy elements in the new content
                this.observeElements();
                this.observeVideos();
            } catch (error) {
                el.classList.remove('loading');
                el.classList.add('error');
                console.error('Section load failed:', sectionUrl, error);
            }
        },

        animateElement(el) {
            const animation = el.dataset.animateOnScroll || 'fade-in';
            const delay = parseInt(el.dataset.animateDelay) || 0;
            
            setTimeout(() => {
                el.classList.add('animated', animation);
                el.removeAttribute('data-animate-on-scroll');
            }, delay);
        },

        executeComponentScripts(container) {
            const scripts = container.querySelectorAll('script');
            scripts.forEach(script => {
                const newScript = document.createElement('script');
                if (script.src) {
                    newScript.src = script.src;
                } else {
                    newScript.textContent = script.textContent;
                }
                script.parentNode.replaceChild(newScript, script);
            });
        },

        loadAllImages() {
            // For browsers without IntersectionObserver
            document.querySelectorAll('img[data-src]').forEach(img => {
                img.src = img.dataset.src;
            });
        },

        // Public API: manually observe an element
        observe(el) {
            if (this.observer && el) {
                this.observer.observe(el);
            }
        },

        // Public API: refresh observers for new content
        refresh() {
            this.observeElements();
            this.observeVideos();
        }
    };

    // ============================================
    // Infinite Scroll Module
    // ============================================

    const InfiniteScroll = {
        containers: new Map(),

        init() {
            document.querySelectorAll('[data-infinite-scroll]').forEach(container => {
                this.initContainer(container);
            });
        },

        initContainer(container) {
            const config = {
                endpoint: container.dataset.infiniteScroll,
                page: 1,
                loading: false,
                hasMore: true,
                itemsContainer: container.querySelector('[data-items]') || container,
                loadingIndicator: container.querySelector('[data-loading]'),
                noMoreIndicator: container.querySelector('[data-no-more]')
            };

            this.containers.set(container, config);

            // Initial load if needed
            if (container.dataset.autoload !== 'false') {
                this.loadMore(container);
            }

            // Scroll event listener
            const scrollHandler = throttle(() => {
                this.checkScroll(container);
            }, 100);

            window.addEventListener('scroll', scrollHandler);

            // Button click handler
            const loadMoreBtn = container.querySelector('[data-load-more]');
            if (loadMoreBtn) {
                loadMoreBtn.addEventListener('click', () => this.loadMore(container));
            }
        },

        checkScroll(container) {
            const config = this.containers.get(container);
            if (!config || config.loading || !config.hasMore) return;

            const containerRect = container.getBoundingClientRect();
            const bottomDistance = containerRect.bottom - window.innerHeight;

            if (bottomDistance < CONFIG.INFINITE_SCROLL_THRESHOLD) {
                this.loadMore(container);
            }
        },

        async loadMore(container) {
            const config = this.containers.get(container);
            if (!config || config.loading || !config.hasMore) return;

            config.loading = true;
            this.showLoading(container, true);

            try {
                config.page++;
                const response = await API.get(`${config.endpoint}?page=${config.page}`);

                if (response.results && response.results.length > 0) {
                    this.appendItems(container, response.results);
                    config.hasMore = !!response.next;
                } else {
                    config.hasMore = false;
                }

                if (!config.hasMore) {
                    this.showNoMore(container);
                }
            } catch (error) {
                config.page--;
                console.error('Failed to load more items:', error);
            } finally {
                config.loading = false;
                this.showLoading(container, false);
            }
        },

        appendItems(container, items) {
            const config = this.containers.get(container);
            const template = container.querySelector('template[data-item-template]');

            items.forEach(item => {
                const html = this.renderItem(template, item);
                config.itemsContainer.insertAdjacentHTML('beforeend', html);
            });

            // Re-init lazy loading for new images
            LazyLoader.observeElements();

            // Dispatch event
            container.dispatchEvent(new CustomEvent('items-loaded', { detail: { items } }));
        },

        renderItem(template, item) {
            if (!template) {
                return `<div class="item">${JSON.stringify(item)}</div>`;
            }

            let html = template.innerHTML;
            Object.keys(item).forEach(key => {
                const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
                html = html.replace(regex, item[key] ?? '');
            });
            return html;
        },

        showLoading(container, show) {
            const config = this.containers.get(container);
            if (config.loadingIndicator) {
                config.loadingIndicator.hidden = !show;
            }
        },

        showNoMore(container) {
            const config = this.containers.get(container);
            if (config.noMoreIndicator) {
                config.noMoreIndicator.hidden = false;
            }
        }
    };

    // ============================================
    // Live Search Module
    // ============================================

    const LiveSearch = {
        searchInput: null,
        resultsContainer: null,
        debounceTimer: null,
        minChars: 2,

        init() {
            this.searchInput = document.querySelector('[data-live-search]');
            if (!this.searchInput) return;

            this.resultsContainer = document.querySelector(
                this.searchInput.dataset.liveSearch || '[data-search-results]'
            );
            this.minChars = parseInt(this.searchInput.dataset.minChars) || 2;

            this.bindEvents();
        },

        bindEvents() {
            this.searchInput.addEventListener('input', debounce((e) => {
                this.handleInput(e.target.value);
            }, CONFIG.DEBOUNCE_DELAY));

            this.searchInput.addEventListener('focus', () => {
                if (this.searchInput.value.length >= this.minChars) {
                    this.showResults();
                }
            });

            // Close on outside click
            document.addEventListener('click', (e) => {
                if (!this.searchInput.contains(e.target) && 
                    !this.resultsContainer?.contains(e.target)) {
                    this.hideResults();
                }
            });

            // Keyboard navigation
            this.searchInput.addEventListener('keydown', (e) => {
                this.handleKeyboard(e);
            });
        },

        async handleInput(query) {
            if (query.length < this.minChars) {
                this.hideResults();
                return;
            }

            this.showLoading();

            try {
                const response = await API.get(`/search/suggestions/?q=${encodeURIComponent(query)}`);
                this.renderResults(response.results || response);
            } catch (error) {
                this.showError();
            }
        },

        renderResults(results) {
            if (!this.resultsContainer) return;

            if (!results || results.length === 0) {
                this.resultsContainer.innerHTML = `
                    <div class="search-no-results">
                        <p>No results found</p>
                    </div>
                `;
            } else {
                const html = results.map((item, index) => `
                    <a href="${item.url}" 
                       class="search-result-item" 
                       data-index="${index}"
                       role="option">
                        ${item.image ? `<img src="${item.image}" alt="" loading="lazy">` : ''}
                        <div class="search-result-content">
                            <span class="search-result-title">${item.name || item.title}</span>
                            ${item.price ? `<span class="search-result-price">${formatCurrency(item.price)}</span>` : ''}
                            ${item.category ? `<span class="search-result-category">${item.category}</span>` : ''}
                        </div>
                    </a>
                `).join('');

                this.resultsContainer.innerHTML = html;
            }

            this.showResults();
        },

        showResults() {
            if (this.resultsContainer) {
                this.resultsContainer.hidden = false;
                this.resultsContainer.setAttribute('aria-expanded', 'true');
            }
        },

        hideResults() {
            if (this.resultsContainer) {
                this.resultsContainer.hidden = true;
                this.resultsContainer.setAttribute('aria-expanded', 'false');
            }
        },

        showLoading() {
            if (this.resultsContainer) {
                this.resultsContainer.innerHTML = `
                    <div class="search-loading" role="status">
                        <span class="spinner"></span>
                        <span>Searching...</span>
                    </div>
                `;
                this.showResults();
            }
        },

        showError() {
            if (this.resultsContainer) {
                this.resultsContainer.innerHTML = `
                    <div class="search-error" role="alert">
                        <p>Search failed. Please try again.</p>
                    </div>
                `;
            }
        },

        handleKeyboard(e) {
            const items = this.resultsContainer?.querySelectorAll('.search-result-item');
            if (!items || items.length === 0) return;

            const current = this.resultsContainer.querySelector('.search-result-item.highlighted');
            let index = current ? parseInt(current.dataset.index) : -1;

            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    index = Math.min(index + 1, items.length - 1);
                    this.highlightItem(items, index);
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    index = Math.max(index - 1, 0);
                    this.highlightItem(items, index);
                    break;
                case 'Enter':
                    if (current) {
                        e.preventDefault();
                        window.location.href = current.href;
                    }
                    break;
                case 'Escape':
                    this.hideResults();
                    break;
            }
        },

        highlightItem(items, index) {
            items.forEach((item, i) => {
                item.classList.toggle('highlighted', i === index);
            });
            if (items[index]) {
                items[index].scrollIntoView({ block: 'nearest' });
            }
        }
    };

    // ============================================
    // Cart Module
    // ============================================

    const Cart = {
        items: [],
        count: 0,

        init() {
            this.bindEvents();
            this.loadCart();
        },

        bindEvents() {
            // Add to cart buttons
            document.addEventListener('click', (e) => {
                const addBtn = e.target.closest('[data-add-to-cart]');
                if (addBtn) {
                    e.preventDefault();
                    this.addItem(addBtn.dataset.addToCart, addBtn.dataset);
                }

                const removeBtn = e.target.closest('[data-remove-from-cart]');
                if (removeBtn) {
                    e.preventDefault();
                    this.removeItem(removeBtn.dataset.removeFromCart);
                }
            });

            // Quantity changes
            document.addEventListener('change', (e) => {
                const qtyInput = e.target.closest('[data-cart-quantity]');
                if (qtyInput) {
                    this.updateQuantity(qtyInput.dataset.cartQuantity, qtyInput.value);
                }
            });
        },

        async loadCart() {
            try {
                const response = await API.get('/commerce/cart/');
                this.items = response.items || [];
                this.count = response.count || this.items.length;
                this.updateUI();
            } catch (error) {
                console.error('Failed to load cart:', error);
            }
        },

        async addItem(productId, options = {}) {
            try {
                const response = await API.post('/commerce/cart/add/', {
                    product_id: productId,
                    quantity: options.quantity || 1,
                    ...options
                });

                this.items = response.items || this.items;
                this.count = response.count || this.count + 1;
                this.updateUI();
                this.showNotification('Item added to cart!');

                // Track for analytics
                this.trackAddToCart(productId, options);
            } catch (error) {
                this.showNotification('Failed to add item', 'error');
            }
        },

        async removeItem(productId) {
            try {
                const response = await API.post('/commerce/cart/remove/', {
                    product_id: productId
                });

                this.items = response.items || this.items;
                this.count = response.count || Math.max(0, this.count - 1);
                this.updateUI();
                this.showNotification('Item removed from cart');
            } catch (error) {
                this.showNotification('Failed to remove item', 'error');
            }
        },

        async updateQuantity(productId, quantity) {
            try {
                const response = await API.post('/commerce/cart/update/', {
                    product_id: productId,
                    quantity: parseInt(quantity)
                });

                this.items = response.items || this.items;
                this.updateUI();
            } catch (error) {
                this.showNotification('Failed to update quantity', 'error');
            }
        },

        updateUI() {
            // Update cart count badges
            document.querySelectorAll('[data-cart-count]').forEach(el => {
                el.textContent = this.count;
                el.hidden = this.count === 0;
            });

            // Update mini cart
            const miniCart = document.querySelector('[data-mini-cart]');
            if (miniCart) {
                this.renderMiniCart(miniCart);
            }
        },

        renderMiniCart(container) {
            if (this.items.length === 0) {
                container.innerHTML = `
                    <div class="mini-cart-empty">
                        <p>Your cart is empty</p>
                        <a href="/products/" class="btn btn-primary">Start Shopping</a>
                    </div>
                `;
                return;
            }

            const total = this.items.reduce((sum, item) => 
                sum + (item.price * item.quantity), 0);

            container.innerHTML = `
                <div class="mini-cart-items">
                    ${this.items.slice(0, 3).map(item => `
                        <div class="mini-cart-item">
                            ${item.image ? `<img src="${item.image}" alt="${item.name}" loading="lazy">` : ''}
                            <div class="mini-cart-item-details">
                                <span class="mini-cart-item-name">${item.name}</span>
                                <span class="mini-cart-item-price">
                                    ${item.quantity} × ${formatCurrency(item.price)}
                                </span>
                            </div>
                        </div>
                    `).join('')}
                    ${this.items.length > 3 ? `
                        <p class="mini-cart-more">
                            +${this.items.length - 3} more items
                        </p>
                    ` : ''}
                </div>
                <div class="mini-cart-footer">
                    <div class="mini-cart-total">
                        <span>Total:</span>
                        <strong>${formatCurrency(total)}</strong>
                    </div>
                    <a href="/cart/" class="btn btn-secondary">View Cart</a>
                    <a href="/checkout/" class="btn btn-primary">Checkout</a>
                </div>
            `;
        },

        showNotification(message, type = 'success') {
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.setAttribute('role', 'alert');
            notification.textContent = message;

            document.body.appendChild(notification);

            // Animate in
            requestAnimationFrame(() => {
                notification.classList.add('show');
            });

            // Remove after delay
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => notification.remove(), 300);
            }, 3000);
        },

        trackAddToCart(productId, options) {
            // Send to analytics
            if (window.BunoraAnalytics) {
                window.BunoraAnalytics.track('add_to_cart', {
                    product_id: productId,
                    quantity: options.quantity || 1
                });
            }
        }
    };

    // ============================================
    // Wishlist Module
    // ============================================

    const Wishlist = {
        items: new Set(),

        init() {
            this.loadWishlist();
            this.bindEvents();
        },

        bindEvents() {
            document.addEventListener('click', (e) => {
                const wishlistBtn = e.target.closest('[data-wishlist-toggle]');
                if (wishlistBtn) {
                    e.preventDefault();
                    this.toggleItem(wishlistBtn.dataset.wishlistToggle, wishlistBtn);
                }
            });
        },

        async loadWishlist() {
            try {
                const response = await API.get('/wishlist/');
                this.items = new Set(response.items?.map(item => item.product_id) || []);
                this.updateUI();
            } catch (error) {
                console.error('Failed to load wishlist:', error);
            }
        },

        async toggleItem(productId, button) {
            const isInWishlist = this.items.has(productId);

            try {
                if (isInWishlist) {
                    await API.post('/wishlist/remove/', { product_id: productId });
                    this.items.delete(productId);
                } else {
                    await API.post('/wishlist/add/', { product_id: productId });
                    this.items.add(productId);
                }

                this.updateButton(button, !isInWishlist);
                this.updateUI();
            } catch (error) {
                console.error('Wishlist error:', error);
            }
        },

        updateButton(button, isInWishlist) {
            button.classList.toggle('active', isInWishlist);
            button.setAttribute('aria-pressed', isInWishlist);
            button.querySelector('.wishlist-icon')?.classList.toggle('filled', isInWishlist);
        },

        updateUI() {
            // Update all wishlist buttons
            document.querySelectorAll('[data-wishlist-toggle]').forEach(btn => {
                const productId = btn.dataset.wishlistToggle;
                const isInWishlist = this.items.has(productId);
                this.updateButton(btn, isInWishlist);
            });

            // Update wishlist count
            document.querySelectorAll('[data-wishlist-count]').forEach(el => {
                el.textContent = this.items.size;
                el.hidden = this.items.size === 0;
            });
        }
    };

    // ============================================
    // Theme Module (Dark/Light Mode)
    // ============================================

    const Theme = {
        STORAGE_KEY: 'bunoraa_theme',

        init() {
            this.applyTheme(this.getTheme());
            this.bindEvents();
        },

        getTheme() {
            const stored = localStorage.getItem(this.STORAGE_KEY);
            if (stored) return stored;

            // Check system preference
            if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
                return 'dark';
            }
            return 'light';
        },

        setTheme(theme) {
            localStorage.setItem(this.STORAGE_KEY, theme);
            this.applyTheme(theme);
        },

        applyTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            document.body.classList.remove('theme-light', 'theme-dark');
            document.body.classList.add(`theme-${theme}`);

            // Update meta theme-color
            const metaThemeColor = document.querySelector('meta[name="theme-color"]');
            if (metaThemeColor) {
                metaThemeColor.content = theme === 'dark' ? '#1a1a1a' : '#ffffff';
            }
        },

        toggle() {
            const current = this.getTheme();
            this.setTheme(current === 'dark' ? 'light' : 'dark');
        },

        bindEvents() {
            document.querySelectorAll('[data-theme-toggle]').forEach(btn => {
                btn.addEventListener('click', () => this.toggle());
            });

            // Listen for system preference changes
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
                if (!localStorage.getItem(this.STORAGE_KEY)) {
                    this.applyTheme(e.matches ? 'dark' : 'light');
                }
            });
        }
    };

    // ============================================
    // Analytics Module
    // ============================================

    const Analytics = {
        sessionId: null,

        init() {
            this.sessionId = this.getSessionId();
            this.trackPageView();
            this.setupEventTracking();
        },

        getSessionId() {
            let id = sessionStorage.getItem('bunoraa_session');
            if (!id) {
                id = 'ses_' + Math.random().toString(36).substring(2);
                sessionStorage.setItem('bunoraa_session', id);
            }
            return id;
        },

        track(eventType, data = {}) {
            const payload = {
                event_type: eventType,
                session_id: this.sessionId,
                timestamp: new Date().toISOString(),
                url: window.location.href,
                referrer: document.referrer,
                ...data
            };

            // Use sendBeacon for reliability
            if (navigator.sendBeacon) {
                navigator.sendBeacon('/api/v1/analytics/track/', JSON.stringify(payload));
            } else {
                // Fallback
                API.post('/analytics/track/', payload).catch(() => {});
            }
        },

        trackPageView() {
            this.track('page_view', {
                title: document.title,
                path: window.location.pathname
            });
        },

        setupEventTracking() {
            // Track product clicks
            document.addEventListener('click', (e) => {
                const productCard = e.target.closest('[data-product-id]');
                if (productCard) {
                    this.track('product_click', {
                        product_id: productCard.dataset.productId,
                        product_name: productCard.dataset.productName
                    });
                }
            });

            // Track scroll depth
            let maxScroll = 0;
            window.addEventListener('scroll', throttle(() => {
                const scrollPercent = Math.round(
                    (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100
                );
                if (scrollPercent > maxScroll) {
                    maxScroll = scrollPercent;
                    if ([25, 50, 75, 100].includes(scrollPercent)) {
                        this.track('scroll_depth', { depth: scrollPercent });
                    }
                }
            }, 500));

            // Track time on page
            window.addEventListener('beforeunload', () => {
                const timeOnPage = Math.round((Date.now() - performance.now()) / 1000);
                this.track('time_on_page', { seconds: timeOnPage });
            });
        }
    };

    // ============================================
    // Initialize
    // ============================================

    function init() {
        LazyLoader.init();
        InfiniteScroll.init();
        LiveSearch.init();
        Cart.init();
        Wishlist.init();
        Theme.init();
        Analytics.init();

        // Expose for external use
        window.Bunoraa = {
            API,
            LazyLoader,
            InfiniteScroll,
            LiveSearch,
            Cart,
            Wishlist,
            Theme,
            Analytics,
            formatCurrency
        };

        window.BunoraAnalytics = Analytics;

        // Dispatch ready event
        document.dispatchEvent(new CustomEvent('bunoraa:ready'));
    }

    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})(window, document);
