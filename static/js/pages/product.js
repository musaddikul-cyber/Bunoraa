/**
 * Product Detail Page - Enhanced with Advanced E-commerce Features
 * @module pages/product
 */

const ProductPage = (function() {
    'use strict';

    let currentProduct = null;
    let selectedVariant = null;
    let gallery = null;
    let initialized = false;
    let serverRendered = false;
    let zoomEnabled = false;
    let stockAlertEmail = null;

    async function init() {
        // Prevent multiple initializations
        if (initialized) return;
        initialized = true;

        const container = document.getElementById('product-detail');
        if (!container) return;

        // Check if content is already server-rendered
        const isServerRendered = container.querySelector('h1') || container.dataset.productId;
        serverRendered = Boolean(isServerRendered);
        
        if (isServerRendered) {
            // Server-rendered content - just bind event handlers
            currentProduct = {
                id: container.dataset.productId,
                slug: container.dataset.productSlug
            };
            bindExistingEvents();
            initEnhancedFeatures();
            return;
        }

        // Dynamic loading for SPA navigation
        const productSlug = getProductSlugFromUrl();
        if (!productSlug) {
            window.location.href = '/products/';
            return;
        }

        await loadProduct(productSlug);
        initEnhancedFeatures();
    }

    // ============================================
    // ENHANCED FEATURES INITIALIZATION
    // ============================================
    function initEnhancedFeatures() {
        initImageZoom();
        initSizeGuideModal();
        initStockAlerts();
        initSocialSharing();
        initQASection();
        initDeliveryEstimate();
        initRecentlyViewedTracking();
        initStickyAddToCart();
        initProductVideoPlayer();
    }

    // ============================================
    // ENHANCED FEATURE: Image Zoom on Hover
    // ============================================
    function initImageZoom() {
        const mainImage = document.getElementById('main-product-image') || document.getElementById('main-image');
        const container = mainImage?.parentElement;
        
        if (!mainImage || !container) return;

        // Create zoom lens
        const lens = document.createElement('div');
        lens.className = 'zoom-lens absolute w-32 h-32 border-2 border-primary-500 bg-white/30 pointer-events-none opacity-0 transition-opacity duration-200 z-10';
        lens.style.backgroundRepeat = 'no-repeat';
        
        // Create zoom result container
        const result = document.createElement('div');
        result.className = 'zoom-result fixed right-8 top-1/2 -translate-y-1/2 w-96 h-96 border border-stone-200 dark:border-stone-700 rounded-xl shadow-2xl bg-white dark:bg-stone-800 opacity-0 pointer-events-none transition-opacity duration-200 z-50 hidden lg:block';
        result.style.backgroundRepeat = 'no-repeat';
        
        container.classList.add('relative');
        container.appendChild(lens);
        document.body.appendChild(result);

        container.addEventListener('mouseenter', () => {
            if (window.innerWidth < 1024) return;
            lens.classList.remove('opacity-0');
            result.classList.remove('opacity-0');
            zoomEnabled = true;
        });

        container.addEventListener('mouseleave', () => {
            lens.classList.add('opacity-0');
            result.classList.add('opacity-0');
            zoomEnabled = false;
        });

        container.addEventListener('mousemove', (e) => {
            if (!zoomEnabled || window.innerWidth < 1024) return;

            const rect = container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Position lens
            const lensX = x - lens.offsetWidth / 2;
            const lensY = y - lens.offsetHeight / 2;
            
            lens.style.left = `${Math.max(0, Math.min(rect.width - lens.offsetWidth, lensX))}px`;
            lens.style.top = `${Math.max(0, Math.min(rect.height - lens.offsetHeight, lensY))}px`;

            // Calculate zoom
            const zoomLevel = 3;
            const bgX = -x * zoomLevel + result.offsetWidth / 2;
            const bgY = -y * zoomLevel + result.offsetHeight / 2;
            
            result.style.backgroundImage = `url(${mainImage.src})`;
            result.style.backgroundSize = `${rect.width * zoomLevel}px ${rect.height * zoomLevel}px`;
            result.style.backgroundPosition = `${bgX}px ${bgY}px`;
        });
    }

    // ============================================
    // ENHANCED FEATURE: Size Guide Modal
    // ============================================
    function initSizeGuideModal() {
        const sizeGuideBtn = document.getElementById('size-guide-btn');
        if (!sizeGuideBtn) return;

        sizeGuideBtn.addEventListener('click', () => {
            const modal = document.createElement('div');
            modal.id = 'size-guide-modal';
            modal.className = 'fixed inset-0 z-50 flex items-center justify-center p-4';
            modal.innerHTML = `
                <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('size-guide-modal').remove()"></div>
                <div class="relative bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-auto">
                    <div class="sticky top-0 bg-white dark:bg-stone-800 border-b border-stone-200 dark:border-stone-700 p-4 flex items-center justify-between">
                        <h2 class="text-xl font-bold text-stone-900 dark:text-white">Size Guide</h2>
                        <button onclick="document.getElementById('size-guide-modal').remove()" class="w-10 h-10 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors">
                            <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                        </button>
                    </div>
                    <div class="p-6">
                        <div class="mb-6">
                            <h3 class="text-lg font-semibold text-stone-900 dark:text-white mb-2">How to Measure</h3>
                            <p class="text-stone-600 dark:text-stone-400 text-sm">Use a flexible measuring tape for accurate measurements. Measure over your undergarments for best results.</p>
                        </div>
                        <div class="overflow-x-auto">
                            <table class="w-full text-sm">
                                <thead>
                                    <tr class="bg-stone-50 dark:bg-stone-700">
                                        <th class="px-4 py-3 text-left font-semibold text-stone-900 dark:text-white">Size</th>
                                        <th class="px-4 py-3 text-center font-semibold text-stone-900 dark:text-white">Chest (in)</th>
                                        <th class="px-4 py-3 text-center font-semibold text-stone-900 dark:text-white">Waist (in)</th>
                                        <th class="px-4 py-3 text-center font-semibold text-stone-900 dark:text-white">Length (in)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr class="border-b border-stone-100 dark:border-stone-600">
                                        <td class="px-4 py-3 font-medium text-stone-900 dark:text-white">XS</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">32-34</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">26-28</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">26</td>
                                    </tr>
                                    <tr class="border-b border-stone-100 dark:border-stone-600">
                                        <td class="px-4 py-3 font-medium text-stone-900 dark:text-white">S</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">35-37</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">29-31</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">27</td>
                                    </tr>
                                    <tr class="border-b border-stone-100 dark:border-stone-600">
                                        <td class="px-4 py-3 font-medium text-stone-900 dark:text-white">M</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">38-40</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">32-34</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">28</td>
                                    </tr>
                                    <tr class="border-b border-stone-100 dark:border-stone-600">
                                        <td class="px-4 py-3 font-medium text-stone-900 dark:text-white">L</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">41-43</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">35-37</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">29</td>
                                    </tr>
                                    <tr>
                                        <td class="px-4 py-3 font-medium text-stone-900 dark:text-white">XL</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">44-46</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">38-40</td>
                                        <td class="px-4 py-3 text-center text-stone-600 dark:text-stone-400">30</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        <div class="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-xl">
                            <p class="text-sm text-amber-800 dark:text-amber-200">💡 <strong>Tip:</strong> If you're between sizes, we recommend sizing up for a more comfortable fit.</p>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        });
    }

    // ============================================
    // ENHANCED FEATURE: Stock Alerts
    // ============================================
    function initStockAlerts() {
        const stockAlertBtn = document.getElementById('stock-alert-btn');
        if (!stockAlertBtn) return;

        stockAlertBtn.addEventListener('click', () => {
            const productId = document.getElementById('product-detail')?.dataset.productId;
            if (!productId) return;

            const modal = document.createElement('div');
            modal.id = 'stock-alert-modal';
            modal.className = 'fixed inset-0 z-50 flex items-center justify-center p-4';
            modal.innerHTML = `
                <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('stock-alert-modal').remove()"></div>
                <div class="relative bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-md w-full p-6">
                    <button onclick="document.getElementById('stock-alert-modal').remove()" class="absolute top-4 right-4 w-8 h-8 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors">
                        <svg class="w-4 h-4 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                    <div class="text-center mb-6">
                        <div class="w-16 h-16 bg-primary-100 dark:bg-amber-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                            <svg class="w-8 h-8 text-primary-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"/></svg>
                        </div>
                        <h3 class="text-xl font-bold text-stone-900 dark:text-white mb-2">Notify Me When Available</h3>
                        <p class="text-stone-600 dark:text-stone-400 text-sm">We'll email you when this product is back in stock.</p>
                    </div>
                    <form id="stock-alert-form" class="space-y-4">
                        <input type="email" id="stock-alert-email" placeholder="Enter your email" required class="w-full px-4 py-3 border border-stone-300 dark:border-stone-600 rounded-xl bg-white dark:bg-stone-700 text-stone-900 dark:text-white focus:ring-2 focus:ring-primary-500 dark:focus:ring-amber-500">
                        <button type="submit" class="w-full py-3 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-xl hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                            Notify Me
                        </button>
                    </form>
                </div>
            `;
            document.body.appendChild(modal);

            document.getElementById('stock-alert-form')?.addEventListener('submit', async (e) => {
                e.preventDefault();
                const email = document.getElementById('stock-alert-email')?.value;
                if (!email) return;

                try {
                    // In production, call API
                    // await ProductsApi.subscribeStockAlert(productId, email);
                    stockAlertEmail = email;
                    Toast.success('You will be notified when this product is back in stock!');
                    modal.remove();
                } catch (error) {
                    Toast.error('Failed to subscribe. Please try again.');
                }
            });
        });
    }

    // ============================================
    // ENHANCED FEATURE: Social Sharing
    // ============================================
    function initSocialSharing() {
        document.querySelectorAll('.share-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const platform = btn.dataset.platform;
                const url = encodeURIComponent(window.location.href);
                const title = encodeURIComponent(document.title);
                const productName = document.querySelector('h1')?.textContent || '';

                const shareUrls = {
                    facebook: `https://www.facebook.com/sharer/sharer.php?u=${url}`,
                    twitter: `https://twitter.com/intent/tweet?url=${url}&text=${encodeURIComponent(productName)}`,
                    pinterest: `https://pinterest.com/pin/create/button/?url=${url}&description=${encodeURIComponent(productName)}`,
                    whatsapp: `https://api.whatsapp.com/send?text=${encodeURIComponent(productName + ' ' + window.location.href)}`,
                    copy: null
                };

                if (platform === 'copy') {
                    navigator.clipboard.writeText(window.location.href).then(() => {
                        Toast.success('Link copied to clipboard!');
                    }).catch(() => {
                        Toast.error('Failed to copy link');
                    });
                } else if (shareUrls[platform]) {
                    window.open(shareUrls[platform], '_blank', 'width=600,height=400');
                }
            });
        });
    }

    // ============================================
    // ENHANCED FEATURE: Q&A Section
    // ============================================
    function initQASection() {
        const qaContainer = document.getElementById('qa-section');
        if (!qaContainer) return;

        const productId = document.getElementById('product-detail')?.dataset.productId;
        if (!productId) return;

        // Load existing Q&As (mock data for now)
        const mockQAs = [
            { question: 'Is this product machine washable?', answer: 'Yes, we recommend washing on a gentle cycle with cold water.', askedBy: 'John D.', date: '2 days ago' },
            { question: 'What materials is this made from?', answer: 'This product is crafted from 100% organic cotton sourced from sustainable farms.', askedBy: 'Sarah M.', date: '1 week ago' },
        ];

        qaContainer.innerHTML = `
            <div class="space-y-6">
                <div class="flex items-center justify-between">
                    <h3 class="text-lg font-semibold text-stone-900 dark:text-white">Questions & Answers</h3>
                    <button id="ask-question-btn" class="text-sm font-medium text-primary-600 dark:text-amber-400 hover:underline">Ask a Question</button>
                </div>
                <div id="qa-list" class="space-y-4">
                    ${mockQAs.map(qa => `
                        <div class="bg-stone-50 dark:bg-stone-700/50 rounded-xl p-4">
                            <div class="flex items-start gap-3 mb-2">
                                <span class="text-primary-600 dark:text-amber-400 font-bold">Q:</span>
                                <div>
                                    <p class="text-stone-900 dark:text-white font-medium">${Templates.escapeHtml(qa.question)}</p>
                                    <p class="text-xs text-stone-500 dark:text-stone-400 mt-1">${qa.askedBy} • ${qa.date}</p>
                                </div>
                            </div>
                            ${qa.answer ? `
                                <div class="flex items-start gap-3 mt-3 pl-6">
                                    <span class="text-emerald-600 dark:text-emerald-400 font-bold">A:</span>
                                    <p class="text-stone-600 dark:text-stone-300">${Templates.escapeHtml(qa.answer)}</p>
                                </div>
                            ` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        document.getElementById('ask-question-btn')?.addEventListener('click', () => {
            const modal = document.createElement('div');
            modal.id = 'ask-question-modal';
            modal.className = 'fixed inset-0 z-50 flex items-center justify-center p-4';
            modal.innerHTML = `
                <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('ask-question-modal').remove()"></div>
                <div class="relative bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-md w-full p-6">
                    <button onclick="document.getElementById('ask-question-modal').remove()" class="absolute top-4 right-4 w-8 h-8 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                    <h3 class="text-xl font-bold text-stone-900 dark:text-white mb-4">Ask a Question</h3>
                    <form id="question-form" class="space-y-4">
                        <textarea id="question-input" rows="4" placeholder="Type your question here..." required class="w-full px-4 py-3 border border-stone-300 dark:border-stone-600 rounded-xl bg-white dark:bg-stone-700 text-stone-900 dark:text-white resize-none focus:ring-2 focus:ring-primary-500"></textarea>
                        <button type="submit" class="w-full py-3 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-xl hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                            Submit Question
                        </button>
                    </form>
                </div>
            `;
            document.body.appendChild(modal);

            document.getElementById('question-form')?.addEventListener('submit', (e) => {
                e.preventDefault();
                Toast.success('Your question has been submitted!');
                modal.remove();
            });
        });
    }

    // ============================================
    // ENHANCED FEATURE: Delivery Estimate
    // ============================================
    function initDeliveryEstimate() {
        const deliveryContainer = document.getElementById('delivery-estimate');
        if (!deliveryContainer) return;

        // Calculate estimated delivery (mock)
        const today = new Date();
        const minDays = 3;
        const maxDays = 7;
        const minDate = new Date(today.getTime() + minDays * 24 * 60 * 60 * 1000);
        const maxDate = new Date(today.getTime() + maxDays * 24 * 60 * 60 * 1000);

        const formatDate = (date) => date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });

        deliveryContainer.innerHTML = `
            <div class="flex items-start gap-3 p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-xl">
                <svg class="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4"/>
                </svg>
                <div>
                    <p class="text-sm font-medium text-emerald-700 dark:text-emerald-300">Estimated Delivery</p>
                    <p class="text-emerald-600 dark:text-emerald-400 font-semibold">${formatDate(minDate)} - ${formatDate(maxDate)}</p>
                    <p class="text-xs text-emerald-600 dark:text-emerald-400 mt-1">Free shipping on orders over $50</p>
                </div>
            </div>
        `;
    }

    // ============================================
    // ENHANCED FEATURE: Recently Viewed Tracking
    // ============================================
    function initRecentlyViewedTracking() {
        const productId = document.getElementById('product-detail')?.dataset.productId;
        const productSlug = document.getElementById('product-detail')?.dataset.productSlug;
        const productName = document.querySelector('h1')?.textContent;
        const productImage = document.getElementById('main-product-image')?.src || document.getElementById('main-image')?.src;
        const productPrice = document.getElementById('product-price')?.textContent;

        if (!productId) return;

        const recentlyViewed = JSON.parse(localStorage.getItem('recentlyViewed') || '[]');
        
        // Remove if already exists
        const existingIndex = recentlyViewed.findIndex(p => p.id === productId);
        if (existingIndex > -1) {
            recentlyViewed.splice(existingIndex, 1);
        }

        // Add to front
        recentlyViewed.unshift({
            id: productId,
            slug: productSlug,
            name: productName,
            image: productImage,
            price: productPrice,
            viewedAt: new Date().toISOString()
        });

        // Keep only last 20 items
        localStorage.setItem('recentlyViewed', JSON.stringify(recentlyViewed.slice(0, 20)));
    }

    // ============================================
    // ENHANCED FEATURE: Sticky Add to Cart (Mobile)
    // ============================================
    function initStickyAddToCart() {
        const existingSticky = document.getElementById('mobile-sticky-atc') || document.getElementById('mobile-sticky-atc-js');
        if (existingSticky || window.innerWidth >= 1024) return;

        const product = currentProduct;
        if (!product) return;

        const sticky = document.createElement('div');
        sticky.id = 'mobile-sticky-atc-enhanced';
        sticky.className = 'fixed bottom-0 inset-x-0 z-40 lg:hidden bg-white dark:bg-stone-800 border-t border-stone-200 dark:border-stone-700 shadow-2xl p-3 transform translate-y-full transition-transform duration-300';
        sticky.innerHTML = `
            <div class="flex items-center gap-3">
                <div class="flex-1 min-w-0">
                    <p class="text-xs text-stone-500 dark:text-stone-400 truncate">${product.name || ''}</p>
                    <p class="font-bold text-stone-900 dark:text-white">${product.sale_price ? Templates.formatPrice(product.sale_price) : Templates.formatPrice(product.price || 0)}</p>
                </div>
                <button id="sticky-add-to-cart" class="px-6 py-3 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-xl hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                    Add to Cart
                </button>
            </div>
        `;
        document.body.appendChild(sticky);

        // Show on scroll past add to cart button
        const mainAddBtn = document.getElementById('add-to-cart-btn');
        if (mainAddBtn) {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        sticky.classList.add('translate-y-full');
                    } else {
                        sticky.classList.remove('translate-y-full');
                    }
                });
            }, { threshold: 0 });

            observer.observe(mainAddBtn);
        }

        document.getElementById('sticky-add-to-cart')?.addEventListener('click', () => {
            document.getElementById('add-to-cart-btn')?.click();
        });
    }

    // ============================================
    // ENHANCED FEATURE: Product Video Player
    // ============================================
    function initProductVideoPlayer() {
        const videoThumbnails = document.querySelectorAll('[data-video-url]');
        
        videoThumbnails.forEach(thumb => {
            thumb.addEventListener('click', () => {
                const videoUrl = thumb.dataset.videoUrl;
                if (!videoUrl) return;

                const modal = document.createElement('div');
                modal.id = 'video-player-modal';
                modal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/90';
                modal.innerHTML = `
                    <button onclick="document.getElementById('video-player-modal').remove()" class="absolute top-4 right-4 w-12 h-12 bg-white/20 rounded-full flex items-center justify-center hover:bg-white/30 transition-colors">
                        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                    <video controls autoplay class="max-w-full max-h-[90vh] rounded-xl">
                        <source src="${videoUrl}" type="video/mp4">
                        Your browser does not support video playback.
                    </video>
                `;
                document.body.appendChild(modal);
            });
        });
    }

    function bindExistingEvents() {
        // Bind quantity controls
        initQuantityControls();
        // Bind add to cart
        initAddToCartFromExisting();
        // Bind wishlist
        initWishlistFromExisting();
        markWishlistButtonIfNeeded();
            async function markWishlistButtonIfNeeded() {
                const btn = document.getElementById('add-to-wishlist-btn');
                if (!btn) return;
                const productId = document.getElementById('product-detail')?.dataset.productId;
                if (!productId) return;
                if (typeof WishlistApi === 'undefined') return;
                try {
                    const resp = await WishlistApi.getWishlist({ pageSize: 100 });
                    if (resp.success && resp.data?.items) {
                        const found = resp.data.items.some(item => item.product_id === productId || item.product === productId);
                        if (found) {
                            btn.querySelector('svg')?.setAttribute('fill', 'currentColor');
                            btn.classList.add('text-red-500');
                        } else {
                            btn.querySelector('svg')?.setAttribute('fill', 'none');
                            btn.classList.remove('text-red-500');
                        }
                    }
                } catch (e) {}
            }
        // Bind variant selection
        initVariantSelectionFromExisting();
        // Bind image gallery
        initGalleryFromExisting();
        // Bind compare (server-rendered path)
        initCompareFromExisting();
        // Bind tabs
        initTabsFromExisting();
    }

    function initTabsFromExisting() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabName = btn.dataset.tab;
                
                // Update button styles
                tabButtons.forEach(b => {
                    b.classList.remove('border-primary-500', 'text-primary-600');
                    b.classList.add('border-transparent', 'text-gray-500');
                });
                btn.classList.add('border-primary-500', 'text-primary-600');
                btn.classList.remove('border-transparent', 'text-gray-500');
                
                // Show/hide content
                tabContents.forEach(content => {
                    if (content.id === `${tabName}-tab`) {
                        content.classList.remove('hidden');
                    } else {
                        content.classList.add('hidden');
                    }
                });
            });
        });
    }

    function initQuantityControls() {
        const decreaseBtn = document.getElementById('decrease-qty');
        const increaseBtn = document.getElementById('increase-qty');
        const qtyInput = document.getElementById('quantity');

        decreaseBtn?.addEventListener('click', () => {
            const current = parseInt(qtyInput?.value) || 1;
            if (current > 1) qtyInput.value = current - 1;
        });

        increaseBtn?.addEventListener('click', () => {
            const current = parseInt(qtyInput?.value) || 1;
            const max = parseInt(qtyInput?.max) || 99;
            if (current < max) qtyInput.value = current + 1;
        });
    }

    function initAddToCartFromExisting() {
        const btn = document.getElementById('add-to-cart-btn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            const productId = document.getElementById('product-detail')?.dataset.productId;
            const quantity = parseInt(document.getElementById('quantity')?.value) || 1;
            const variantInput = document.querySelector('input[name="variant"]');
            const hasVariants = !!variantInput;
            const variantId = document.querySelector('input[name="variant"]:checked')?.value;

            if (!productId) return;

            // If product has variants, enforce selection
            if (hasVariants && !variantId) {
                Toast.warning('Please select a variant before adding to cart.');
                return;
            }

            btn.disabled = true;
            const originalHtml = btn.innerHTML;
            btn.innerHTML = '<svg class="animate-spin h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';



            try {

                await CartApi.addItem(productId, quantity, variantId || null);

                Toast.success('Added to cart!');
                document.dispatchEvent(new CustomEvent('cart:updated'));
            } catch (error) {
                Toast.error(error.message || 'Failed to add to cart.');
            } finally {
                btn.disabled = false;
                btn.innerHTML = originalHtml;
            }
        });
    }

    function initWishlistFromExisting() {
        const btn = document.getElementById('add-to-wishlist-btn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            const productId = document.getElementById('product-detail')?.dataset.productId;
            if (!productId) return;

            if (typeof AuthApi !== 'undefined' && !AuthApi.isAuthenticated()) {
                Toast.warning('Please login to add items to your wishlist.');
                window.location.href = '/account/login/?next=' + encodeURIComponent(window.location.pathname);
                return;
            }

            try {
                // Check if already in wishlist
                let inWishlist = false;
                if (typeof WishlistApi !== 'undefined') {
                    const resp = await WishlistApi.getWishlist({ pageSize: 100 });
                    if (resp.success && resp.data?.items) {
                        inWishlist = resp.data.items.some(item => item.product_id === productId || item.product === productId);
                    }
                }
                if (inWishlist) {
                    // Remove from wishlist
                    // Find the wishlist item id
                    const resp = await WishlistApi.getWishlist({ pageSize: 100 });
                    const item = resp.data.items.find(item => item.product_id === productId || item.product === productId);
                    if (item) {
                        await WishlistApi.removeItem(item.id);
                        Toast.success('Removed from wishlist!');
                        btn.querySelector('svg')?.setAttribute('fill', 'none');
                        btn.classList.remove('text-red-500');
                        btn.setAttribute('aria-pressed', 'false');
                    }
                } else {
                    // Add to wishlist
                    await WishlistApi.addItem(productId);
                    Toast.success('Added to wishlist!');
                    btn.querySelector('svg')?.setAttribute('fill', 'currentColor');
                    btn.classList.add('text-red-500');
                    btn.setAttribute('aria-pressed', 'true');
                }
            } catch (error) {
                Toast.error(error.message || 'Wishlist action failed.');
            }
        });
    }

    function initVariantSelectionFromExisting() {
        const variantInputs = document.querySelectorAll('input[name="variant"]');
        variantInputs.forEach(input => {
            input.addEventListener('change', () => {
                selectedVariant = input.value;
                // Update price if variant has different price
                const price = input.dataset.price;
                const stock = parseInt(input.dataset.stock || '0');
                if (price) {
                    const priceEl = document.getElementById('product-price');
                    if (priceEl && window.Templates?.formatPrice) {
                        priceEl.textContent = window.Templates.formatPrice(parseFloat(price));
                    }
                }
                // Update stock display
                const stockEl = document.getElementById('stock-status');
                const addBtn = document.getElementById('add-to-cart-btn');
                const mobileStock = document.getElementById('mobile-stock');
                const mobileBtn = document.getElementById('mobile-add-to-cart');
                if (stockEl) {
                    if (stock > 10) stockEl.innerHTML = '<span class="text-green-600">In Stock</span>';
                    else if (stock > 0) stockEl.innerHTML = `<span class="text-orange-500">Only ${stock} left</span>`;
                    else stockEl.innerHTML = `<span class="text-red-600">Out of Stock</span>`;
                }
                if (addBtn) addBtn.disabled = stock <= 0;
                if (mobileBtn) mobileBtn.disabled = stock <= 0;
                if (mobileStock) mobileStock.textContent = stock > 0 ? (stock > 10 ? 'In stock' : `${stock} available`) : 'Out of stock';
            });
        });
    }

    function initGalleryFromExisting() {
        // Image change is handled via onclick in template
        // Just setup zoom functionality if needed
        const mainImage = document.getElementById('main-image');
        mainImage?.addEventListener('click', () => {
            // Could open lightbox here
        });
    }

    function getProductSlugFromUrl() {
        const path = window.location.pathname;
        const match = path.match(/\/products\/([^\/]+)/);
        return match ? match[1] : null;
    }

    async function loadProduct(slug) {
        const container = document.getElementById('product-detail');
        if (!container) return;

        Loader.show(container, 'skeleton');

        try {
            const response = await ProductsApi.getProduct(slug);
            currentProduct = response.data;

            if (!currentProduct) {
                window.location.href = '/404/';
                return;
            }

            document.title = `${currentProduct.name} | Bunoraa`;
            
            renderProduct(currentProduct);

            // Update SEO/meta for SPA flows
            updateMetaTags(currentProduct);
            updateStructuredData(currentProduct);

            // Load related data in parallel but also lazily for non-critical sections
            await Promise.all([
                loadBreadcrumbs(currentProduct),
                loadRelatedProducts(currentProduct),
                loadReviews(currentProduct),
                loadRecommendations(currentProduct)
            ]);

            // Setup lazy-loading for below-the-fold sections
            initLazySections();

            trackProductView(currentProduct);
        } catch (error) {
            console.error('Failed to load product:', error);
            container.innerHTML = '<p class="text-red-500 text-center py-8">Failed to load product. Please try again.</p>';
        }
    }

    // Refresh product when currency changes (support SPA flows)
    document.addEventListener('currency:changed', async (e) => {
        try {
            if (!serverRendered && currentProduct && currentProduct.slug) {
                await loadProduct(currentProduct.slug);
            } else {
                // For server-rendered pages, perform a full reload to pick up server-side formatting
                location.reload();
            }
        } catch (err) {
        }
    });

    function renderProduct(product) {
        const container = document.getElementById('product-detail');
        if (!container) return;

        const images = product.images || [];
        const mainImage = product.image || images[0]?.image || '';
        const hasVariants = product.variants && product.variants.length > 0;
        const inStock = product.stock_quantity > 0 || product.in_stock;
        const onSale = product.sale_price && product.sale_price < product.price;

        container.innerHTML = `
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
                <!-- Gallery -->
                <div id="product-gallery" class="product-gallery">
                    <div class="main-image-container relative rounded-xl overflow-hidden bg-gray-100" style="aspect-ratio: ${product?.aspect?.css || '1/1'};">
                        <img 
                            src="${mainImage}" 
                            alt="${Templates.escapeHtml(product.name)}"
                            loading="lazy"
                            decoding="async"
                            class="main-image w-full h-full object-cover cursor-zoom-in"
                            id="main-product-image"
                        >
                        ${onSale ? `
                            <span class="absolute top-4 left-4 px-3 py-1 bg-red-500 text-white text-sm font-medium rounded-full">
                                Sale
                            </span>
                        ` : ''}
                        ${!inStock ? `
                            <span class="absolute top-4 right-4 px-3 py-1 bg-gray-900 text-white text-sm font-medium rounded-full">
                                Out of Stock
                            </span>
                        ` : ''}
                    </div>
                    ${images.length > 1 ? `
                        <div class="thumbnails flex gap-2 mt-4 overflow-x-auto pb-2">
                            ${images.map((img, index) => `
                                <button 
                                    class="thumbnail flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden border-2 ${index === 0 ? 'border-primary-500' : 'border-transparent'} hover:border-primary-500 transition-colors"
                                    data-image="${img.image}"
                                    data-index="${index}"
                                >
                                    <img src="${img.image}" alt="" loading="lazy" decoding="async" class="w-full h-full object-cover">
                                </button>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>

                <!-- Product Info -->
                <div class="product-info">
                    <!-- Brand -->
                    ${product.brand ? `
                        <a href="/products/?brand=${product.brand.id}" class="text-sm text-primary-600 hover:text-primary-700 font-medium">
                            ${Templates.escapeHtml(product.brand.name)}
                        </a>
                    ` : ''}

                    <!-- Title -->
                    <h1 class="text-2xl md:text-3xl font-bold text-gray-900 mt-2">
                        ${Templates.escapeHtml(product.name)}
                    </h1>

                    <!-- Rating -->
                    ${product.average_rating ? `
                        <div class="flex items-center gap-2 mt-3">
                            <div class="flex items-center">
                                ${Templates.renderStars(product.average_rating)}
                            </div>
                            <span class="text-sm text-gray-600">
                                ${product.average_rating.toFixed(1)} (${product.review_count || 0} reviews)
                            </span>
                            <a href="#reviews" class="text-sm text-primary-600 hover:text-primary-700">
                                Read reviews
                            </a>
                        </div>
                    ` : ''}

                    <!-- Price -->
                    <div class="mt-4">
                        ${Price.render({
                            price: product.current_price ?? product.price_converted ?? product.price,
                            salePrice: product.sale_price_converted ?? product.sale_price,
                            size: 'large'
                        })}
                    </div>

                    <!-- Short Description -->
                    ${product.short_description ? `
                        <p class="mt-4 text-gray-600">${Templates.escapeHtml(product.short_description)}</p>
                    ` : ''}

                    <!-- Variants -->
                    ${hasVariants ? renderVariants(product.variants) : ''}

                    <!-- Quantity & Add to Cart -->
                    <div class="mt-6 space-y-4">
                        <div class="flex items-center gap-4">
                            <label class="text-sm font-medium text-gray-700">Quantity:</label>
                            <div class="flex items-center border border-gray-300 rounded-lg">
                                <button 
                                    class="qty-decrease px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
                                    aria-label="Decrease quantity"
                                >
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"/>
                                    </svg>
                                </button>
                                <input 
                                    type="number" 
                                    id="product-quantity"
                                    value="1" 
                                    min="1" 
                                    max="${product.stock_quantity || 99}"
                                    class="w-16 text-center border-0 focus:ring-0"
                                >
                                <button 
                                    class="qty-increase px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
                                    aria-label="Increase quantity"
                                >
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
                                    </svg>
                                </button>
                            </div>
                            ${product.stock_quantity ? `
                                <div id="stock-status" class="text-sm text-gray-500">${product.stock_quantity > 10 ? 'In stock' : product.stock_quantity + ' available'}</div>
                            ` : `<div id="stock-status" class="text-red-600">Out of Stock</div>`}
                        </div>

                        <div class="flex gap-3">
                            <button 
                                id="add-to-cart-btn"
                                class="flex-1 px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                                ${!inStock ? 'disabled' : ''}
                            >
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"/>
                                </svg>
                                ${inStock ? 'Add to Cart' : 'Out of Stock'}
                            </button>
                            <button 
                                id="add-to-wishlist-btn"
                                class="px-4 py-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                                aria-label="Add to wishlist"
                                data-product-id="${product.id}"
                            >
                                <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
                                </svg>
                            </button>
                        </div>
                    </div>

                    <!-- Product Meta -->
                    <div class="mt-6 pt-6 border-t border-gray-200 space-y-3 text-sm">
                        ${product.sku ? `
                            <div class="flex">
                                <span class="text-gray-500 w-24">SKU:</span>
                                <span class="text-gray-900">${Templates.escapeHtml(product.sku)}</span>
                            </div>
                        ` : ''}
                        ${product.category ? `
                            <div class="flex">
                                <span class="text-gray-500 w-24">Category:</span>
                                <a href="/categories/${product.category.slug}/" class="text-primary-600 hover:text-primary-700">
                                    ${Templates.escapeHtml(product.category.name)}
                                </a>
                            </div>
                        ` : ''}
                        ${product.tags && product.tags.length ? `
                            <div class="flex">
                                <span class="text-gray-500 w-24">Tags:</span>
                                <div class="flex flex-wrap gap-1">
                                    ${product.tags.map(tag => `
                                        <a href="/products/?tag=${tag.slug}" class="px-2 py-0.5 bg-gray-100 text-gray-600 rounded hover:bg-gray-200 transition-colors">
                                            ${Templates.escapeHtml(tag.name)}
                                        </a>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                    </div>

                    <!-- Share -->
                    <div class="mt-6 pt-6 border-t border-gray-200">
                        <span class="text-sm text-gray-500">Share:</span>
                        <div class="flex gap-2 mt-2">
                            <button class="share-btn p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors" data-platform="facebook" aria-label="Share on Facebook">
                                <svg class="w-5 h-5 text-[#1877F2]" fill="currentColor" viewBox="0 0 24 24"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
                            </button>
                            <button class="share-btn p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors" data-platform="twitter" aria-label="Share on Twitter">
                                <svg class="w-5 h-5 text-[#1DA1F2]" fill="currentColor" viewBox="0 0 24 24"><path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/></svg>
                            </button>
                            <button class="share-btn p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors" data-platform="pinterest" aria-label="Share on Pinterest">
                                <svg class="w-5 h-5 text-[#E60023]" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0C5.373 0 0 5.372 0 12c0 5.084 3.163 9.426 7.627 11.174-.105-.949-.2-2.405.042-3.441.218-.937 1.407-5.965 1.407-5.965s-.359-.719-.359-1.782c0-1.668.967-2.914 2.171-2.914 1.023 0 1.518.769 1.518 1.69 0 1.029-.655 2.568-.994 3.995-.283 1.194.599 2.169 1.777 2.169 2.133 0 3.772-2.249 3.772-5.495 0-2.873-2.064-4.882-5.012-4.882-3.414 0-5.418 2.561-5.418 5.207 0 1.031.397 2.138.893 2.738a.36.36 0 01.083.345l-.333 1.36c-.053.22-.174.267-.402.161-1.499-.698-2.436-2.889-2.436-4.649 0-3.785 2.75-7.262 7.929-7.262 4.163 0 7.398 2.967 7.398 6.931 0 4.136-2.607 7.464-6.227 7.464-1.216 0-2.359-.631-2.75-1.378l-.748 2.853c-.271 1.043-1.002 2.35-1.492 3.146C9.57 23.812 10.763 24 12 24c6.627 0 12-5.373 12-12 0-6.628-5.373-12-12-12z"/></svg>
                            </button>
                            <button class="share-btn p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors" data-platform="copy" aria-label="Copy link">
                                <svg class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Product Tabs -->
            <div class="mt-12" data-tabs data-variant="underline" id="product-tabs">
                <div class="border-b border-gray-200">
                    <nav class="flex -mb-px">
                        <button data-tab class="px-6 py-4 text-sm font-medium text-gray-600 hover:text-gray-900 border-b-2 border-transparent">
                            Description
                        </button>
                        ${product.specifications && Object.keys(product.specifications).length ? `
                            <button data-tab class="px-6 py-4 text-sm font-medium text-gray-600 hover:text-gray-900 border-b-2 border-transparent">
                                Specifications
                            </button>
                        ` : ''}
                        <button data-tab class="px-6 py-4 text-sm font-medium text-gray-600 hover:text-gray-900 border-b-2 border-transparent">
                            Reviews (${product.review_count || 0})
                        </button>
                    </nav>
                </div>
                <div class="py-6">
                    <div data-tab-panel>
                        <div class="prose max-w-none">
                            ${product.description || '<p class="text-gray-500">No description available.</p>'}
                        </div>
                    </div>
                    ${product.specifications && Object.keys(product.specifications).length ? `
                        <div data-tab-panel>
                            <table class="w-full">
                                <tbody>
                                    ${Object.entries(product.specifications).map(([key, value]) => `
                                        <tr class="border-b border-gray-100">
                                            <td class="py-3 text-sm font-medium text-gray-500 w-1/3">${Templates.escapeHtml(key)}</td>
                                            <td class="py-3 text-sm text-gray-900">${Templates.escapeHtml(String(value))}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    ` : ''}
                    <div data-tab-panel id="reviews">
                        <div id="reviews-container">
                            <!-- Reviews loaded here -->
                        </div>
                    </div>
                </div>
            </div>
        `;

        initGallery();
        initQuantityControls();
        initAddToCart();
        initWishlist();
        initCompare();
        initShare();
        Tabs.init();

        // Ensure mobile sticky ATC exists and is in sync
        createOrUpdateMobileAtc(product);
    }

    function createOrUpdateMobileAtc(product) {
        let el = document.getElementById('mobile-sticky-atc-js');
        if (!el) {
            el = document.createElement('div');
            el.id = 'mobile-sticky-atc-js';
            el.className = 'fixed inset-x-4 bottom-4 z-40 lg:hidden';
            el.innerHTML = `
                <div class="bg-white shadow-lg rounded-xl p-3 flex items-center gap-3">
                    <div class="flex-1">
                        <div class="text-sm text-gray-500">${product.sale_price ? 'Now' : ''}</div>
                        <div class="font-semibold text-lg ${product.sale_price ? 'text-red-600' : ''}">${product.sale_price ? Templates.formatPrice(product.sale_price) + ' <span class="text-sm line-through text-gray-400">' + Templates.formatPrice(product.price) + '</span>' : Templates.formatPrice(product.price)}</div>
                        <div id="mobile-stock-js" class="text-xs text-gray-500">${product.stock_quantity > 0 ? (product.stock_quantity > 10 ? 'In stock' : product.stock_quantity + ' available') : 'Out of stock'}</div>
                    </div>
                    ${product.stock_quantity > 0 ? `<button id="mobile-add-to-cart-js" class="bg-primary-600 text-white px-4 py-2 rounded-lg font-semibold">Add</button>` : `<button class="bg-gray-300 text-gray-500 px-4 py-2 rounded-lg font-semibold cursor-not-allowed" disabled>Out</button>`}
                </div>
            `;
            document.body.appendChild(el);

            // Wire mobile button
            const mobBtn = document.getElementById('mobile-add-to-cart-js');
            if (mobBtn) mobBtn.addEventListener('click', () => document.getElementById('add-to-cart-btn')?.click());
        } else {
            // Update existing
            const priceEl = el.querySelector('.font-semibold');
            if (priceEl) priceEl.innerHTML = product.sale_price ? Templates.formatPrice(product.sale_price) + ' <span class="text-sm line-through text-gray-400">' + Templates.formatPrice(product.price) + '</span>' : Templates.formatPrice(product.price);
            const stockEl = document.getElementById('mobile-stock-js');
            if (stockEl) stockEl.textContent = product.stock_quantity > 0 ? (product.stock_quantity > 10 ? 'In stock' : product.stock_quantity + ' available') : 'Out of stock';
            const mobBtn = document.getElementById('mobile-add-to-cart-js');
            if (mobBtn) mobBtn.disabled = product.stock_quantity <= 0;
        }
    }

    function renderVariants(variants) {
        const grouped = {};
        variants.forEach(variant => {
            if (!grouped[variant.attribute_name]) {
                grouped[variant.attribute_name] = [];
            }
            grouped[variant.attribute_name].push(variant);
        });

        return Object.entries(grouped).map(([attrName, options]) => `
            <div class="mt-6">
                <label class="text-sm font-medium text-gray-700">${Templates.escapeHtml(attrName)}:</label>
                <div class="flex flex-wrap gap-2 mt-2" role="radiogroup" aria-label="${Templates.escapeHtml(attrName)}">
                    ${options.map((opt, index) => `
                        <button 
                            class="variant-btn px-4 py-2 border rounded-lg text-sm transition-colors ${index === 0 ? 'border-primary-500 bg-primary-50 text-primary-700' : 'border-gray-300 hover:border-gray-400'}"
                            role="radio"
                            aria-checked="${index === 0 ? 'true' : 'false'}"
                            data-variant-id="${opt.id}"
                            data-price="${opt.price_converted ?? opt.price ?? ''}"
                            data-stock="${opt.stock_quantity || 0}"
                            tabindex="0"
                        >
                            ${Templates.escapeHtml(opt.value)}
                            ${((opt.price_converted ?? opt.price) && (opt.price !== currentProduct.price)) ? `
                                <span class="text-xs text-gray-500">(${Templates.formatPrice(opt.price_converted ?? opt.price)})</span>
                            ` : ''}
                        </button>
                    `).join('')}
                </div>
            </div>
        `).join('');
    }

    function initGallery() {
        const thumbnails = document.querySelectorAll('.thumbnail');
        const mainImage = document.getElementById('main-product-image');

        // Keyboard navigation for thumbnails
        let focusedIndex = 0;
        thumbnails.forEach((thumb, idx) => {
            thumb.setAttribute('tabindex', '0');
            thumb.addEventListener('click', () => {
                thumbnails.forEach(t => t.classList.remove('border-primary-500'));
                thumb.classList.add('border-primary-500');
                mainImage.src = thumb.dataset.image || thumb.dataset.src;
                focusedIndex = idx;
            });
            thumb.addEventListener('keydown', (ev) => {
                if (ev.key === 'Enter' || ev.key === ' ') {
                    ev.preventDefault();
                    thumb.click();
                } else if (ev.key === 'ArrowRight') {
                    ev.preventDefault();
                    const next = thumbnails[(idx + 1) % thumbnails.length];
                    next.focus();
                    next.click();
                } else if (ev.key === 'ArrowLeft') {
                    ev.preventDefault();
                    const prev = thumbnails[(idx - 1 + thumbnails.length) % thumbnails.length];
                    prev.focus();
                    prev.click();
                }
            });
        });

        // Lightbox via Modal (supports images, video, model-viewer)
        mainImage?.addEventListener('click', () => {
            const images = currentProduct.images?.map(img => img.image) || [currentProduct.image];
            const currentIndex = parseInt(document.querySelector('.thumbnail.border-primary-500')?.dataset.index) || 0;

            // Build content for lightbox - support videos and 3D models if present
            const slides = (currentProduct.images || []).map(img => {
                return {
                    type: img.type || (img.video_url ? 'video' : 'image'),
                    src: img.video_url || img.model_url || img.image
                };
            });

            // Fallback simple image carousel in Modal
            const content = slides.map(s => {
                if (s.type === 'video') {
                    return `<div class="w-full h-full max-h-[70vh]"><video controls class="w-full h-full object-contain"><source src="${s.src}" type="video/mp4">Your browser does not support video.</video></div>`;
                }
                if (s.type === 'model') {
                    // Lazy load model-viewer
                    if (!window.customElements || !window['model-viewer']) {
                        const script = document.createElement('script');
                        script.type = 'module';
                        script.src = 'https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js';
                        document.head.appendChild(script);
                    }
                    return `<div class="w-full h-full max-h-[70vh]"><model-viewer src="${s.src}" camera-controls ar ar-modes="webxr scene-viewer quick-look" class="w-full h-full"></model-viewer></div>`;
                }
                return `<div class="w-full h-full max-h-[70vh] flex items-center justify-center"><img src="${s.src}" class="max-w-full max-h-[70vh] object-contain" alt="${Templates.escapeHtml(currentProduct.name)}"></div>`;
            }).join('');

            Modal.open({
                title: Templates.escapeHtml(currentProduct.name),
                content: `<div class="space-y-2">${content}</div>`,
                size: 'xl'
            });
        });
    }

    function initQuantityControls() {
        const qtyInput = document.getElementById('product-quantity');
        const decreaseBtn = document.querySelector('.qty-decrease');
        const increaseBtn = document.querySelector('.qty-increase');

        decreaseBtn?.addEventListener('click', () => {
            const current = parseInt(qtyInput.value) || 1;
            if (current > 1) {
                qtyInput.value = current - 1;
            }
        });

        increaseBtn?.addEventListener('click', () => {
            const current = parseInt(qtyInput.value) || 1;
            const max = parseInt(qtyInput.max) || 99;
            if (current < max) {
                qtyInput.value = current + 1;
            }
        });

        const variantBtns = document.querySelectorAll('.variant-btn');
        variantBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                variantBtns.forEach(b => {
                    b.classList.remove('border-primary-500', 'bg-primary-50', 'text-primary-700');
                    b.classList.add('border-gray-300');
                    b.setAttribute('aria-checked', 'false');
                });
                btn.classList.add('border-primary-500', 'bg-primary-50', 'text-primary-700');
                btn.classList.remove('border-gray-300');
                btn.setAttribute('aria-checked', 'true');

                selectedVariant = btn.dataset.variantId;

                // Update price if variant has custom price
                if (btn.dataset.price) {
                    const priceContainer = document.querySelector('.product-info .mt-4');
                    if (priceContainer) {
                        priceContainer.innerHTML = Price.render({
                            price: parseFloat(btn.dataset.price),
                            size: 'large'
                        });
                    }
                }

                // Update stock info
                const stock = parseInt(btn.dataset.stock || '0');
                const stockEl = document.getElementById('stock-status');
                const addToCartBtn = document.getElementById('add-to-cart-btn');
                const mobileStock = document.getElementById('mobile-stock');
                const mobileATC = document.getElementById('mobile-add-to-cart');

                if (stockEl) {
                    if (stock > 10) {
                        stockEl.innerHTML = `<span class="text-green-600 flex items-center"><svg class="w-5 h-5 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>In Stock</span>`;
                    } else if (stock > 0) {
                        stockEl.innerHTML = `<span class="text-orange-500 flex items-center"><svg class="w-5 h-5 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path></svg>Only ${stock} left</span>`;
                    } else {
                        stockEl.innerHTML = `<span class="text-red-600 flex items-center"><svg class="w-5 h-5 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path></svg>Out of Stock</span>`;
                    }
                }

                if (qtyInput) {
                    qtyInput.max = Math.max(stock, 1);
                    if (parseInt(qtyInput.value) > parseInt(qtyInput.max)) qtyInput.value = qtyInput.max;
                }

                if (mobileStock) {
                    mobileStock.textContent = stock > 0 ? (stock > 10 ? 'In stock' : `${stock} available`) : 'Out of stock';
                }

                if (addToCartBtn) {
                    addToCartBtn.disabled = stock <= 0;
                }
                if (mobileATC) {
                    mobileATC.disabled = stock <= 0;
                }
            });
        });

        if (variantBtns.length > 0) {
            const first = variantBtns[0];
            first.setAttribute('aria-checked', 'true');
            selectedVariant = first.dataset.variantId;
        }
    }

    function initAddToCart() {
        const btn = document.getElementById('add-to-cart-btn');
        const mobileBtn = document.getElementById('mobile-add-to-cart');
        if (!btn && !mobileBtn) return;

        const performAddToCart = async (triggerBtn) => {
            const quantity = parseInt(document.getElementById('product-quantity')?.value) || 1;
            const stockEl = document.getElementById('stock-status');
            // Validate variant selection if product has variants
            const hasVariants = !!document.querySelector('.variant-btn');
            if (hasVariants && !selectedVariant) {
                Toast.warning('Please select a variant before adding to cart.');
                return;
            }

            triggerBtn.disabled = true;
            const originalHtml = triggerBtn.innerHTML;
            triggerBtn.innerHTML = '<svg class="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';

            try {
                // If variant selected, verify stock (optimistic)
                const variantBtn = document.querySelector(`.variant-btn[data-variant-id="${selectedVariant}"]`);
                const stock = variantBtn ? parseInt(variantBtn.dataset.stock || '0') : (currentProduct.stock_quantity || 0);
                if (stock <= 0) {
                    Toast.error('This variant is out of stock.');
                    return;
                }

                await CartApi.addItem(currentProduct.id, quantity, selectedVariant || null);
                Toast.success('Added to cart!');
                document.dispatchEvent(new CustomEvent('cart:updated'));
            } catch (error) {
                Toast.error(error.message || 'Failed to add to cart.');
            } finally {
                triggerBtn.disabled = false;
                triggerBtn.innerHTML = originalHtml;
            }
        };

        btn?.addEventListener('click', () => performAddToCart(btn));
        mobileBtn?.addEventListener('click', () => performAddToCart(mobileBtn));
    }

    function initWishlist() {
        const btn = document.getElementById('add-to-wishlist-btn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            if (!AuthApi.isAuthenticated()) {
                Toast.warning('Please login to add items to your wishlist.');
                window.location.href = '/account/login/?next=' + encodeURIComponent(window.location.pathname);
                return;
            }

            try {
                await WishlistApi.addItem(currentProduct.id);
                Toast.success('Added to wishlist!');
                
                btn.querySelector('svg').setAttribute('fill', 'currentColor');
                btn.classList.add('text-red-500');
                btn.setAttribute('aria-pressed', 'true');
            } catch (error) {
                if (error.message?.includes('already')) {
                    Toast.info('This item is already in your wishlist.');
                } else {
                    Toast.error(error.message || 'Failed to add to wishlist.');
                }
            }
        });
    }

    function initShare() {
        const shareBtns = document.querySelectorAll('.share-btn');
        const url = encodeURIComponent(window.location.href);
        const title = encodeURIComponent(currentProduct?.name || document.title);

        shareBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const platform = btn.dataset.platform;
                let shareUrl = '';

                switch (platform) {
                    case 'facebook':
                        shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${url}`;
                        break;
                    case 'twitter':
                        shareUrl = `https://twitter.com/intent/tweet?url=${url}&text=${title}`;
                        break;
                    case 'pinterest':
                        const image = encodeURIComponent(currentProduct?.image || '');
                        shareUrl = `https://pinterest.com/pin/create/button/?url=${url}&media=${image}&description=${title}`;
                        break;
                    case 'copy':
                        navigator.clipboard.writeText(window.location.href)
                            .then(() => Toast.success('Link copied to clipboard!'))
                            .catch(() => Toast.error('Failed to copy link.'));
                        return;
                }

                if (shareUrl) {
                    window.open(shareUrl, '_blank', 'width=600,height=400');
                }
            });
        });
    }

    function initCompare() {
        const btn = document.getElementById('add-to-compare-btn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            if (typeof AuthApi !== 'undefined' && !AuthApi.isAuthenticated()) {
                Toast.warning('Please login to compare products.');
                window.location.href = '/account/login/?next=' + encodeURIComponent(window.location.pathname);
                return;
            }

            const productId = currentProduct?.id || document.getElementById('product-detail')?.dataset.productId;
            if (!productId) return;

            try {
                const resp = await ApiClient.post('/compare/', { product_id: productId }, { requiresAuth: true });
                if (resp.success) {
                    Toast.success(resp.message || 'Added to compare');
                    btn.setAttribute('aria-pressed', 'true');
                    btn.classList.add('text-primary-600');
                    btn.querySelector('svg')?.setAttribute('fill', 'currentColor');
                } else {
                    Toast.error(resp.message || 'Failed to add to compare');
                }
            } catch (err) {
                // Fallback: store locally if compare endpoint not available
                try {
                    const key = 'b_compare';
                    const list = JSON.parse(localStorage.getItem(key) || '[]');
                    if (!list.includes(productId)) {
                        list.push(productId);
                        localStorage.setItem(key, JSON.stringify(list));
                        Toast.success('Added to compare (local)');
                        btn.setAttribute('aria-pressed', 'true');
                        btn.classList.add('text-primary-600');
                        return;
                    }
                    Toast.info('Already in compare list');
                } catch (e) {
                    Toast.error(err.message || 'Failed to add to compare');
                }
            }
        });
    }

    // Bind compare button when page content is server-rendered (existing DOM)
    function initCompareFromExisting() {
        const btn = document.getElementById('add-to-compare-btn');
        if (!btn) return;
        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            // Ensure we have a product id in scope for both try and catch paths
            const productId = document.getElementById('product-detail')?.dataset.productId;
            if (!productId) return;

            try {
                if (typeof AuthApi !== 'undefined' && !AuthApi.isAuthenticated()) {
                    Toast.warning('Please login to compare products.');
                    window.location.href = '/account/login/?next=' + encodeURIComponent(window.location.pathname);
                    return;
                }

                // Try API call directly
                const resp = await ApiClient.post('/compare/', { product_id: productId }, { requiresAuth: true });
                if (resp.success) {
                    Toast.success(resp.message || 'Added to compare');
                    btn.setAttribute('aria-pressed', 'true');
                    btn.classList.add('text-primary-600');
                    btn.querySelector('svg')?.setAttribute('fill', 'currentColor');
                } else {
                    Toast.error(resp.message || 'Failed to add to compare');
                }
            } catch (err) {
                // Local fallback
                try {
                    const key = 'b_compare';
                    const list = JSON.parse(localStorage.getItem(key) || '[]');
                    if (!list.includes(productId)) {
                        list.push(productId);
                        localStorage.setItem(key, JSON.stringify(list));
                        Toast.success('Added to compare (local)');
                        btn.setAttribute('aria-pressed', 'true');
                        btn.classList.add('text-primary-600');
                        return;
                    }
                    Toast.info('Already in compare list');
                } catch (e) {
                    Toast.error(err.message || 'Failed to add to compare');
                }
            }
        });
    }

    async function loadBreadcrumbs(product) {
        const container = document.getElementById('breadcrumbs');
        if (!container) return;

        const items = [
            { label: 'Home', url: '/' },
            { label: 'Products', url: '/products/' }
        ];

        if (product.category) {
            try {
                const response = await CategoriesAPI.getBreadcrumbs(product.category.id);
                const categoryBreadcrumbs = response.data || [];
                categoryBreadcrumbs.forEach(cat => {
                    items.push({ label: cat.name, url: `/categories/${cat.slug}/` });
                });
            } catch (error) {
                items.push({ label: product.category.name, url: `/categories/${product.category.slug}/` });
            }
        }

        items.push({ label: product.name });

        container.innerHTML = Breadcrumb.render(items);
    }

    async function loadRelatedProducts(product) {
        const container = document.getElementById('related-products');
        if (!container) return;

        try {
            const response = await ProductsAPI.getRelated(product.id, { limit: 4 });
            const products = response.data || [];

            if (products.length === 0) {
                container.innerHTML = '';
                return;
            }

            container.innerHTML = `
                <h2 class="text-2xl font-bold text-gray-900 mb-6">You may also like</h2>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
                    ${products.map(p => ProductCard.render(p)).join('')}
                </div>
            `;

            ProductCard.bindEvents(container);
        } catch (error) {
            console.error('Failed to load related products:', error);
            container.innerHTML = '';
        }
    }

    async function loadReviews(product) {
        const container = document.getElementById('reviews-container');
        if (!container) return;

        Loader.show(container, 'spinner');

        try {
            const response = await ProductsAPI.getReviews(product.id);
            const reviews = response.data || [];

            container.innerHTML = `
                <!-- Review Summary -->
                <div class="flex flex-col md:flex-row gap-8 mb-8 pb-8 border-b border-gray-200">
                    <div class="text-center">
                        <div class="text-5xl font-bold text-gray-900">${product.average_rating?.toFixed(1) || '0.0'}</div>
                        <div class="flex justify-center mt-2">
                            ${Templates.renderStars(product.average_rating || 0)}
                        </div>
                        <div class="text-sm text-gray-500 mt-1">${product.review_count || 0} reviews</div>
                    </div>
                    ${AuthAPI.isAuthenticated() ? `
                        <div class="flex-1">
                            <button id="write-review-btn" class="px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 transition-colors">
                                Write a Review
                            </button>
                        </div>
                    ` : `
                        <div class="flex-1">
                            <p class="text-gray-600">
                                <a href="/account/login/?next=${encodeURIComponent(window.location.pathname)}" class="text-primary-600 hover:text-primary-700">Sign in</a> 
                                to write a review.
                            </p>
                        </div>
                    `}
                </div>

                <!-- Reviews List -->
                ${reviews.length > 0 ? `
                    <div class="space-y-6">
                        ${reviews.map(review => `
                            <div class="border-b border-gray-100 pb-6">
                                <div class="flex items-start gap-4">
                                    <div class="flex-shrink-0 w-10 h-10 bg-gray-200 rounded-full flex items-center justify-center">
                                        <span class="text-gray-600 font-medium">${(review.user?.first_name?.[0] || review.user?.email?.[0] || 'U').toUpperCase()}</span>
                                    </div>
                                    <div class="flex-1">
                                        <div class="flex items-center gap-2">
                                            <span class="font-medium text-gray-900">${Templates.escapeHtml(review.user?.first_name || 'Anonymous')}</span>
                                            ${review.verified_purchase ? `
                                                <span class="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded">Verified Purchase</span>
                                            ` : ''}
                                        </div>
                                        <div class="flex items-center gap-2 mt-1">
                                            ${Templates.renderStars(review.rating)}
                                            <span class="text-sm text-gray-500">${Templates.formatDate(review.created_at)}</span>
                                        </div>
                                        ${review.title ? `<h4 class="font-medium text-gray-900 mt-2">${Templates.escapeHtml(review.title)}</h4>` : ''}
                                        <p class="text-gray-600 mt-2">${Templates.escapeHtml(review.comment)}</p>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : `
                    <p class="text-gray-500 text-center py-8">No reviews yet. Be the first to review this product!</p>
                `}
            `;

            document.getElementById('write-review-btn')?.addEventListener('click', () => {
                showReviewForm(product);
            });
        } catch (error) {
            console.error('Failed to load reviews:', error);
            container.innerHTML = '<p class="text-red-500">Failed to load reviews.</p>';
        }
    }

    function showReviewForm(product) {
        Modal.open({
            title: 'Write a Review',
            content: `
                <form id="review-form" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Rating</label>
                        <div class="flex gap-1" id="rating-stars">
                            ${[1, 2, 3, 4, 5].map(i => `
                                <button type="button" class="rating-star text-gray-300 hover:text-yellow-400" data-rating="${i}">
                                    <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
                                    </svg>
                                </button>
                            `).join('')}
                        </div>
                        <input type="hidden" id="review-rating" value="0">
                    </div>
                    <div>
                        <label for="review-title" class="block text-sm font-medium text-gray-700 mb-1">Title (optional)</label>
                        <input type="text" id="review-title" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                    </div>
                    <div>
                        <label for="review-comment" class="block text-sm font-medium text-gray-700 mb-1">Your Review</label>
                        <textarea id="review-comment" rows="4" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500" required></textarea>
                    </div>
                </form>
            `,
            confirmText: 'Submit Review',
            onConfirm: async () => {
                const rating = parseInt(document.getElementById('review-rating').value);
                const title = document.getElementById('review-title').value.trim();
                const comment = document.getElementById('review-comment').value.trim();

                if (!rating || rating < 1) {
                    Toast.error('Please select a rating.');
                    return false;
                }

                if (!comment) {
                    Toast.error('Please write a review.');
                    return false;
                }

                try {
                    await ProductsAPI.createReview(product.id, { rating, title, comment });
                    Toast.success('Thank you for your review!');
                    loadReviews(product);
                    return true;
                } catch (error) {
                    Toast.error(error.message || 'Failed to submit review.');
                    return false;
                }
            }
        });

        const stars = document.querySelectorAll('.rating-star');
        const ratingInput = document.getElementById('review-rating');

        stars.forEach(star => {
            star.addEventListener('click', () => {
                const rating = parseInt(star.dataset.rating);
                ratingInput.value = rating;
                
                stars.forEach((s, i) => {
                    if (i < rating) {
                        s.classList.remove('text-gray-300');
                        s.classList.add('text-yellow-400');
                    } else {
                        s.classList.add('text-gray-300');
                        s.classList.remove('text-yellow-400');
                    }
                });
            });
        });
    }

    // Update meta tags dynamically for SPA
    function updateMetaTags(product) {
        try {
            document.title = `${product.name} | Bunoraa`;
            const desc = product.meta_description || product.short_description || '';
            document.querySelector('meta[name="description"]')?.setAttribute('content', desc);
            document.querySelector('meta[property="og:title"]')?.setAttribute('content', product.meta_title || product.name);
            document.querySelector('meta[property="og:description"]')?.setAttribute('content', desc);
            const ogImage = product.images?.[0]?.image || product.image;
            if (ogImage) document.querySelector('meta[property="og:image"]')?.setAttribute('content', ogImage);
            document.querySelector('meta[name="twitter:title"]')?.setAttribute('content', product.meta_title || product.name);
            document.querySelector('meta[name="twitter:description"]')?.setAttribute('content', desc);
        } catch (e) {}
    }

    // Update structured JSON-LD for SPA
    function updateStructuredData(product) {
        try {
            const productScript = document.querySelector('script[type="application/ld+json"][data-ld="product"]');
            if (!productScript) return;
            const data = {
                "@context": "https://schema.org",
                "@type": "Product",
                name: product.name,
                image: (product.images || []).map(i => (i.image || i)),
                description: product.short_description || product.description || '',
                sku: product.sku || '',
                offers: {
                    "@type": "Offer",
                    url: window.location.href,
                    priceCurrency: product.currency || window.BUNORAA_PRODUCT?.currency || 'BDT',
                    price: product.current_price || product.price
                }
            };
            productScript.textContent = JSON.stringify(data);
        } catch (e) {}
    }

    // Load recommendations (frequently bought, similar, you may also like)
    async function loadRecommendations(product) {
        const container = document.getElementById('related-products');
        if (!container) return;

        try {
            const [fbt, similar, liked] = await Promise.all([
                ProductsApi.getRecommendations(product.id, 'frequently_bought_together', 3),
                ProductsApi.getRecommendations(product.id, 'similar', 4),
                ProductsApi.getRecommendations(product.id, 'you_may_also_like', 6)
            ]);

            // Render frequently bought together
            if (fbt.success && fbt.data?.length) {
                const html = `
                    <section class="mt-8">
                        <h3 class="text-lg font-semibold mb-4">Frequently Bought Together</h3>
                        <div class="grid grid-cols-3 gap-3">${(fbt.data || []).map(p => ProductCard.render(p)).join('')}</div>
                    </section>
                `;
                container.insertAdjacentHTML('beforeend', html);
            }

            // Similar
            if (similar.success && similar.data?.length) {
                const html = `
                    <section class="mt-8">
                        <h3 class="text-lg font-semibold mb-4">Similar Products</h3>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">${(similar.data || []).map(p => ProductCard.render(p)).join('')}</div>
                    </section>
                `;
                container.insertAdjacentHTML('beforeend', html);
            }

            // You may also like
            if (liked.success && liked.data?.length) {
                const html = `
                    <section class="mt-8">
                        <h3 class="text-lg font-semibold mb-4">You May Also Like</h3>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">${(liked.data || []).map(p => ProductCard.render(p)).join('')}</div>
                    </section>
                `;
                container.insertAdjacentHTML('beforeend', html);
            }

            ProductCard.bindEvents(container);
        } catch (err) {
            // ignore gracefully
        }
    }

    // Defer heavy below-the-fold sections until visible
    function initLazySections() {
        const container = document.getElementById('product-detail');
        if (!container || typeof IntersectionObserver === 'undefined') return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (!entry.isIntersecting) return;
                const target = entry.target;
                if (target.id === 'related-products') {
                    // ensure recommendations already loaded in parallel; nothing to do
                }
                if (target.id === 'reviews' || target.id === 'reviews-container') {
                    // reviews are already loaded; nothing to do
                }
                observer.unobserve(target);
            });
        }, { rootMargin: '200px' });

        document.querySelectorAll('#related-products, #reviews').forEach(el => {
            observer.observe(el);
        });
    }

    async function trackProductView(product) {
        try {
            await ProductsAPI.trackView(product.id);
        } catch (error) {
            // error logging removed
        }
    }

    function destroy() {
        currentProduct = null;
        selectedVariant = null;
        gallery = null;
        initialized = false;
    }

    return {
        init,
        destroy
    };
})();

window.ProductPage = ProductPage;
export default ProductPage;
