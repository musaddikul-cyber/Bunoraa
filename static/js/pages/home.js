/**
 * Home Page - Enhanced with Advanced E-commerce Features
 * @module pages/home
 */

const HomePage = (function() {
    'use strict';

    let heroSliderInterval = null;
    let liveVisitorCount = 0;
    let socialProofInterval = null;
    let countdownInterval = null;

    async function init() {
        // OPTIMIZED: Load critical sections first, then non-critical sections in background
        
        // Ensure page starts at the top
        window.scrollTo(0, 0);
        
        // Critical sections (above the fold) - load with Promise.all
        await Promise.all([
            loadHeroBanners(),
            loadFeaturedProducts(),
            loadNewArrivals(),
        ]);
        
        // Initialize interactive features immediately
        initNewsletterForm();
        initScrollAnimations();
        initQuickViewModal();
        
        // Non-critical sections (below the fold) - load in background without blocking
        // These will load while user is viewing the top sections
        Promise.all([
            loadCategoriesShowcase(),
            loadBestSellers(),
            loadTestimonials()
        ]).catch(err => console.error('Failed to load secondary sections:', err));
        
        // Delayed features (don't impact initial load)
        setTimeout(() => {
            initLiveVisitorCounter();
            initSocialProofPopups();
            initRecentlyViewed();
            initFlashSaleCountdown();
        }, 2000);
        
        // Non-blocking enhancements
        try {
            loadPromotions();
            loadCustomOrderCTA();
        } catch (e) {
            console.warn('Failed to load promotions/CTA:', e);
        }
    }

    // ============================================
    // ENHANCED FEATURE: Live Visitor Counter
    // ============================================
    function initLiveVisitorCounter() {
        const container = document.getElementById('live-visitors');
        if (!container) return;
        
        async function fetchAndUpdateCount() {
            try {
                // Fetch active sessions from analytics
                const response = await window.ApiClient.get('/analytics/active-visitors/', {});
                const data = response.data || response;
                
                // Get real active visitor count
                liveVisitorCount = data.active_visitors || data.count || 0;
                
                // If no real data, don't show anything (don't use fallback)
                if (liveVisitorCount === 0) {
                    container.innerHTML = '';
                    return;
                }
                
                container.innerHTML = `
                    <div class="flex items-center gap-2 px-3 py-1.5 bg-emerald-100 dark:bg-emerald-900/30 rounded-full">
                        <span class="relative flex h-2 w-2">
                            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        <span class="text-xs font-medium text-emerald-700 dark:text-emerald-300">${liveVisitorCount} browsing now</span>
                    </div>
                `;
            } catch (error) {
                console.warn('Failed to fetch active visitors:', error);
                // Don't show fallback, just log error
                container.innerHTML = '';
            }
        }
        
        // Fetch immediately and then every 8 seconds
        fetchAndUpdateCount();
        setInterval(fetchAndUpdateCount, 8000);
    }

    // ============================================
    // ENHANCED FEATURE: Social Proof Popups
    // ============================================
    function initSocialProofPopups() {
        
        let recentPurchases = [];
        let index = 0;
        let proofCount = 0;
        const maxProofs = 10; // Limit to 10 popups per session

        async function fetchRecentPurchases() {
            try {
                // Fetch real recent purchases from analytics API
                const response = await window.ApiClient.get('/analytics/recent-purchases/', {});
                recentPurchases = response.data || response.purchases || [];
                
                // If no real purchases, don't start the popup rotation
                if (recentPurchases.length === 0) {
                    return;
                }
                
                // Start showing popups after 10 seconds
                setTimeout(() => {
                    showProof();
                    socialProofInterval = setInterval(() => {
                        if (proofCount < maxProofs) showProof();
                        else clearInterval(socialProofInterval);
                    }, 30000);
                }, 10000);
            } catch (error) {
                console.warn('Failed to fetch recent purchases:', error);
                // Don't show any fallback data
            }
        }

        function showProof() {
            if (recentPurchases.length === 0 || proofCount >= maxProofs) return;
            
            const proof = recentPurchases[index];
            if (!proof) return;
            
            const popup = document.createElement('div');
            popup.className = 'social-proof-popup fixed bottom-4 left-4 z-50 max-w-xs bg-white dark:bg-stone-800 rounded-xl shadow-2xl border border-stone-200 dark:border-stone-700 p-4 transform translate-y-full opacity-0 transition-all duration-500';
            
            // Using real data structure from API
            let content = `
                <div class="flex items-start gap-3">
                    <div class="w-10 h-10 bg-emerald-100 dark:bg-emerald-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                        <svg class="w-5 h-5 text-emerald-600 dark:text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-stone-900 dark:text-white">${proof.message}</p>
                        <p class="text-xs text-stone-400 dark:text-stone-500 mt-1">${proof.time_ago}</p>
                    </div>
                </div>
            `;

            popup.innerHTML = `
                ${content}
                <button class="absolute top-2 right-2 text-stone-400 hover:text-stone-600 dark:hover:text-stone-300" onclick="this.parentElement.remove()">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                </button>
            `;

            document.body.appendChild(popup);
            proofCount++; // Increment counter
            
            // Animate in
            requestAnimationFrame(() => {
                popup.classList.remove('translate-y-full', 'opacity-0');
            });

            // Remove after 5 seconds
            setTimeout(() => {
                popup.classList.add('translate-y-full', 'opacity-0');
                setTimeout(() => popup.remove(), 500);
            }, 5000);

            index = (index + 1) % recentPurchases.length;
            
            // Stop showing if we've reached max count
            if (proofCount >= maxProofs && socialProofInterval) {
                clearInterval(socialProofInterval);
            }
        }
        // Fetch real purchases and start popup rotation
        fetchRecentPurchases();
    }

    // ============================================
    // ENHANCED FEATURE: Recently Viewed Products
    // ============================================
    function initRecentlyViewed() {
        const section = document.getElementById('recently-viewed-section');
        const container = document.getElementById('recently-viewed');
        const clearBtn = document.getElementById('clear-recently-viewed');
        
        if (!section || !container) return;

        const viewed = JSON.parse(localStorage.getItem('recentlyViewed') || '[]');
        
        if (viewed.length === 0) {
            section.classList.add('hidden');
            return;
        }

        section.classList.remove('hidden');
        
        container.innerHTML = viewed.slice(0, 5).map(product => {
            let badge = null;
            if (product.discount_percent && product.discount_percent > 0) {
                badge = `-${product.discount_percent}%`;
            }
            return ProductCard.render(product, { showBadge: !!badge, badge, priceSize: 'small' });
        }).join('');
        
        ProductCard.bindEvents(container);

        clearBtn?.addEventListener('click', () => {
            localStorage.removeItem('recentlyViewed');
            section.classList.add('hidden');
            Toast.success('Recently viewed items cleared');
        });
    }

    // ============================================
    // ENHANCED FEATURE: Flash Sale Countdown
    // ============================================
    function initFlashSaleCountdown() {
        const section = document.getElementById('flash-sale-section');
        const countdown = document.getElementById('flash-sale-countdown');
        
        if (!section || !countdown) return;

        // Check if there's an active flash sale (in production, fetch from API)
        const saleEndTime = localStorage.getItem('flashSaleEnd');
        
        if (!saleEndTime) {
            // Set a mock flash sale for demo (ends in 4 hours)
            const endTime = new Date().getTime() + (4 * 60 * 60 * 1000);
            localStorage.setItem('flashSaleEnd', endTime.toString());
        }

        const endTime = parseInt(localStorage.getItem('flashSaleEnd'));
        
        function updateCountdown() {
            const now = new Date().getTime();
            const distance = endTime - now;

            if (distance <= 0) {
                section.classList.add('hidden');
                clearInterval(countdownInterval);
                localStorage.removeItem('flashSaleEnd');
                return;
            }

            section.classList.remove('hidden');

            const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
            const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
            const seconds = Math.floor((distance % (1000 * 60)) / 1000);

            countdown.innerHTML = `
                <div class="flex items-center gap-2 text-white">
                    <span class="text-sm font-medium">Ends in:</span>
                    <div class="flex items-center gap-1">
                        <span class="bg-white/20 px-2 py-1 rounded font-mono font-bold">${hours.toString().padStart(2, '0')}</span>
                        <span class="font-bold">:</span>
                        <span class="bg-white/20 px-2 py-1 rounded font-mono font-bold">${minutes.toString().padStart(2, '0')}</span>
                        <span class="font-bold">:</span>
                        <span class="bg-white/20 px-2 py-1 rounded font-mono font-bold">${seconds.toString().padStart(2, '0')}</span>
                    </div>
                </div>
            `;
        }

        updateCountdown();
        countdownInterval = setInterval(updateCountdown, 1000);
    }

    // // ============================================
    // // ENHANCED FEATURE: Artisan Spotlight
    // // ============================================
    // async function initArtisanSpotlight() {
    //     const container = document.getElementById('artisan-spotlight');
    //     if (!container) return;
    //     try {
    //         // In production, fetch from API
    //         const artisans = [
    //             {
    //                 name: 'Sarah Chen',
    //                 specialty: 'Ceramic Art',
    //                 image: '/static/images/artisans/sarah.jpg',
    //                 story: 'Third-generation potter from the mountain villages, crafting unique pieces for over 15 years.',
    //                 products: 45,
    //                 rating: 4.9
    //             },
    //             {
    //                 name: 'Ahmed Hassan',
    //                 specialty: 'Leatherwork',
    //                 image: '/static/images/artisans/ahmed.jpg',
    //                 story: 'Master craftsman preserving traditional techniques passed down through generations.',
    //                 products: 32,
    //                 rating: 4.8
    //             },
    //             {
    //                 name: 'Maria Santos',
    //                 specialty: 'Textile Weaving',
    //                 image: '/static/images/artisans/maria.jpg',
    //                 story: 'Creating vibrant handwoven textiles using natural dyes and ancestral patterns.',
    //                 products: 28,
    //                 rating: 4.9
    //             }
    //         ];

    //         container.innerHTML = `
    //             <div class="grid md:grid-cols-3 gap-6">
    //                 ${artisans.map(artisan => `
    //                     <div class="group relative bg-white dark:bg-stone-800 rounded-2xl overflow-hidden shadow-lg hover:shadow-xl transition-shadow">
    //                         <div class="aspect-[4/3] overflow-hidden bg-stone-100 dark:bg-stone-700">
    //                             <img src="${artisan.image}" alt="${Templates.escapeHtml(artisan.name)}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" loading="lazy" decoding="async">
    //                         </div>
    //                         <div class="p-6">
    //                             <div class="flex items-center justify-between mb-2">
    //                                 <h3 class="text-lg font-semibold text-stone-900 dark:text-white">${Templates.escapeHtml(artisan.name)}</h3>
    //                                 <div class="flex items-center gap-1 text-amber-500">
    //                                     <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.178c.969 0 1.371 1.24.588 1.81l-3.385 2.46a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.385-2.46a1 1 0 00-1.175 0l-3.385 2.46c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118l-3.385-2.46c-.783-.57-.38-1.81.588-1.81h4.178a1 1 0 00.95-.69l1.286-3.967z"/></svg>
    //                                     <span class="text-sm font-medium">${artisan.rating}</span>
    //                                 </div>
    //                             </div>
    //                             <p class="text-sm text-primary-600 dark:text-amber-400 font-medium mb-3">${Templates.escapeHtml(artisan.specialty)}</p>
    //                             <p class="text-sm text-stone-600 dark:text-stone-400 mb-4 line-clamp-2">${Templates.escapeHtml(artisan.story)}</p>
    //                             <div class="flex items-center justify-between">
    //                                 <span class="text-xs text-stone-500 dark:text-stone-500">${artisan.products} products</span>
    //                                 <a href="/artisans/${artisan.name.toLowerCase().replace(/\s+/g, '-')}/" class="text-sm font-medium text-primary-600 dark:text-amber-400 hover:underline">View Profile →</a>
    //                             </div>
    //                         </div>
    //                     </div>
    //                 `).join('')}
    //             </div>
    //         `;
    //     } catch (error) {
    //         console.warn('Artisan spotlight unavailable:', error);
    //     }
    // }

    // ============================================
    // ENHANCED FEATURE: Scroll Animations
    // ============================================
    function initScrollAnimations() {
        const animatedElements = document.querySelectorAll('[data-animate]');
        
        if (!animatedElements.length) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const animation = entry.target.dataset.animate || 'fadeInUp';
                    entry.target.classList.add('animate-' + animation);
                    entry.target.classList.remove('opacity-0');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        animatedElements.forEach(el => {
            el.classList.add('opacity-0');
            observer.observe(el);
        });
    }

    // ============================================
    // ENHANCED FEATURE: Quick View Modal
    // ============================================
    function initQuickViewModal() {
        document.addEventListener('click', async (e) => {
            const quickViewBtn = e.target.closest('[data-quick-view]');
            if (!quickViewBtn) return;

            const productId = quickViewBtn.dataset.quickView;
            if (!productId) return;

            e.preventDefault();
            
            // Show loading modal
            const modal = document.createElement('div');
            modal.id = 'quick-view-modal';
            modal.className = 'fixed inset-0 z-50 flex items-center justify-center p-4';
            modal.innerHTML = `
                <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" onclick="document.getElementById('quick-view-modal').remove()"></div>
                <div class="relative bg-white dark:bg-stone-800 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-auto">
                    <button class="absolute top-4 right-4 z-10 w-10 h-10 bg-stone-100 dark:bg-stone-700 rounded-full flex items-center justify-center hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors" onclick="document.getElementById('quick-view-modal').remove()">
                        <svg class="w-5 h-5 text-stone-600 dark:text-stone-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                    <div class="p-8 flex items-center justify-center min-h-[400px]">
                        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
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
                        <div class="aspect-square rounded-xl overflow-hidden bg-stone-100 dark:bg-stone-700">
                            <img src="${product.primary_image || product.image || '/static/images/placeholder.jpg'}" alt="${Templates.escapeHtml(product.name)}" class="w-full h-full object-cover">
                        </div>
                        <div class="flex flex-col">
                            <h2 class="text-2xl font-bold text-stone-900 dark:text-white mb-2">${Templates.escapeHtml(product.name)}</h2>
                            <div class="flex items-center gap-2 mb-4">
                                <div class="flex text-amber-400">
                                    ${'<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.178c.969 0 1.371 1.24.588 1.81l-3.385 2.46a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.385-2.46a1 1 0 00-1.175 0l-3.385 2.46c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118l-3.385-2.46c-.783-.57-.38-1.81.588-1.81h4.178a1 1 0 00.95-.69l1.286-3.967z"/></svg>'.repeat(Math.round(product.rating || 4))}
                                </div>
                                <span class="text-sm text-stone-500 dark:text-stone-400">(${product.review_count || 0} reviews)</span>
                            </div>
                            <div class="mb-6">
                                ${product.sale_price || product.discounted_price ? `
                                    <span class="text-3xl font-bold text-primary-600 dark:text-amber-400">${Templates.formatPrice(product.sale_price || product.discounted_price)}</span>
                                    <span class="text-lg text-stone-400 line-through ml-2">${Templates.formatPrice(product.price)}</span>
                                ` : `
                                    <span class="text-3xl font-bold text-stone-900 dark:text-white">${Templates.formatPrice(product.price)}</span>
                                `}
                            </div>
                            <p class="text-stone-600 dark:text-stone-400 mb-6 line-clamp-3">${Templates.escapeHtml(product.short_description || product.description || '')}</p>
                            <div class="mt-auto space-y-3">
                                <button class="w-full py-3 px-6 bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-xl transition-colors" onclick="CartApi.addItem(${product.id}, 1).then(() => { Toast.success('Added to cart'); document.getElementById('quick-view-modal').remove(); })">
                                    Add to Cart
                                </button>
                                <a href="/products/${product.slug || product.id}/" class="block w-full py-3 px-6 border-2 border-stone-200 dark:border-stone-600 text-stone-700 dark:text-stone-300 font-semibold rounded-xl text-center hover:bg-stone-50 dark:hover:bg-stone-700 transition-colors">
                                    View Full Details
                                </a>
                            </div>
                        </div>
                    </div>
                `;
            } catch (error) {
                console.error('Failed to load product:', error);
                modal.remove();
                Toast.error('Failed to load product details');
            }
        });
    }


        async function loadTestimonials() {
            const container = document.getElementById('testimonials-grid');
            if (!container) return;
            Loader.show(container, 'skeleton');
            try {
                // OPTIMIZED: Fetch reviews directly instead of looping through categories->products->reviews
                // This reduces API calls from 50+ to just 1
                let reviewsResponse = await ProductsApi.getReviews(null, { pageSize: 6, orderBy: '-rating' });
                let allReviews = reviewsResponse?.data?.results || reviewsResponse?.data || reviewsResponse?.results || [];
                
                container.innerHTML = '';
                if (!allReviews.length) {
                    container.innerHTML = '<p class="text-gray-500 text-center py-8">No user reviews available.</p>';
                    return;
                }
                
                // Limit to 6 reviews
                allReviews = allReviews.slice(0, 6);
                allReviews.forEach(review => {
                    const card = document.createElement('div');
                    card.className = 'rounded-2xl bg-white dark:bg-stone-800 shadow p-6 flex flex-col gap-3';
                    card.innerHTML = `
                        <div class="flex items-center gap-3 mb-2">
                            <div class="w-10 h-10 rounded-full bg-primary-100 dark:bg-stone-700 flex items-center justify-center text-lg font-bold text-primary-700 dark:text-amber-400">
                                ${review.user?.first_name?.[0] || review.user?.username?.[0] || '?'}
                            </div>
                            <div>
                                <div class="font-semibold text-gray-900 dark:text-white">${review.user?.first_name || review.user?.username || 'Anonymous'}</div>
                                <div class="text-xs text-gray-500 dark:text-stone-400">${review.created_at ? new Date(review.created_at).toLocaleDateString() : ''}</div>
                            </div>
                        </div>
                        <div class="flex gap-1 mb-1">
                            ${'<svg class="w-4 h-4 text-amber-400" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.178c.969 0 1.371 1.24.588 1.81l-3.385 2.46a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.385-2.46a1 1 0 00-1.175 0l-3.385 2.46c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118l-3.385-2.46c-.783-.57-.38-1.81.588-1.81h4.178a1 1 0 00.95-.69l1.286-3.967z"/></svg>'.repeat(Math.round(review.rating || 5))}
                        </div>
                        <div class="text-gray-800 dark:text-stone-200 text-base mb-2">${Templates.escapeHtml(review.title || '')}</div>
                        <div class="text-gray-600 dark:text-stone-400 text-sm">${Templates.escapeHtml(review.content || '')}</div>
                    `;
                    container.appendChild(card);
                });
            } catch (error) {
                console.error('Failed to load testimonials:', error);
                container.innerHTML = '<p class="text-red-500 text-center py-8">Failed to load reviews. Please try again later.</p>';
            }
    }
    async function loadBestSellers() {
        const container = document.getElementById('best-sellers');
        if (!container) return;

        const grid = container.querySelector('.products-grid') || container;
        Loader.show(grid, 'skeleton');

        try {
            // Use backend bestseller logic, like featured
            const response = await ProductsApi.getProducts({ bestseller: true, pageSize: 5 });
            const products = response.data?.results || response.data || response.results || [];

            if (products.length === 0) {
                grid.innerHTML = '<p class="text-gray-500 text-center py-8">No best sellers available.</p>';
                return;
            }

            grid.innerHTML = products.map(product => {
                // Only show discount badge if available
                let badge = null;
                if (product.discount_percent && product.discount_percent > 0) {
                    badge = `-${product.discount_percent}%`;
                }
                return ProductCard.render(product, { showBadge: !!badge, badge, priceSize: 'small' });
            }).join('');
            ProductCard.bindEvents(grid);
        } catch (error) {
            console.error('Failed to load best sellers:', error);
            grid.innerHTML = '<p class="text-red-500 text-center py-8">Failed to load products. Please try again later.</p>';
        }
    }

    async function loadHeroBanners() {
        const container = document.getElementById('hero-slider');
        if (!container) return;

        try {
            const response = await PagesApi.getBanners('home_hero');
            const banners = response.data?.results || response.data || response.results || [];

            if (banners.length === 0) {
                container.innerHTML = '';
                return;
            }

            container.innerHTML = `
                <div class="relative overflow-hidden w-full h-[55vh] sm:h-[70vh] md:h-[80vh]">
                    <div class="hero-slides relative w-full h-full">
                        ${banners.map((banner, index) => `
                            <div class="hero-slide ${index === 0 ? '' : 'hidden'} w-full h-full" data-index="${index}">
                                <a href="${banner.link_url || '#'}" class="block relative w-full h-full">
                                    <img 
                                        src="${banner.image}" 
                                        alt="${Templates.escapeHtml(banner.title || '')}"
                                        class="absolute inset-0 w-full h-full object-cover"
                                        loading="${index === 0 ? 'eager' : 'lazy'}"
                                        decoding="async"
                                    >
                                    ${banner.title || banner.subtitle ? `
                                        <div class="absolute inset-0 bg-gradient-to-r from-black/60 via-black/30 to-transparent flex items-center">
                                            <div class="px-8 md:px-16 max-w-xl">
                                                ${banner.title ? `<h2 class="text-2xl sm:text-3xl md:text-5xl font-bold text-white mb-4">${Templates.escapeHtml(banner.title)}</h2>` : ''}
                                                ${banner.subtitle ? `<p class="text-sm sm:text-lg text-white/90 mb-6">${Templates.escapeHtml(banner.subtitle)}</p>` : ''}
                                                ${(banner.link_text || banner.button_text) ? `
                                                    <span class="inline-flex items-center px-6 py-3 bg-white text-gray-900 font-semibold rounded-lg hover:bg-gray-100 transition-colors">
                                                        ${Templates.escapeHtml(banner.link_text || banner.button_text)}
                                                        <svg class="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/>
                                                        </svg>
                                                    </span>
                                                ` : ''}
                                            </div>
                                        </div>
                                    ` : ''}
                                </a>
                            </div>
                        `).join('')}
                    </div>
                    ${banners.length > 1 ? `
                        <button class="hero-prev absolute left-4 top-1/2 -translate-y-1/2 w-10 h-10 bg-white/30 dark:bg-stone-800/30 hover:bg-white/40 dark:hover:bg-stone-700/40 rounded-full text-stone-900 dark:text-stone-100 flex items-center justify-center shadow-lg transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500" aria-label="Previous slide">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                            </svg>
                        </button>
                        <button class="hero-next absolute right-4 top-1/2 -translate-y-1/2 w-10 h-10 bg-white/30 dark:bg-stone-800/30 hover:bg-white/40 dark:hover:bg-stone-700/40 rounded-full text-stone-900 dark:text-stone-100 flex items-center justify-center shadow-lg transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500" aria-label="Next slide">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                            </svg>
                        </button>
                        <div class="hero-dots absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2">
                            ${banners.map((_, index) => `
                                <button class="w-3 h-3 rounded-full transition-colors ${index === 0 ? 'bg-white dark:bg-stone-200' : 'bg-white/50 dark:bg-stone-600/60 hover:bg-white/75 dark:hover:bg-stone-500/80'}" data-slide="${index}" aria-label="Go to slide ${index + 1}"></button>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `;

            if (banners.length > 1) {
                initHeroSlider(banners.length);
            }
        } catch (error) {
            // Gracefully ignore missing banners endpoint and render a simple fallback hero
            console.warn('Hero banners unavailable:', error?.status || error);
        }
    }

    function initHeroSlider(totalSlides) {
        let currentSlide = 0;
        const slides = document.querySelectorAll('.hero-slide');
        const dots = document.querySelectorAll('.hero-dots button');
        const prevBtn = document.querySelector('.hero-prev');
        const nextBtn = document.querySelector('.hero-next');

        function goToSlide(index) {
            slides[currentSlide].classList.add('hidden');
            dots[currentSlide]?.classList.remove('bg-stone-100');
            dots[currentSlide]?.classList.add('bg-white/50');

            currentSlide = (index + totalSlides) % totalSlides;

            slides[currentSlide].classList.remove('hidden');
            dots[currentSlide]?.classList.add('bg-stone-100');
            dots[currentSlide]?.classList.remove('bg-white/50');
        }

        prevBtn?.addEventListener('click', () => {
            goToSlide(currentSlide - 1);
            resetAutoplay();
        });

        nextBtn?.addEventListener('click', () => {
            goToSlide(currentSlide + 1);
            resetAutoplay();
        });

        dots.forEach((dot, index) => {
            dot.addEventListener('click', () => {
                goToSlide(index);
                resetAutoplay();
            });
        });

        function resetAutoplay() {
            if (heroSliderInterval) {
                clearInterval(heroSliderInterval);
            }
            heroSliderInterval = setInterval(() => goToSlide(currentSlide + 1), 5000);
        }

        // Add simple touch/swipe support for mobile
        try {
            const slidesContainer = document.querySelector('.hero-slides');
            let touchStartX = 0;
            slidesContainer?.addEventListener('touchstart', (e) => {
                touchStartX = e.touches[0].clientX;
            }, { passive: true });
            slidesContainer?.addEventListener('touchend', (e) => {
                const touchEndX = e.changedTouches[0].clientX;
                const dx = touchEndX - touchStartX;
                if (Math.abs(dx) > 50) {
                    if (dx < 0) {
                        goToSlide(currentSlide + 1);
                    } else {
                        goToSlide(currentSlide - 1);
                    }
                    resetAutoplay();
                }
            });
        } catch (err) {
            // ignore if touch listeners fail
        }

        resetAutoplay();
    }

    async function loadFeaturedProducts() {
        const container = document.getElementById('featured-products');
        if (!container) return;

        const grid = container.querySelector('.products-grid') || container;
        Loader.show(grid, 'skeleton');

        try {
            const response = await ProductsApi.getFeatured(8);
            const products = response.data?.results || response.data || response.results || [];

            if (products.length === 0) {
                grid.innerHTML = '<p class="text-gray-500 text-center py-8">No featured products available.</p>';
                return;
            }

            grid.innerHTML = products.map(product => {
                let badge = null;
                if (product.discount_percent && product.discount_percent > 0) {
                    badge = `-${product.discount_percent}%`;
                }
                return ProductCard.render(product, { showBadge: !!badge, badge, priceSize: 'small' });
            }).join('');
            ProductCard.bindEvents(grid);
        } catch (error) {
            console.error('Failed to load featured products:', error);
            grid.innerHTML = '<p class="text-red-500 text-center py-8">Failed to load products. Please try again later.</p>';
        }
    }

    async function loadCategoriesShowcase() {
        const container = document.getElementById('categories-showcase');
        if (!container) return;



        Loader.show(container, 'skeleton');

        try {
            // Clear any client cache for categories and fetch fresh data
            try { window.ApiClient?.clearCache('/api/v1/catalog/categories/'); } catch (e) {}
            const response = await window.ApiClient.get('/catalog/categories/', { page_size: 6, is_featured: true }, { useCache: false });
            // info removed
            const categories = response.data?.results || response.data || response.results || [];

            if (categories.length === 0) {
                container.innerHTML = '';
                return;
            }

            // Import CategoryCard component
            let CategoryCard;
            try {
                CategoryCard = (await import('../components/CategoryCard.js')).CategoryCard;
            } catch (e) {
                console.error('Failed to import CategoryCard:', e);
                return;
            }

            container.innerHTML = '';
            // info removed
            categories.forEach(category => {
                const card = CategoryCard(category);
                // Log whether the created card contains an <img> and its src (if present)
                try {
                    const imgEl = card.querySelector('img');
                    console.info('[Home] card image for', category.name, imgEl ? imgEl.src : 'NO_IMAGE');
                } catch (e) { /* error logging removed */ }
                container.appendChild(card);
            });
            container.classList.add('grid','grid-cols-2','sm:grid-cols-2','md:grid-cols-3','lg:grid-cols-6','gap-4','lg:gap-6');
        } catch (error) {
            console.error('Failed to load categories:', error);
            container.innerHTML = '';
        }
    }

    async function loadNewArrivals() {
        const container = document.getElementById('new-arrivals');
        if (!container) return;

        const grid = container.querySelector('.products-grid') || container;
        Loader.show(grid, 'skeleton');

        try {
            const response = await ProductsApi.getNewArrivals(4);
            const products = response.data?.results || response.data || response.results || [];

            if (products.length === 0) {
                grid.innerHTML = '<p class="text-gray-500 text-center py-8">No new products available.</p>';
                return;
            }

            grid.innerHTML = products.map(product => {
                let badge = null;
                if (product.discount_percent && product.discount_percent > 0) {
                    badge = `-${product.discount_percent}%`;
                }
                return ProductCard.render(product, { showBadge: !!badge, badge, priceSize: 'small' });
            }).join('');
            ProductCard.bindEvents(grid);
        } catch (error) {
            console.error('Failed to load new arrivals:', error);
            grid.innerHTML = '<p class="text-red-500 text-center py-8">Failed to load products.</p>';
        }
    }

    async function loadPromotions() {
        const container = document.getElementById('promotions-banner') || document.getElementById('promotion-banners');
        if (!container) return;

        try {
            const response = await PagesApi.getPromotions();
            let promotions = response?.data?.results ?? response?.results ?? response?.data ?? [];
            if (!Array.isArray(promotions)) {
                if (promotions && typeof promotions === 'object') {
                    promotions = Array.isArray(promotions.items) ? promotions.items : [promotions];
                } else {
                    promotions = [];
                }
            }

            if (promotions.length === 0) {
                container.innerHTML = '';
                return;
            }

            const promo = promotions[0] || {};
            container.innerHTML = `
                <div class="bg-gradient-to-r from-primary-600 to-primary-700 rounded-2xl overflow-hidden">
                    <div class="px-6 py-8 md:px-12 md:py-12 flex flex-col md:flex-row items-center justify-between gap-6">
                        <div class="text-center md:text-left">
                            <span class="inline-block px-3 py-1 bg-white/20 text-white text-sm font-medium rounded-full mb-3">
                                Limited Time Offer
                            </span>
                            <h3 class="text-2xl md:text-3xl font-bold text-white mb-2">
                                ${Templates.escapeHtml(promo.title || promo.name || '')}
                            </h3>
                            ${promo.description ? `
                                <p class="text-white/90 max-w-lg">${Templates.escapeHtml(promo.description)}</p>
                            ` : ''}
                            ${promo.discount_value ? `
                                <p class="text-3xl font-bold text-white mt-4">
                                    ${promo.discount_type === 'percentage' ? `${promo.discount_value}% OFF` : `Save ${Templates.formatPrice(promo.discount_value)}`}
                                </p>
                            ` : ''}
                        </div>
                        <div class="flex flex-col items-center gap-4">
                            ${promo.code ? `
                                <div class="bg-white/10 backdrop-blur-sm px-6 py-3 rounded-lg border-2 border-dashed border-white/30">
                                    <p class="text-sm text-white/80 mb-1">Use code:</p>
                                    <p class="text-2xl font-mono font-bold text-white tracking-wider">${Templates.escapeHtml(promo.code)}</p>
                                </div>
                            ` : ''}
                            <a href="/products/?promotion=${promo.id || ''}" class="inline-flex items-center px-6 py-3 bg-stone-100 text-primary-600 font-semibold rounded-lg hover:bg-gray-100 transition-colors">
                                Shop Now
                                <svg class="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/>
                                </svg>
                            </a>
                        </div>
                    </div>
                </div>
            `;
        } catch (error) {
            console.warn('Promotions unavailable:', error?.status || error);
            container.innerHTML = '';
        }
    }

    async function loadCustomOrderCTA() {
        const container = document.getElementById('custom-order-cta');
        if (!container) return;
        // prevent double initialization
        if (container.dataset?.loaded) return;
        container.dataset.loaded = '1';

        // Insert a lightweight skeleton first to avoid layout shift / flash
        container.innerHTML = `
            <div class="container mx-auto px-4">
                <div class="max-w-full w-full mx-auto rounded-3xl p-6 md:p-10">
                    <div class="animate-pulse">
                        <div class="h-6 w-1/3 bg-gray-200 dark:bg-stone-700 rounded mb-4"></div>
                        <div class="h-10 w-full bg-gray-200 dark:bg-stone-700 rounded mb-4"></div>
                        <div class="h-44 bg-gray-200 dark:bg-stone-800 rounded"></div>
                    </div>
                </div>
            </div>
        `;

        const routeMap = window.BUNORAA_ROUTES || {};
        const wizardUrl = routeMap.preordersWizard || '/preorders/create/';
        const landingUrl = routeMap.preordersLanding || '/preorders/';

        try {
            // Try to load featured pre-order categories
            let categories = [];
            if (typeof PreordersApi !== 'undefined' && PreordersApi.getCategories) {
                try {
                    const response = await PreordersApi.getCategories({ featured: true, pageSize: 4 });
                    categories = response?.data?.results || response?.data || response?.results || [];
                } catch (e) {
                    console.warn('Pre-order categories unavailable:', e);
                }
            }

            container.innerHTML = `
                <div class="container mx-auto px-4 relative">
                    <div class="max-w-full w-full mx-auto rounded-3xl shadow-lg overflow-hidden bg-white dark:bg-neutral-900 p-6 md:p-10 border border-stone-100 dark:border-stone-700">
                      <div class="grid lg:grid-cols-2 gap-12 items-center">
                        <div class="text-stone-900 dark:text-white">
                            <span class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-100/40 dark:bg-amber-700/20 text-xs uppercase tracking-[0.2em] mb-6 text-amber-800 dark:text-white">
                                <span class="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></span>
                                Made Just For You
                            </span>
                            <h2 class="text-3xl lg:text-5xl font-display font-bold mb-6 leading-tight text-stone-900 dark:text-white">Create Your Perfect Custom Order</h2>
                            <p class="text-stone-700 dark:text-white/80 text-lg mb-8 max-w-xl">Have a unique vision? Our skilled artisans will bring your ideas to life. From personalized gifts to custom designs, we craft exactly what you need.</p>
                            <div class="grid sm:grid-cols-3 gap-4 mb-8">
                                <div class="flex items-center gap-3 bg-white/5 dark:bg-stone-800/40 backdrop-blur-sm rounded-xl p-4 border border-stone-100 dark:border-stone-700">
                                    <div class="w-10 h-10 bg-purple-500/30 rounded-lg flex items-center justify-center">
                                        <svg class="w-5 h-5 text-purple-800" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>
                                    </div>
                                    <div>
                                        <p class="text-sm font-semibold text-stone-900 dark:text-white">Custom Design</p>
                                        <p class="text-xs text-stone-600 dark:text-white/60">Your vision, our craft</p>
                                    </div>
                                </div>
                                <div class="flex items-center gap-3 bg-white/5 dark:bg-stone-800/40 backdrop-blur-sm rounded-xl p-4 border border-stone-100 dark:border-stone-700">
                                    <div class="w-10 h-10 bg-indigo-500/30 rounded-lg flex items-center justify-center">
                                        <svg class="w-5 h-5 text-indigo-700 dark:text-indigo-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
                                    </div>
                                    <div>
                                        <p class="text-sm font-semibold text-stone-900 dark:text-white">Direct Chat</p>
                                        <p class="text-xs text-stone-600 dark:text-white/60">Talk to artisans</p>
                                    </div>
                                </div>
                                <div class="flex items-center gap-3 bg-white/5 dark:bg-stone-800/40 backdrop-blur-sm rounded-xl p-4 border border-stone-100 dark:border-stone-700">
                                    <div class="w-10 h-10 bg-pink-500/30 rounded-lg flex items-center justify-center">
                                        <svg class="w-5 h-5 text-pink-700 dark:text-pink-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                                    </div>
                                    <div>
                                        <p class="text-sm font-semibold text-stone-900 dark:text-white">Quality Assured</p>
                                        <p class="text-xs text-stone-600 dark:text-white/60">Satisfaction guaranteed</p>
                                    </div>
                                </div>
                            </div>
                            ${categories.length > 0 ? `
                                <div class="mb-8">
                                    <p class="text-stone-600 dark:text-white/60 text-sm mb-3">Popular categories:</p>
                                    <div class="flex flex-wrap gap-2">
                                        ${categories.slice(0, 4).map(cat => `
                                            <a href="${landingUrl}category/${cat.slug}/" class="inline-flex items-center gap-2 px-4 py-2 bg-white/10 dark:bg-stone-800/30 hover:bg-white/20 dark:hover:bg-stone-700 rounded-full text-sm text-stone-900 dark:text-white transition-colors">
                                                ${cat.icon ? `<span>${cat.icon}</span>` : ''}
                                                ${Templates.escapeHtml(cat.name)}
                                            </a>
                                        `).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            <div class="flex flex-wrap gap-4">
                                <a href="${wizardUrl}" class="cta-unlock inline-flex items-center gap-2 px-8 py-4 bg-amber-600 text-white font-bold rounded-xl shadow-lg hover:shadow-xl hover:text-black dark:hover:text-black transition-colors group dark:bg-amber-600 dark:text-white">
                                    Start Your Custom Order
                                    <svg class="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/></svg>
                                </a>
                                <a href="${landingUrl}" class="inline-flex items-center gap-2 px-8 py-4 bg-transparent text-stone-900 dark:text-white font-semibold rounded-xl border-2 border-stone-200 dark:border-stone-700 hover:bg-stone-100 dark:hover:bg-stone-800 transition-all">
                                    Learn More
                                </a>
                            </div>
                        </div>
                        <div class="hidden lg:block">
                            <div class="relative">
                                <div class="absolute -inset-4 bg-gradient-to-r from-purple-500/30 to-indigo-500/30 rounded-3xl blur-2xl"></div>
                                <div class="relative bg-white/5 dark:bg-stone-800/40 backdrop-blur-md rounded-3xl p-8 border border-stone-100 dark:border-stone-700">
                                    <div class="space-y-6">
                                        <div class="flex items-start gap-4">
                                            <div class="w-12 h-12 bg-purple-600 rounded-xl flex items-center justify-center flex-shrink-0 text-white text-xl font-bold shadow-sm ring-1 ring-stone-100 dark:ring-stone-700">1</div>
                                            <div>
                                                <h4 class="text-stone-900 dark:text-white font-semibold mb-1">Choose Category</h4>
                                                <p class="text-stone-600 dark:text-white/60 text-sm">Select from custom apparel, gifts, home decor & more</p>
                                            </div>
                                        </div>
                                        <div class="flex items-start gap-4">
                                            <div class="w-12 h-12 bg-indigo-600 rounded-xl flex items-center justify-center flex-shrink-0 text-white text-xl font-bold shadow-sm ring-1 ring-stone-100 dark:ring-stone-700">2</div>
                                            <div>
                                                <h4 class="text-stone-900 dark:text-white font-semibold mb-1">Share Your Vision</h4>
                                                <p class="text-stone-600 dark:text-white/60 text-sm">Upload designs, describe your requirements</p>
                                            </div>
                                        </div>
                                        <div class="flex items-start gap-4">
                                            <div class="w-12 h-12 bg-amber-600 rounded-xl flex items-center justify-center flex-shrink-0 text-white text-xl font-bold shadow-sm ring-1 ring-stone-100 dark:ring-stone-700">3</div>
                                            <div>
                                                <h4 class="text-stone-900 dark:text-white font-semibold mb-1">Get Your Quote</h4>
                                                <p class="text-stone-600 dark:text-white/60 text-sm">Receive pricing and timeline from our team</p>
                                            </div>
                                        </div>
                                        <div class="flex items-start gap-4">
                                            <div class="w-12 h-12 bg-emerald-600 rounded-xl flex items-center justify-center flex-shrink-0 text-white text-xl font-bold shadow-sm ring-1 ring-stone-100 dark:ring-stone-700">4</div>
                                            <div>
                                                <h4 class="text-stone-900 dark:text-white font-semibold mb-1">We Create & Deliver</h4>
                                                <p class="text-stone-600 dark:text-white/60 text-sm">Track progress and receive your masterpiece</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } catch (error) {
            console.warn('Custom order CTA failed to load:', error);
            // Fallback static content
            container.innerHTML = `
                <div class="container mx-auto px-4 text-center text-stone-900 dark:text-white">
                    <h2 class="text-3xl lg:text-4xl font-display font-bold mb-4 text-stone-900 dark:text-white">Create Your Perfect Custom Order</h2>
                    <p class="text-stone-700 dark:text-white/80 mb-8 max-w-2xl mx-auto">Have a unique vision? Our skilled artisans will bring your ideas to life.</p>
                    <a href="${wizardUrl}" class="cta-unlock inline-flex items-center gap-2 px-8 py-4 bg-amber-600 text-white font-bold rounded-xl shadow-lg hover:shadow-xl hover:text-black dark:hover:text-black transition-colors group dark:bg-amber-600 dark:text-white">
                        Start Your Custom Order
                    </a>
                </div>
            `;
        }
    }

    function initNewsletterForm() {
        // Support both IDs for compatibility
        const form = document.getElementById('newsletter-form') || document.getElementById('newsletter-form-home');
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const emailInput = form.querySelector('input[type="email"]');
            const submitBtn = form.querySelector('button[type="submit"]');
            const email = emailInput?.value?.trim();

            if (!email) {
                Toast.error('Please enter your email address.');
                return;
            }

            const originalText = submitBtn.textContent;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<svg class="animate-spin h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';

            try {
                await SupportApi.submitContactForm({ email, type: 'newsletter' });
                Toast.success('Thank you for subscribing!');
                emailInput.value = '';
            } catch (error) {
                Toast.error(error.message || 'Failed to subscribe. Please try again.');
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = originalText;
            }
        });
    }

    function destroy() {
        if (heroSliderInterval) {
            clearInterval(heroSliderInterval);
            heroSliderInterval = null;
        }
        if (socialProofInterval) {
            clearInterval(socialProofInterval);
            socialProofInterval = null;
        }
        if (countdownInterval) {
            clearInterval(countdownInterval);
            countdownInterval = null;
        }
        // Remove quick view modal
        document.getElementById('quick-view-modal')?.remove();
        // Remove social proof popups
        document.querySelectorAll('.social-proof-popup').forEach(el => el.remove());
    }

    return {
        init,
        destroy,
        // Expose for external use
        initRecentlyViewed,
        initFlashSaleCountdown
    };
})();

window.HomePage = HomePage;
export default HomePage;
