/**
 * Bunoraa ML Tracking Library
 * 
 * Comprehensive client-side tracking for ML model training.
 * Silently collects all user behavior data including:
 * - Page views and navigation
 * - Time on page and product
 * - Scroll depth and viewport
 * - Clicks and interactions
 * - Product interactions
 * - Cart and wishlist events
 * - Search behavior
 * - User engagement metrics
 * 
 * Usage:
 *   <script src="/static/js/ml-tracking.js" data-api-url="/api/ml/track/"></script>
 */

(function(window, document) {
    'use strict';

    // Configuration
    const CONFIG = {
        apiUrl: document.currentScript?.dataset?.apiUrl || '/api/ml/track/',
        batchSize: 10,
        flushInterval: 5000, // 5 seconds
        heartbeatInterval: 30000, // 30 seconds
        scrollThreshold: 10, // Minimum scroll change to track
        idleTimeout: 60000, // 1 minute
        storage: {
            prefix: 'bunoraa_ml_',
            sessionExpiry: 30 * 60 * 1000, // 30 minutes
        },
    };

    // Event queue
    let eventQueue = [];
    let flushTimer = null;
    let heartbeatTimer = null;

    // Session state
    let sessionData = {
        sessionId: null,
        userId: null,
        anonymousId: null,
        startTime: Date.now(),
        lastActivity: Date.now(),
        pageViews: 0,
        interactions: 0,
        totalScrollDepth: 0,
        maxScrollDepth: 0,
        currentPage: {
            url: window.location.href,
            startTime: Date.now(),
            scrollDepth: 0,
            activeTime: 0,
            clicks: 0,
            isIdle: false,
        },
    };

    // User profile data
    let userProfile = {
        device: null,
        browser: null,
        os: null,
        screen: null,
        viewport: null,
        language: null,
        timezone: null,
        referrer: null,
        theme: null,
    };

    // Active time tracking
    let activeTimeTracker = {
        isActive: true,
        lastActiveTime: Date.now(),
        totalActiveTime: 0,
        idleStart: null,
    };

    // ================================
    // Utility Functions
    // ================================

    function generateId() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
        return null;
    }

    function setCookie(name, value, days = 365) {
        const expires = new Date(Date.now() + days * 24 * 60 * 60 * 1000).toUTCString();
        document.cookie = `${name}=${value}; expires=${expires}; path=/; SameSite=Lax`;
    }

    function getLocalStorage(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(CONFIG.storage.prefix + key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            return defaultValue;
        }
    }

    function setLocalStorage(key, value) {
        try {
            localStorage.setItem(CONFIG.storage.prefix + key, JSON.stringify(value));
        } catch (e) {
            // Storage full or blocked
        }
    }

    function getScrollDepth() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollHeight = document.documentElement.scrollHeight;
        const clientHeight = document.documentElement.clientHeight;
        
        if (scrollHeight <= clientHeight) return 100;
        return Math.round((scrollTop / (scrollHeight - clientHeight)) * 100);
    }

    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

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

    // ================================
    // Initialization
    // ================================

    function init() {
        // Initialize session
        initSession();

        // Collect user profile
        collectUserProfile();

        // Set up event listeners
        setupEventListeners();

        // Start timers
        startTimers();

        // Track initial page view
        trackPageView();

        // Handle page unload
        setupUnloadHandler();
    }

    function initSession() {
        // Check for existing session
        const existingSession = getLocalStorage('session');
        
        if (existingSession && Date.now() - existingSession.lastActivity < CONFIG.storage.sessionExpiry) {
            sessionData.sessionId = existingSession.sessionId;
            sessionData.pageViews = existingSession.pageViews;
            sessionData.interactions = existingSession.interactions;
        } else {
            sessionData.sessionId = generateId();
        }

        // Get anonymous ID from cookie
        sessionData.anonymousId = getCookie('ml_anon_id');
        if (!sessionData.anonymousId) {
            sessionData.anonymousId = generateId();
            setCookie('ml_anon_id', sessionData.anonymousId);
        }

        // Check for user ID
        const userIdMeta = document.querySelector('meta[name="user-id"]');
        if (userIdMeta) {
            sessionData.userId = userIdMeta.content;
        }

        // Update session storage
        updateSessionStorage();
    }

    function updateSessionStorage() {
        setLocalStorage('session', {
            sessionId: sessionData.sessionId,
            pageViews: sessionData.pageViews,
            interactions: sessionData.interactions,
            lastActivity: Date.now(),
        });
    }

    function collectUserProfile() {
        // Device info
        userProfile.device = {
            type: getDeviceType(),
            touch: 'ontouchstart' in window,
            memory: navigator.deviceMemory || null,
            cores: navigator.hardwareConcurrency || null,
        };

        // Browser info
        userProfile.browser = {
            name: getBrowserName(),
            version: getBrowserVersion(),
            userAgent: navigator.userAgent,
            language: navigator.language,
            languages: navigator.languages,
            cookiesEnabled: navigator.cookieEnabled,
            doNotTrack: navigator.doNotTrack === '1',
        };

        // OS info
        userProfile.os = getOS();

        // Screen info
        userProfile.screen = {
            width: window.screen.width,
            height: window.screen.height,
            colorDepth: window.screen.colorDepth,
            pixelRatio: window.devicePixelRatio || 1,
            orientation: window.screen.orientation?.type || 'unknown',
        };

        // Viewport
        userProfile.viewport = {
            width: window.innerWidth,
            height: window.innerHeight,
        };

        // Timezone
        userProfile.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

        // Referrer
        userProfile.referrer = {
            url: document.referrer || null,
            domain: document.referrer ? new URL(document.referrer).hostname : null,
            isInternal: document.referrer && document.referrer.includes(window.location.hostname),
        };

        // Theme detection
        userProfile.theme = {
            prefers: window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
            current: document.documentElement.getAttribute('data-theme') || 
                     document.body.getAttribute('data-theme') ||
                     getCookie('theme') || 'auto',
        };

        // Connection info
        if (navigator.connection) {
            userProfile.connection = {
                type: navigator.connection.effectiveType,
                downlink: navigator.connection.downlink,
                rtt: navigator.connection.rtt,
                saveData: navigator.connection.saveData,
            };
        }
    }

    function getDeviceType() {
        const ua = navigator.userAgent;
        if (/(tablet|ipad|playbook|silk)|(android(?!.*mobi))/i.test(ua)) return 'tablet';
        if (/Mobile|iP(hone|od)|Android|BlackBerry|IEMobile|Kindle|Silk-Accelerated|(hpw|web)OS|Opera M(obi|ini)/.test(ua)) return 'mobile';
        return 'desktop';
    }

    function getBrowserName() {
        const ua = navigator.userAgent;
        if (ua.includes('Firefox')) return 'Firefox';
        if (ua.includes('Chrome') && !ua.includes('Edg')) return 'Chrome';
        if (ua.includes('Safari') && !ua.includes('Chrome')) return 'Safari';
        if (ua.includes('Edg')) return 'Edge';
        if (ua.includes('Opera') || ua.includes('OPR')) return 'Opera';
        if (ua.includes('MSIE') || ua.includes('Trident')) return 'IE';
        return 'Unknown';
    }

    function getBrowserVersion() {
        const ua = navigator.userAgent;
        const matches = ua.match(/(Firefox|Chrome|Safari|Edg|Opera|OPR|MSIE|rv:?)[\s/:](\d+)/);
        return matches ? matches[2] : 'Unknown';
    }

    function getOS() {
        const ua = navigator.userAgent;
        if (ua.includes('Windows')) return { name: 'Windows', version: getWindowsVersion(ua) };
        if (ua.includes('Mac OS X')) return { name: 'macOS', version: getMacVersion(ua) };
        if (ua.includes('Linux')) return { name: 'Linux', version: null };
        if (ua.includes('Android')) return { name: 'Android', version: getAndroidVersion(ua) };
        if (ua.includes('iPhone') || ua.includes('iPad')) return { name: 'iOS', version: getiOSVersion(ua) };
        return { name: 'Unknown', version: null };
    }

    function getWindowsVersion(ua) {
        const matches = ua.match(/Windows NT (\d+\.\d+)/);
        if (!matches) return null;
        const versions = { '10.0': '10', '6.3': '8.1', '6.2': '8', '6.1': '7' };
        return versions[matches[1]] || matches[1];
    }

    function getMacVersion(ua) {
        const matches = ua.match(/Mac OS X (\d+[._]\d+)/);
        return matches ? matches[1].replace('_', '.') : null;
    }

    function getAndroidVersion(ua) {
        const matches = ua.match(/Android (\d+(\.\d+)?)/);
        return matches ? matches[1] : null;
    }

    function getiOSVersion(ua) {
        const matches = ua.match(/OS (\d+[._]\d+)/);
        return matches ? matches[1].replace('_', '.') : null;
    }

    // ================================
    // Event Tracking
    // ================================

    function queueEvent(eventType, data = {}) {
        const event = {
            event_type: eventType,
            timestamp: new Date().toISOString(),
            session_id: sessionData.sessionId,
            user_id: sessionData.userId,
            anonymous_id: sessionData.anonymousId,
            page_url: window.location.href,
            page_path: window.location.pathname,
            user_profile: userProfile,
            page_context: {
                time_on_page: Date.now() - sessionData.currentPage.startTime,
                active_time: activeTimeTracker.totalActiveTime,
                scroll_depth: sessionData.currentPage.scrollDepth,
                max_scroll_depth: sessionData.maxScrollDepth,
                clicks: sessionData.currentPage.clicks,
            },
            ...data,
        };

        eventQueue.push(event);
        sessionData.lastActivity = Date.now();
        updateSessionStorage();

        // Flush if batch size reached
        if (eventQueue.length >= CONFIG.batchSize) {
            flushEvents();
        }
    }

    function flushEvents() {
        if (eventQueue.length === 0) return;

        const events = [...eventQueue];
        eventQueue = [];

        // Send events to server
        sendEvents(events);
    }

    function sendEvents(events) {
        const payload = {
            events: events,
            meta: {
                batch_id: generateId(),
                sent_at: new Date().toISOString(),
                page_url: window.location.href,
            },
        };

        // Use sendBeacon for unload events
        if (navigator.sendBeacon) {
            navigator.sendBeacon(CONFIG.apiUrl, JSON.stringify(payload));
        } else {
            fetch(CONFIG.apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                keepalive: true,
            }).catch(() => {
                // Store failed events for retry
                const failed = getLocalStorage('failed_events', []);
                failed.push(...events);
                setLocalStorage('failed_events', failed.slice(-100));
            });
        }
    }

    // ================================
    // Event Handlers
    // ================================

    function setupEventListeners() {
        // Scroll tracking
        window.addEventListener('scroll', throttle(handleScroll, 100), { passive: true });

        // Click tracking
        document.addEventListener('click', handleClick, { capture: true });

        // Visibility change
        document.addEventListener('visibilitychange', handleVisibilityChange);

        // Activity tracking
        const activityEvents = ['mousemove', 'keydown', 'touchstart', 'scroll'];
        activityEvents.forEach(event => {
            document.addEventListener(event, throttle(handleActivity, 1000), { passive: true });
        });

        // Resize tracking
        window.addEventListener('resize', debounce(handleResize, 500));

        // Form interactions
        document.addEventListener('submit', handleFormSubmit);
        document.addEventListener('change', handleFormChange);

        // Custom events from the application
        window.addEventListener('bunoraa:product_view', handleProductView);
        window.addEventListener('bunoraa:add_to_cart', handleAddToCart);
        window.addEventListener('bunoraa:remove_from_cart', handleRemoveFromCart);
        window.addEventListener('bunoraa:add_to_wishlist', handleAddToWishlist);
        window.addEventListener('bunoraa:search', handleSearch);
        window.addEventListener('bunoraa:checkout', handleCheckout);
        window.addEventListener('bunoraa:purchase', handlePurchase);

        // Navigation tracking (for SPAs)
        if (window.history && window.history.pushState) {
            const originalPushState = window.history.pushState;
            window.history.pushState = function(...args) {
                originalPushState.apply(this, args);
                handleNavigation();
            };
            window.addEventListener('popstate', handleNavigation);
        }
    }

    function handleScroll() {
        const depth = getScrollDepth();
        sessionData.currentPage.scrollDepth = depth;
        
        if (depth > sessionData.maxScrollDepth) {
            sessionData.maxScrollDepth = depth;
        }
        
        sessionData.totalScrollDepth = Math.max(sessionData.totalScrollDepth, depth);
    }

    function handleClick(event) {
        sessionData.currentPage.clicks++;
        sessionData.interactions++;

        const target = event.target.closest('a, button, [data-track]');
        if (!target) return;

        const trackData = {
            element_type: target.tagName.toLowerCase(),
            element_id: target.id || null,
            element_class: target.className || null,
            element_text: target.innerText?.substring(0, 100) || null,
            href: target.href || null,
            position: {
                x: event.clientX,
                y: event.clientY,
            },
        };

        // Track product clicks
        const productElement = target.closest('[data-product-id]');
        if (productElement) {
            trackData.product_id = productElement.dataset.productId;
            trackData.product_name = productElement.dataset.productName;
            trackData.product_position = productElement.dataset.position;
            
            queueEvent('product_click', trackData);
        } else {
            queueEvent('click', trackData);
        }
    }

    function handleVisibilityChange() {
        if (document.hidden) {
            // Page is hidden - track idle start
            activeTimeTracker.isActive = false;
            activeTimeTracker.idleStart = Date.now();
            
            queueEvent('page_hidden', {
                active_time: activeTimeTracker.totalActiveTime,
            });
        } else {
            // Page is visible again
            if (activeTimeTracker.idleStart) {
                const idleTime = Date.now() - activeTimeTracker.idleStart;
                activeTimeTracker.idleStart = null;
            }
            activeTimeTracker.isActive = true;
            activeTimeTracker.lastActiveTime = Date.now();
            
            queueEvent('page_visible', {});
        }
    }

    function handleActivity() {
        const now = Date.now();
        
        if (activeTimeTracker.isActive) {
            const elapsed = now - activeTimeTracker.lastActiveTime;
            
            // Only count if not idle (less than idle timeout since last activity)
            if (elapsed < CONFIG.idleTimeout) {
                activeTimeTracker.totalActiveTime += elapsed;
            }
        }
        
        activeTimeTracker.isActive = true;
        activeTimeTracker.lastActiveTime = now;
        sessionData.currentPage.isIdle = false;
    }

    function handleResize() {
        userProfile.viewport = {
            width: window.innerWidth,
            height: window.innerHeight,
        };
    }

    function handleFormSubmit(event) {
        const form = event.target;
        
        queueEvent('form_submit', {
            form_id: form.id || null,
            form_action: form.action || null,
            form_method: form.method || null,
        });
    }

    function handleFormChange(event) {
        const input = event.target;
        
        // Don't track sensitive fields
        const sensitiveTypes = ['password', 'email', 'tel', 'credit-card'];
        if (sensitiveTypes.includes(input.type) || input.autocomplete?.includes('password')) {
            return;
        }

        queueEvent('form_interaction', {
            input_type: input.type,
            input_name: input.name || null,
            input_id: input.id || null,
        });
    }

    function handleNavigation() {
        // Track page exit
        queueEvent('page_exit', {
            time_on_page: Date.now() - sessionData.currentPage.startTime,
            active_time: activeTimeTracker.totalActiveTime,
            scroll_depth: sessionData.currentPage.scrollDepth,
            clicks: sessionData.currentPage.clicks,
        });

        // Reset page tracking
        sessionData.currentPage = {
            url: window.location.href,
            startTime: Date.now(),
            scrollDepth: 0,
            activeTime: 0,
            clicks: 0,
            isIdle: false,
        };
        activeTimeTracker.totalActiveTime = 0;

        // Track new page view
        trackPageView();
    }

    // ================================
    // Custom Event Handlers
    // ================================

    function handleProductView(event) {
        const data = event.detail || {};
        
        queueEvent('product_view', {
            product_id: data.product_id,
            product_name: data.product_name,
            product_category: data.category,
            product_price: data.price,
            product_discount: data.discount,
            product_variant: data.variant,
            product_color: data.color,
            product_material: data.material,
            product_available: data.available,
            product_is_new: data.is_new_arrival,
            product_is_bestseller: data.is_bestseller,
            product_rating: data.rating,
            product_reviews: data.review_count,
            source_page: data.source_page,
            position: data.position,
        });
    }

    function handleAddToCart(event) {
        const data = event.detail || {};
        
        queueEvent('add_to_cart', {
            product_id: data.product_id,
            product_name: data.product_name,
            product_price: data.price,
            quantity: data.quantity || 1,
            variant: data.variant,
            color: data.color,
            size: data.size,
            cart_value: data.cart_value,
            source: data.source,
        });
    }

    function handleRemoveFromCart(event) {
        const data = event.detail || {};
        
        queueEvent('remove_from_cart', {
            product_id: data.product_id,
            quantity: data.quantity || 1,
            reason: data.reason,
        });
    }

    function handleAddToWishlist(event) {
        const data = event.detail || {};
        
        queueEvent('add_to_wishlist', {
            product_id: data.product_id,
            product_name: data.product_name,
            source: data.source,
        });
    }

    function handleSearch(event) {
        const data = event.detail || {};
        
        queueEvent('search', {
            query: data.query,
            results_count: data.results_count,
            filters: data.filters,
            sort_by: data.sort_by,
            page: data.page,
        });
    }

    function handleCheckout(event) {
        const data = event.detail || {};
        
        queueEvent('checkout', {
            step: data.step,
            step_name: data.step_name,
            cart_value: data.cart_value,
            items_count: data.items_count,
            shipping_method: data.shipping_method,
            payment_method: data.payment_method,
            coupon: data.coupon,
        });
    }

    function handlePurchase(event) {
        const data = event.detail || {};
        
        queueEvent('purchase', {
            order_id: data.order_id,
            order_value: data.order_value,
            items_count: data.items_count,
            shipping: data.shipping,
            tax: data.tax,
            discount: data.discount,
            coupon: data.coupon,
            payment_method: data.payment_method,
        });
    }

    // ================================
    // Page View Tracking
    // ================================

    function trackPageView() {
        sessionData.pageViews++;

        // Get page-specific data
        const pageData = collectPageData();

        queueEvent('page_view', {
            page_title: document.title,
            page_type: pageData.type,
            page_category: pageData.category,
            page_data: pageData.data,
            referrer: document.referrer || null,
            utm: getUTMParams(),
        });
    }

    function collectPageData() {
        const path = window.location.pathname;
        const data = {
            type: 'other',
            category: null,
            data: {},
        };

        // Detect page type from URL patterns
        if (path === '/' || path === '/home') {
            data.type = 'home';
        } else if (path.includes('/product/') || path.includes('/p/')) {
            data.type = 'product';
            data.data = getProductPageData();
        } else if (path.includes('/category/') || path.includes('/c/')) {
            data.type = 'category';
            data.category = getCategoryFromMeta();
        } else if (path.includes('/cart')) {
            data.type = 'cart';
        } else if (path.includes('/checkout')) {
            data.type = 'checkout';
        } else if (path.includes('/search')) {
            data.type = 'search';
            data.data = { query: new URLSearchParams(window.location.search).get('q') };
        } else if (path.includes('/account') || path.includes('/profile')) {
            data.type = 'account';
        } else if (path.includes('/wishlist')) {
            data.type = 'wishlist';
        } else if (path.includes('/order')) {
            data.type = 'order';
        }

        return data;
    }

    function getProductPageData() {
        // Try to get product data from structured data
        const ldJson = document.querySelector('script[type="application/ld+json"]');
        if (ldJson) {
            try {
                const structured = JSON.parse(ldJson.textContent);
                if (structured['@type'] === 'Product') {
                    return {
                        product_id: structured.sku || null,
                        product_name: structured.name || null,
                        product_price: structured.offers?.price || null,
                        product_currency: structured.offers?.priceCurrency || null,
                        product_available: structured.offers?.availability?.includes('InStock') || null,
                        product_rating: structured.aggregateRating?.ratingValue || null,
                        product_reviews: structured.aggregateRating?.reviewCount || null,
                    };
                }
            } catch (e) {}
        }

        // Try to get from meta tags
        return {
            product_id: document.querySelector('meta[property="product:id"]')?.content || 
                       document.querySelector('[data-product-id]')?.dataset?.productId,
            product_name: document.querySelector('meta[property="og:title"]')?.content,
            product_price: document.querySelector('meta[property="product:price:amount"]')?.content,
        };
    }

    function getCategoryFromMeta() {
        return document.querySelector('meta[name="category"]')?.content ||
               document.querySelector('[data-category]')?.dataset?.category;
    }

    function getUTMParams() {
        const params = new URLSearchParams(window.location.search);
        return {
            source: params.get('utm_source'),
            medium: params.get('utm_medium'),
            campaign: params.get('utm_campaign'),
            term: params.get('utm_term'),
            content: params.get('utm_content'),
        };
    }

    // ================================
    // Timers and Unload
    // ================================

    function startTimers() {
        // Flush timer
        flushTimer = setInterval(flushEvents, CONFIG.flushInterval);

        // Heartbeat timer
        heartbeatTimer = setInterval(() => {
            queueEvent('heartbeat', {
                time_on_page: Date.now() - sessionData.currentPage.startTime,
                active_time: activeTimeTracker.totalActiveTime,
                scroll_depth: sessionData.currentPage.scrollDepth,
                is_idle: sessionData.currentPage.isIdle,
            });
        }, CONFIG.heartbeatInterval);

        // Idle detection
        setInterval(() => {
            const timeSinceActivity = Date.now() - activeTimeTracker.lastActiveTime;
            
            if (timeSinceActivity > CONFIG.idleTimeout && activeTimeTracker.isActive) {
                activeTimeTracker.isActive = false;
                sessionData.currentPage.isIdle = true;
                
                queueEvent('user_idle', {
                    idle_duration: timeSinceActivity,
                    time_on_page: Date.now() - sessionData.currentPage.startTime,
                });
            }
        }, 10000);
    }

    function setupUnloadHandler() {
        const handleUnload = () => {
            // Final page exit event
            queueEvent('page_exit', {
                time_on_page: Date.now() - sessionData.currentPage.startTime,
                active_time: activeTimeTracker.totalActiveTime,
                scroll_depth: sessionData.currentPage.scrollDepth,
                max_scroll_depth: sessionData.maxScrollDepth,
                clicks: sessionData.currentPage.clicks,
                is_bounce: sessionData.pageViews === 1 && sessionData.interactions < 2,
            });

            // Flush remaining events
            flushEvents();
        };

        window.addEventListener('beforeunload', handleUnload);
        window.addEventListener('pagehide', handleUnload);
    }

    // ================================
    // Public API
    // ================================

    window.BunoraaML = {
        // Track custom event
        track: function(eventType, data) {
            queueEvent(eventType, data);
        },

        // Track product view
        trackProductView: function(productData) {
            window.dispatchEvent(new CustomEvent('bunoraa:product_view', { detail: productData }));
        },

        // Track add to cart
        trackAddToCart: function(cartData) {
            window.dispatchEvent(new CustomEvent('bunoraa:add_to_cart', { detail: cartData }));
        },

        // Track remove from cart
        trackRemoveFromCart: function(cartData) {
            window.dispatchEvent(new CustomEvent('bunoraa:remove_from_cart', { detail: cartData }));
        },

        // Track wishlist
        trackWishlist: function(wishlistData) {
            window.dispatchEvent(new CustomEvent('bunoraa:add_to_wishlist', { detail: wishlistData }));
        },

        // Track search
        trackSearch: function(searchData) {
            window.dispatchEvent(new CustomEvent('bunoraa:search', { detail: searchData }));
        },

        // Track checkout step
        trackCheckout: function(checkoutData) {
            window.dispatchEvent(new CustomEvent('bunoraa:checkout', { detail: checkoutData }));
        },

        // Track purchase
        trackPurchase: function(purchaseData) {
            window.dispatchEvent(new CustomEvent('bunoraa:purchase', { detail: purchaseData }));
        },

        // Set user ID
        setUserId: function(userId) {
            sessionData.userId = userId;
            queueEvent('user_identified', { user_id: userId });
        },

        // Get session ID
        getSessionId: function() {
            return sessionData.sessionId;
        },

        // Flush events immediately
        flush: function() {
            flushEvents();
        },

        // Debug mode
        debug: function() {
            console.log('Session Data:', sessionData);
            console.log('User Profile:', userProfile);
            console.log('Event Queue:', eventQueue);
            console.log('Active Time:', activeTimeTracker);
        },
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})(window, document);
