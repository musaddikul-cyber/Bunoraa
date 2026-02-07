/**
 * Cart API Module
 * @module api/cart
 */

const CartApi = (function() {
    'use strict';

    const CART_PATH = '/commerce/cart/';
    const PROMOTIONS_PATH = '/promotions/';

    async function getCart() {
        const response = await ApiClient.get(CART_PATH);
        if (response.success) {
            updateBadge(response.data);
            window.dispatchEvent(new CustomEvent('cart:updated', { detail: response.data }));
        }
        return response;
    }

    async function addItem(productId, quantity = 1, variantId = null) {
        const data = { product_id: productId, quantity };
        if (variantId) data.variant_id = variantId;

        // Optimistic UI: mark pending add so UI can react quickly
        try {
            const pendingKey = 'cart:pending_add';
            localStorage.setItem(pendingKey, JSON.stringify({ product_id: productId, quantity, variant_id: variantId, ts: Date.now() }));
            window.dispatchEvent(new CustomEvent('cart:pending', { detail: { product_id: productId, quantity } }));
        } catch (e) {}

        const response = await ApiClient.post(`${CART_PATH}add/`, data);
        
        // Clear pending marker
        try { localStorage.removeItem('cart:pending_add'); } catch (e) {}

        if (response.success) {
            updateBadge(response.data?.cart);
            window.dispatchEvent(new CustomEvent('cart:item-added', { detail: response.data }));
            window.dispatchEvent(new CustomEvent('cart:updated', { detail: response.data?.cart }));
        }
        
        return response;
    }

    async function updateItem(itemId, quantity) {
        // Use POST on the update endpoint: POST /api/v1/commerce/cart/update/{itemId}/
        const response = await ApiClient.post(`${CART_PATH}update/${itemId}/`, { quantity });
        
        if (response.success) {
            updateBadge(response.data?.cart);
            window.dispatchEvent(new CustomEvent('cart:item-updated', { detail: response.data }));
            window.dispatchEvent(new CustomEvent('cart:updated', { detail: response.data?.cart }));
        }
        
        return response;
    }

    async function removeItem(itemId) {
        // Use POST on the remove endpoint: POST /api/v1/commerce/cart/remove/{itemId}/
        const response = await ApiClient.post(`${CART_PATH}remove/${itemId}/`);
        
        if (response.success) {
            updateBadge(response.data?.cart);
            window.dispatchEvent(new CustomEvent('cart:item-removed', { detail: { itemId, cart: response.data?.cart } }));
            window.dispatchEvent(new CustomEvent('cart:updated', { detail: response.data?.cart }));
        }
        
        return response;
    }

    async function clearCart() {
        // Use POST on the clear endpoint: POST /api/v1/commerce/cart/clear/
        const response = await ApiClient.post(`${CART_PATH}clear/`);
        
        if (response.success) {
            updateBadge({ item_count: 0 });
            window.dispatchEvent(new CustomEvent('cart:cleared'));
            window.dispatchEvent(new CustomEvent('cart:updated', { detail: response.data?.cart }));
        }
        
        return response;
    }

    async function applyCoupon(code, options = {}) {
        const subtotal = options.subtotal;
        const payload = { code };
        if (subtotal !== undefined && subtotal !== null) {
            payload.subtotal = subtotal;
        }

        const response = await ApiClient.post('/promotions/coupons/apply/', payload);

        if (response.success) {
            window.dispatchEvent(new CustomEvent('cart:coupon-applied', { detail: response.data }));
            window.dispatchEvent(new CustomEvent('cart:updated', { detail: response.data?.cart }));
        }

        return response;
    }

    async function updateGiftOptions(options = {}) {
        const response = await ApiClient.post(`${CART_PATH}gift/`, options);

        if (response.success) {
            window.dispatchEvent(new CustomEvent('cart:gift-updated', { detail: response.data }));
        }

        return response;
    }

    async function removeCoupon() {
        // Use POST on the remove_coupon endpoint: POST /api/v1/commerce/cart/remove_coupon/
        const response = await ApiClient.post(`${CART_PATH}remove_coupon/`);
        
        if (response.success) {
            window.dispatchEvent(new CustomEvent('cart:coupon-removed'));
            window.dispatchEvent(new CustomEvent('cart:updated', { detail: response.data?.cart }));
        }
        
        return response;
    }

    async function validate() {
        return ApiClient.get(`${CART_PATH}summary/`);
    }

    async function validateCart() {
        return ApiClient.post(`${CART_PATH}validate/`);
    }

    async function lockPrices(durationHours = null) {
        const payload = {};
        if (durationHours) payload.duration_hours = durationHours;
        return ApiClient.post(`${CART_PATH}lock-prices/`, payload);
    }

    async function shareCart(options = {}) {
        return ApiClient.post(`${CART_PATH}share/`, options);
    }

    async function merge() {
        return ApiClient.post(`${CART_PATH}`, {}, { requiresAuth: true });
    }

    function updateBadge(cart) {
        const count = cart?.item_count || cart?.items?.length || 0;
        const badges = document.querySelectorAll('[data-cart-count]');
        
        badges.forEach(badge => {
            // Cap visible badge to a single digit display to reduce layout width on small viewports
            badge.textContent = count > 9 ? '9+' : count;
            badge.classList.toggle('hidden', count === 0);
        });
    }

    return {
        getCart,
        addItem,
        updateItem,
        removeItem,
        clearCart,
        applyCoupon,
        removeCoupon,
        updateGiftOptions,
        validate,
        validateCart,
        lockPrices,
        shareCart,
        merge,
        updateBadge
    };
})();

window.CartApi = CartApi;
