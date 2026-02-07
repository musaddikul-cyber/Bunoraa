/**
 * Checkout API Module
 * @module api/checkout
 */

const CheckoutApi = (function() {
    'use strict';

    const CHECKOUT_PATH = '/commerce/checkout/';

    async function getSession() {
        return ApiClient.get(CHECKOUT_PATH);
    }

    async function startCheckout() {
        return ApiClient.post(`${CHECKOUT_PATH}start/`);
    }

    async function updateShipping(data) {
        return ApiClient.post(`${CHECKOUT_PATH}shipping/`, data);
    }

    async function getShippingOptions() {
        return ApiClient.get(`${CHECKOUT_PATH}shipping-options/`);
    }

    async function setShippingMethod(method) {
        return ApiClient.post(`${CHECKOUT_PATH}shipping-method/`, { method });
    }

    async function setPaymentMethod(method, data = {}) {
        return ApiClient.post(`${CHECKOUT_PATH}payment-method/`, {
            payment_method: method,
            ...data
        });
    }

    async function createPaymentIntent() {
        return ApiClient.post(`${CHECKOUT_PATH}payment-intent/`);
    }

    async function getSummary() {
        return ApiClient.get(`${CHECKOUT_PATH}summary/`);
    }

    async function complete(paymentData = {}) {
        const response = await ApiClient.post(`${CHECKOUT_PATH}complete/`, paymentData);
        
        if (response.success) {
            ApiClient.clearCache('/commerce/cart/');
            window.dispatchEvent(new CustomEvent('checkout:completed', { detail: response.data }));
            window.dispatchEvent(new CustomEvent('cart:cleared'));
        }
        
        return response;
    }

    async function validateStep(step, data) {
        return ApiClient.post(`${CHECKOUT_PATH}validate/${step}/`, data);
    }

    return {
        getSession,
        startCheckout,
        updateShipping,
        getShippingOptions,
        setShippingMethod,
        setPaymentMethod,
        createPaymentIntent,
        getSummary,
        complete,
        validateStep
    };
})();

window.CheckoutApi = CheckoutApi;
