/**
 * Orders API Module
 * @module api/orders
 */

const OrdersApi = (function() {
    'use strict';

    const ORDERS_PATH = '/orders/orders/';

    async function getOrders(params = {}) {
        return ApiClient.get(ORDERS_PATH, {
            page: params.page || 1,
            page_size: params.pageSize || 10,
            status: params.status || undefined,
            ordering: params.ordering || '-created_at'
        }, { requiresAuth: true });
    }

    async function getOrder(orderId) {
        return ApiClient.get(`${ORDERS_PATH}${orderId}/`, {}, { requiresAuth: true });
    }

    async function getOrderByNumber(orderNumber, email = null) {
        const params = { order_number: orderNumber };
        if (email) params.email = email;
        return ApiClient.get(`${ORDERS_PATH}by-number/`, params);
    }

    async function trackOrder(orderNumber, email = null) {
        const params = { order_number: orderNumber };
        if (email) params.email = email;
        return ApiClient.get(`${ORDERS_PATH}track/`, params);
    }

    async function cancelOrder(orderId, reason = '') {
        return ApiClient.post(`${ORDERS_PATH}${orderId}/cancel/`, { reason }, { requiresAuth: true });
    }

    async function requestReturn(orderId, items, reason) {
        return ApiClient.post(`${ORDERS_PATH}${orderId}/return/`, {
            items,
            reason
        }, { requiresAuth: true });
    }

    async function getInvoice(orderId) {
        return ApiClient.get(`${ORDERS_PATH}${orderId}/invoice/`, {}, { requiresAuth: true });
    }

    async function reorder(orderId) {
        const response = await ApiClient.post(`${ORDERS_PATH}${orderId}/reorder/`, {}, { requiresAuth: true });
        
        if (response.success) {
            window.dispatchEvent(new CustomEvent('cart:updated'));
        }
        
        return response;
    }

    return {
        getOrders,
        getOrder,
        getOrderByNumber,
        trackOrder,
        cancelOrder,
        requestReturn,
        getInvoice,
        reorder
    };
})();

window.OrdersApi = OrdersApi;
