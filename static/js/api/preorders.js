/**
 * Pre-orders API Module
 * @module api/preorders
 */

const PreordersApi = (function() {
    'use strict';

    /**
     * Get pre-order categories
     */
    async function getCategories(params = {}) {
        return ApiClient.get('/preorders/categories/', {
            page: params.page || 1,
            page_size: params.pageSize || 20,
            featured: params.featured || undefined,
            active: params.active !== false
        }, { useCache: true, cacheTTL: 300000 });
    }

    /**
     * Get single pre-order category
     */
    async function getCategory(slug) {
        return ApiClient.get(`/preorders/categories/${slug}/`, {}, { useCache: true, cacheTTL: 300000 });
    }

    /**
     * Get category options for customization
     */
    async function getCategoryOptions(categoryId) {
        return ApiClient.get(`/preorders/api/category/${categoryId}/options/`, {}, { useCache: true, cacheTTL: 60000 });
    }

    /**
     * Calculate pre-order price
     */
    async function calculatePrice(data) {
        return ApiClient.post('/preorders/api/calculate-price/', data);
    }

    /**
     * Get user's pre-orders
     */
    async function getMyPreorders(params = {}) {
        return ApiClient.get('/preorders/my-orders/', {
            page: params.page || 1,
            page_size: params.pageSize || 10,
            status: params.status || undefined
        });
    }

    /**
     * Get single pre-order details
     */
    async function getPreorder(preorderNumber) {
        return ApiClient.get(`/preorders/order/${preorderNumber}/`);
    }

    /**
     * Get pre-order status
     */
    async function getPreorderStatus(preorderNumber) {
        return ApiClient.get(`/preorders/api/order/${preorderNumber}/status/`);
    }

    /**
     * Send message for a pre-order
     */
    async function sendMessage(preorderNumber, message, attachments = []) {
        const formData = new FormData();
        formData.append('message', message);
        attachments.forEach((file, index) => {
            formData.append(`attachment_${index}`, file);
        });
        return ApiClient.post(`/preorders/order/${preorderNumber}/message/`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
    }

    /**
     * Request revision for a pre-order
     */
    async function requestRevision(preorderNumber, reason) {
        return ApiClient.post(`/preorders/order/${preorderNumber}/revision/`, { reason });
    }

    /**
     * Approve pre-order
     */
    async function approvePreorder(preorderNumber) {
        return ApiClient.post(`/preorders/order/${preorderNumber}/approve/`);
    }

    /**
     * Respond to quote
     */
    async function respondToQuote(preorderNumber, quoteId, action) {
        return ApiClient.post(`/preorders/order/${preorderNumber}/quote/${quoteId}/respond/`, { action });
    }

    /**
     * Mark messages as read
     */
    async function markMessagesRead(preorderNumber) {
        return ApiClient.post(`/preorders/order/${preorderNumber}/mark-read/`);
    }

    /**
     * Use a template for pre-order
     */
    async function useTemplate(templateId) {
        return ApiClient.post(`/preorders/api/template/${templateId}/use/`);
    }

    /**
     * Get pre-order landing page data (featured categories, templates, etc.)
     */
    async function getLandingData() {
        try {
            const [categoriesRes] = await Promise.all([
                getCategories({ featured: true, pageSize: 6 })
            ]);
            
            const categories = categoriesRes?.data?.results || categoriesRes?.data || categoriesRes?.results || [];
            
            return {
                categories,
                stats: {
                    totalOrders: '500+',
                    happyCustomers: '450+',
                    avgRating: '4.9'
                }
            };
        } catch (error) {
            return {
                categories: [],
                stats: {
                    totalOrders: '500+',
                    happyCustomers: '450+',
                    avgRating: '4.9'
                }
            };
        }
    }

    return {
        getCategories,
        getCategory,
        getCategoryOptions,
        calculatePrice,
        getMyPreorders,
        getPreorder,
        getPreorderStatus,
        sendMessage,
        requestRevision,
        approvePreorder,
        respondToQuote,
        markMessagesRead,
        useTemplate,
        getLandingData
    };
})();

window.PreordersApi = PreordersApi;
