/**
 * Products API Module
 * @module api/products
 */

const ProductsApi = (function() {
    'use strict';

    async function getProducts(params = {}) {
        const currencyCode = (typeof window !== 'undefined' && window.BUNORAA_CURRENCY && window.BUNORAA_CURRENCY.code) ? window.BUNORAA_CURRENCY.code : undefined;
        const queryParams = {
            page: params.page || 1,
            page_size: params.pageSize || 20,
            ordering: params.ordering || '-created_at',
            search: params.search || undefined,
            category: params.category || undefined,
            tag: params.tag || undefined,
            min_price: params.minPrice || undefined,
            max_price: params.maxPrice || undefined,
            in_stock: params.inStock ? 'true' : undefined,
            is_featured: params.featured ? 'true' : undefined,
            is_on_sale: params.onSale ? 'true' : undefined,
            bestseller: params.bestseller ? 'true' : undefined,
            currency: params.currency || currencyCode
        };

        return ApiClient.get('/catalog/products/', queryParams, { useCache: true, cacheTTL: 60000 });
    }

    async function getProduct(idOrSlug) {
        const currencyCode = (typeof window !== 'undefined' && window.BUNORAA_CURRENCY && window.BUNORAA_CURRENCY.code) ? window.BUNORAA_CURRENCY.code : undefined;
        return ApiClient.get(`/catalog/products/${idOrSlug}/`, { currency: currencyCode }, { useCache: true, cacheTTL: 120000 });
    }

    async function getFeatured(limit = 8) {
        return ApiClient.get('/catalog/products/', { is_featured: 'true', page_size: limit }, { useCache: true });
    }

    async function getNewArrivals(limit = 10) {
        return ApiClient.get('/catalog/products/', { ordering: '-created_at', page_size: limit }, { useCache: true });
    }

    async function getBestSellers(limit = 10) {
        return ApiClient.get('/catalog/products/', { ordering: '-sold_count', page_size: limit }, { useCache: true });
    }

    async function getOnSale(limit = 10) {
        return ApiClient.get('/catalog/products/', { is_on_sale: 'true', page_size: limit }, { useCache: true });
    }

    async function getRelated(productId, limit = 6) {
        return ApiClient.get(`/catalog/products/${productId}/related/`, { limit }, { useCache: true });
    }

    async function search(query, params = {}) {
        return ApiClient.get('/catalog/products/', {
            search: query,
            page: params.page || 1,
            page_size: params.pageSize || 20,
            ordering: params.ordering || '-created_at',
            category: params.category || undefined,
            min_price: params.minPrice || undefined,
            max_price: params.maxPrice || undefined
        });
    }

    async function getReviews(productId, params = {}) {
        return ApiClient.get('/reviews/', {
            product: productId,
            page: params.page || 1,
            page_size: params.pageSize || 10,
            ordering: params.ordering || '-created_at'
        }, { useCache: true });
    }

    async function submitReview(productId, data) {
        return ApiClient.post('/reviews/', {
            product: productId,
            rating: data.rating,
            title: data.title,
            content: data.content
        }, { requiresAuth: true });
    }

    async function getTags() {
        return ApiClient.get('/catalog/tags/', {}, { useCache: true, cacheTTL: 300000 });
    }

    async function getAttributes() {
        return ApiClient.get('/catalog/products/attributes/', {}, { useCache: true, cacheTTL: 300000 });
    }

    // Backwards-compatible alias
    async function getBySlug(slugOrId) {
        return getProduct(slugOrId);
    }

    // Recommendations: frequently_bought_together | similar | you_may_also_like | all
    async function getRecommendations(productId, type = 'all', limit = 6) {
        return ApiClient.get('/recommendations/', { product_id: productId, type, limit }, { useCache: true, cacheTTL: 120000 });
    }

    return {
        getProducts,
        getProduct,
        getBySlug,
        getFeatured,
        getNewArrivals,
        getBestSellers,
        getOnSale,
        getRelated,
        getRecommendations,
        search,
        getReviews,
        submitReview,
        getTags,
        getAttributes
    };
})();

window.ProductsApi = ProductsApi;
