/**
 * Categories API Module
 * @module api/categories
 */

const CategoriesApi = (function() {
    'use strict';

    async function getCategories(params = {}) {
        return ApiClient.get('/catalog/categories/', {
            page: params.page || 1,
            page_size: params.pageSize || 100,
            parent_id: params.parentId || undefined
        }, { useCache: true, cacheTTL: 300000 });
    }

    async function getCategory(idOrSlug) {
        return ApiClient.get(`/catalog/categories/${idOrSlug}/`, {}, { useCache: true, cacheTTL: 300000 });
    }

    async function getTree() {
        return ApiClient.get('/catalog/categories/tree/', {}, { useCache: true, cacheTTL: 300000 });
    }

    async function getFeatured(limit = 6) {
        // Note: Categories don't have is_featured field, so we return root categories
        // which are the primary navigation categories
        return ApiClient.get('/catalog/categories/', {
            parent_id: 'null',
            page_size: limit
        }, { useCache: true, cacheTTL: 300000 });
    }

    async function getRootCategories() {
        return ApiClient.get('/catalog/categories/', {
            parent_id: 'null',
            is_active: true
        }, { useCache: true, cacheTTL: 300000 });
    }

    async function getSubcategories(categoryId) {
        return ApiClient.get('/catalog/categories/', {
            parent_id: categoryId,
            is_active: true
        }, { useCache: true, cacheTTL: 300000 });
    }

    async function getCategoryProducts(categoryId, params = {}) {
        return ApiClient.get(`/catalog/categories/${categoryId}/products/`, {
            page: params.page || 1,
            page_size: params.pageSize || 20,
            ordering: params.ordering || '-created_at'
        });
    }

    async function getBreadcrumb(categoryId) {
        return ApiClient.get(`/catalog/categories/${categoryId}/breadcrumb/`, {}, { useCache: true });
    }

    function flattenTree(categories, parent = null, level = 0) {
        const result = [];
        
        for (const cat of categories) {
            result.push({
                ...cat,
                parent,
                level
            });
            
            if (cat.children && cat.children.length > 0) {
                result.push(...flattenTree(cat.children, cat, level + 1));
            }
        }
        
        return result;
    }

    return {
        getCategories,
        getCategory,
        getTree,
        getFeatured,
        getRootCategories,
        getSubcategories,
        getCategoryProducts,
        getBreadcrumb,
        flattenTree
    };
})();

window.CategoriesApi = CategoriesApi;
