/**
 * API Client - Core HTTP Client
 * @module api/client
 */

const ApiClient = (function() {
    'use strict';

    const BASE_URL = '/api/v1';
    const TOKEN_KEY = 'access_token';
    const REFRESH_KEY = 'refresh_token';

    const cache = new Map();
    const pendingRequests = new Map();
    let isRefreshing = false;
    let refreshSubscribers = [];

    function getCsrfToken() {
        const name = 'csrftoken';
        const cookies = document.cookie.split(';');
        for (let cookie of cookies) {
            cookie = cookie.trim();
            if (cookie.startsWith(name + '=')) {
                return decodeURIComponent(cookie.substring(name.length + 1));
            }
        }
        return '';
    }

    function getAccessToken() {
        return localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY);
    }

    function getRefreshToken() {
        return localStorage.getItem(REFRESH_KEY) || sessionStorage.getItem(REFRESH_KEY);
    }

    function setTokens(access, refresh, persistent = true) {
        const storage = persistent ? localStorage : sessionStorage;
        if (access) storage.setItem(TOKEN_KEY, access);
        if (refresh) storage.setItem(REFRESH_KEY, refresh);
    }

    function clearTokens() {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(REFRESH_KEY);
        sessionStorage.removeItem(TOKEN_KEY);
        sessionStorage.removeItem(REFRESH_KEY);
    }

    function isTokenExpired(token) {
        if (!token) return true;
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            return payload.exp * 1000 < Date.now();
        } catch {
            return true;
        }
    }

    function subscribeTokenRefresh(callback) {
        refreshSubscribers.push(callback);
    }

    function onTokenRefreshed(token) {
        refreshSubscribers.forEach(cb => cb(token));
        refreshSubscribers = [];
    }

    async function refreshAccessToken() {
        const refresh = getRefreshToken();
        if (!refresh) {
            clearTokens();
            throw new Error('No refresh token');
        }

        const response = await fetch(`${BASE_URL}/auth/token/refresh/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken()
            },
            body: JSON.stringify({ refresh })
        });

        if (!response.ok) {
            clearTokens();
            throw new Error('Token refresh failed');
        }

        const data = await response.json();
        const persistent = !!localStorage.getItem(REFRESH_KEY);
        setTokens(data.access, data.refresh || refresh, persistent);
        return data.access;
    }

    function buildUrl(endpoint, params = {}) {
        const url = new URL(BASE_URL + endpoint, window.location.origin);
        Object.entries(params).forEach(([key, value]) => {
            if (value !== null && value !== undefined && value !== '') {
                if (Array.isArray(value)) {
                    value.forEach(v => url.searchParams.append(key, v));
                } else {
                    url.searchParams.append(key, String(value));
                }
            }
        });
        return url.toString();
    }

    function getCacheKey(method, url, body) {
        return `${method}:${url}:${body ? JSON.stringify(body) : ''}`;
    }

    function getFromCache(key, ttl = 60000) {
        const cached = cache.get(key);
        if (cached && Date.now() - cached.timestamp < ttl) {
            return cached.data;
        }
        cache.delete(key);
        return null;
    }

    function setCache(key, data) {
        cache.set(key, { data, timestamp: Date.now() });
    }

    function clearCache(pattern = null) {
        if (pattern) {
            for (const key of cache.keys()) {
                if (key.includes(pattern)) cache.delete(key);
            }
        } else {
            cache.clear();
        }
    }

    async function request(method, endpoint, options = {}) {
        const {
            body = null,
            params = {},
            headers = {},
            useCache = false,
            cacheTTL = 60000,
            requiresAuth = false,
            signal = null
        } = options;

        const url = buildUrl(endpoint, params);
        const cacheKey = getCacheKey(method, url, body);

        if (useCache && method === 'GET') {
            const cached = getFromCache(cacheKey, cacheTTL);
            if (cached) return cached;
        }

        if (pendingRequests.has(cacheKey)) {
            return pendingRequests.get(cacheKey);
        }

        // Use server-configured currency code for requests (remove localStorage-based client preference)
        const selectedCurrencyHeader = (window.BUNORAA_CURRENCY && window.BUNORAA_CURRENCY.code) || null;

        const requestHeaders = {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken(),
            'X-Requested-With': 'XMLHttpRequest',
            ...(selectedCurrencyHeader ? { 'X-User-Currency': selectedCurrencyHeader } : {}),
            ...headers
        };

        let token = getAccessToken();
        let usedAuthHeader = false;
        if (token) {
            if (isTokenExpired(token)) {
                if (!isRefreshing) {
                    isRefreshing = true;
                    try {
                        token = await refreshAccessToken();
                        onTokenRefreshed(token);
                    } catch (err) {
                        isRefreshing = false;
                        if (requiresAuth) {
                            window.dispatchEvent(new CustomEvent('auth:required'));
                            throw { status: 401, message: 'Authentication required' };
                        }
                    }
                    isRefreshing = false;
                } else {
                    token = await new Promise(resolve => subscribeTokenRefresh(resolve));
                }
            }
            if (token) {
                requestHeaders['Authorization'] = `Bearer ${token}`;
                usedAuthHeader = true;
            }
        }

        const usingSessionAuth = requiresAuth && !usedAuthHeader;

        const config = {
            method,
            headers: requestHeaders,
            credentials: 'same-origin'
        };

        if (body && method !== 'GET') {
            config.body = JSON.stringify(body);
        }

        if (signal) {
            config.signal = signal;
        }

        const requestPromise = (async () => {
            try {
                const response = await fetch(url, config);

                if (response.status === 204) {
                    return { success: true, message: 'Success', data: null, meta: null };
                }

                let data;
                try {
                    data = await response.json();
                } catch {
                    data = { success: false, message: 'Invalid response format' };
                }

                if (!response.ok) {
                    const error = {
                        status: response.status,
                        message: data.message || data.detail || `HTTP ${response.status}`,
                        errors: data.errors || data.meta?.errors || null,
                        data: data.data || null
                    };

                    if (response.status === 401) {
                        if (usedAuthHeader) {
                            clearTokens();
                            window.dispatchEvent(new CustomEvent('auth:expired'));
                        } else if (usingSessionAuth) {
                            window.__DJANGO_SESSION_AUTH__ = false;
                            window.dispatchEvent(new CustomEvent('auth:required'));
                        }
                    }

                    throw error;
                }

                const result = {
                    success: data.success !== undefined ? data.success : true,
                    message: data.message || 'Success',
                    data: data.data !== undefined ? data.data : data,
                    meta: data.meta || null
                };

                if (useCache && method === 'GET') {
                    setCache(cacheKey, result);
                }

                return result;
            } finally {
                pendingRequests.delete(cacheKey);
            }
        })();

        pendingRequests.set(cacheKey, requestPromise);
        return requestPromise;
    }

    async function upload(endpoint, file, fieldName = 'file', additionalData = {}) {
        const formData = new FormData();
        formData.append(fieldName, file);
        Object.entries(additionalData).forEach(([key, value]) => {
            formData.append(key, value);
        });

        const headers = {
            'X-CSRFToken': getCsrfToken(),
            'X-Requested-With': 'XMLHttpRequest'
        };

        const token = getAccessToken();
        if (token && !isTokenExpired(token)) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const response = await fetch(BASE_URL + endpoint, {
            method: 'POST',
            headers,
            credentials: 'same-origin',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw {
                status: response.status,
                message: data.message || data.detail || 'Upload failed',
                errors: data.errors || null
            };
        }

        return {
            success: true,
            message: data.message || 'Upload successful',
            data: data.data || data
        };
    }

    return {
        get: (endpoint, params = {}, options = {}) =>
            request('GET', endpoint, { ...options, params }),

        post: (endpoint, body = {}, options = {}) =>
            request('POST', endpoint, { ...options, body }),

        put: (endpoint, body = {}, options = {}) =>
            request('PUT', endpoint, { ...options, body }),

        patch: (endpoint, body = {}, options = {}) =>
            request('PATCH', endpoint, { ...options, body }),

        delete: (endpoint, options = {}) =>
            request('DELETE', endpoint, options),

        upload,
        clearCache,
        setTokens,
        clearTokens,
        getAccessToken,
        isAuthenticated: () => !!getAccessToken() && !isTokenExpired(getAccessToken())
    };
})();

window.ApiClient = ApiClient;
