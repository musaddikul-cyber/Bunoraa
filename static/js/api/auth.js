/**
 * Auth API Module
 * @module api/auth
 */

const AuthApi = (function() {
    'use strict';

    const USER_KEY = 'current_user';

    function hasSessionAuth() {
        if (typeof window === 'undefined') return false;
        return !!window.__DJANGO_SESSION_AUTH__;
    }

    function setSessionAuth(value) {
        if (typeof window === 'undefined') return;
        window.__DJANGO_SESSION_AUTH__ = value;
    }

    async function login(email, password, remember = false) {
        const response = await ApiClient.post('/auth/token/', { email, password });
        
        if (response.data?.access) {
            ApiClient.setTokens(response.data.access, response.data.refresh, remember);
            
            try {
                const profile = await getProfile();
                if (profile.success && profile.data) {
                    setUser(profile.data);
                }
            } catch (e) {
            }
            
            ApiClient.clearCache();
            window.dispatchEvent(new CustomEvent('auth:login'));
            
            return { success: true, message: 'Login successful', data: getUser() };
        }
        
        return response;
    }

    async function register(userData) {
        const response = await ApiClient.post('/accounts/register/', userData);
        
        if (response.success) {
            window.dispatchEvent(new CustomEvent('auth:registered'));
        }
        
        return response;
    }

    async function logout() {
        try {
            await ApiClient.post('/accounts/logout/');
        } catch (e) {
            // Ignore logout API errors
        }
        
        ApiClient.clearTokens();
        clearUser();
        ApiClient.clearCache();
        setSessionAuth(false);
        window.dispatchEvent(new CustomEvent('auth:logout'));
    }

    async function getProfile() {
        return ApiClient.get('/accounts/profile/', {}, { requiresAuth: true });
    }

    async function updateProfile(data) {
        const response = await ApiClient.patch('/accounts/profile/', data, { requiresAuth: true });
        
        if (response.success && response.data) {
            setUser(response.data);
            window.dispatchEvent(new CustomEvent('auth:profile-updated'));
        }
        
        return response;
    }

    async function uploadAvatar(file) {
        const response = await ApiClient.upload('/accounts/profile/avatar/', file, 'avatar');
        
        if (response.success) {
            const user = getUser();
            if (user && response.data?.avatar) {
                user.avatar = response.data.avatar;
                setUser(user);
            }
        }
        
        return response;
    }

    async function changePassword(currentPassword, newPassword) {
        return ApiClient.post('/accounts/password/change/', {
            current_password: currentPassword,
            new_password: newPassword
        }, { requiresAuth: true });
    }

    async function requestPasswordReset(email) {
        return ApiClient.post('/accounts/password/reset/request/', { email });
    }

    async function resetPassword(token, password) {
        return ApiClient.post('/accounts/password/reset/', { token, password });
    }

    async function verifyEmail(token) {
        return ApiClient.post('/accounts/email/verify/', { token });
    }

    async function resendVerification(email) {
        return ApiClient.post('/accounts/email/resend/', { email });
    }

    async function getAddresses() {
        return ApiClient.get('/accounts/addresses/', {}, { requiresAuth: true });
    }

    async function getAddress(id) {
        return ApiClient.get(`/accounts/addresses/${id}/`, {}, { requiresAuth: true });
    }

    async function addAddress(data) {
        const response = await ApiClient.post('/accounts/addresses/', data, { requiresAuth: true });
        if (response.success) {
            window.dispatchEvent(new CustomEvent('address:added'));
        }
        return response;
    }

    async function updateAddress(id, data) {
        const response = await ApiClient.patch(`/accounts/addresses/${id}/`, data, { requiresAuth: true });
        if (response.success) {
            window.dispatchEvent(new CustomEvent('address:updated'));
        }
        return response;
    }

    async function deleteAddress(id) {
        const response = await ApiClient.delete(`/accounts/addresses/${id}/`, { requiresAuth: true });
        if (response.success) {
            window.dispatchEvent(new CustomEvent('address:deleted'));
        }
        return response;
    }

    async function setDefaultAddress(id, type = 'both') {
        return ApiClient.post(`/accounts/addresses/${id}/set-default/`, { type }, { requiresAuth: true });
    }

    function setUser(user) {
        localStorage.setItem(USER_KEY, JSON.stringify(user));
        window.dispatchEvent(new CustomEvent('user:changed', { detail: user }));
    }

    function getUser() {
        try {
            const data = localStorage.getItem(USER_KEY);
            return data ? JSON.parse(data) : null;
        } catch {
            return null;
        }
    }

    function clearUser() {
        localStorage.removeItem(USER_KEY);
    }

    function isAuthenticated() {
        return ApiClient.isAuthenticated() || hasSessionAuth();
    }

    return {
        login,
        register,
        logout,
        getProfile,
        updateProfile,
        uploadAvatar,
        changePassword,
        requestPasswordReset,
        resetPassword,
        verifyEmail,
        resendVerification,
        getAddresses,
        getAddress,
        addAddress,
        updateAddress,
        deleteAddress,
        setDefaultAddress,
        getUser,
        setUser,
        isAuthenticated
    };
})();

window.AuthApi = AuthApi;
