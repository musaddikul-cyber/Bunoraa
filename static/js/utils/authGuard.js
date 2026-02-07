/**
 * Auth Guard Utility
 * @module utils/authGuard
 */

const AuthGuard = (function() {
    'use strict';

    let loginUrl = '/account/login/';
    let redirectParam = 'next';

    function setLoginUrl(url) {
        loginUrl = url;
    }

    function requireAuth(callback) {
        return function(...args) {
            if (!AuthApi.isAuthenticated()) {
                const returnUrl = encodeURIComponent(window.location.pathname + window.location.search);
                window.location.href = `${loginUrl}?${redirectParam}=${returnUrl}`;
                return;
            }
            return callback.apply(this, args);
        };
    }

    function checkAuth() {
        return AuthApi.isAuthenticated();
    }

    function redirectToLogin(returnUrl = null) {
        const url = returnUrl || window.location.pathname + window.location.search;
        window.location.href = `${loginUrl}?${redirectParam}=${encodeURIComponent(url)}`;
    }

    function getReturnUrl() {
        const params = new URLSearchParams(window.location.search);
        return params.get(redirectParam) || '/';
    }

    function protectPage() {
        if (!checkAuth()) {
            redirectToLogin();
            return false;
        }
        return true;
    }

    function protectElement(element, options = {}) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        if (!element) return;

        if (!checkAuth()) {
            if (options.hide) {
                element.classList.add('hidden');
            } else if (options.disable) {
                element.setAttribute('disabled', 'disabled');
                element.classList.add('opacity-50', 'cursor-not-allowed');
            }
            
            if (options.onClick !== false) {
                element.addEventListener('click', (e) => {
                    e.preventDefault();
                    if (options.showModal) {
                        Toast.info('Please log in to continue');
                    }
                    redirectToLogin();
                });
            }
        }
    }

    function init() {
        window.addEventListener('auth:required', () => {
            redirectToLogin();
        });

        window.addEventListener('auth:expired', () => {
            Toast.warning('Your session has expired. Please log in again.');
            setTimeout(() => redirectToLogin(), 2000);
        });

        document.querySelectorAll('[data-auth-required]').forEach(el => {
            protectElement(el, {
                hide: el.dataset.authHide !== undefined,
                disable: el.dataset.authDisable !== undefined,
                showModal: true
            });
        });
    }

    return {
        setLoginUrl,
        requireAuth,
        checkAuth,
        redirectToLogin,
        getReturnUrl,
        protectPage,
        protectElement,
        init
    };
})();

window.AuthGuard = AuthGuard;
