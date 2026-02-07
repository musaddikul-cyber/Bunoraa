/**
 * Localization API Module
 * @module api/localization
 */

const LocalizationApi = (function() {
    'use strict';

    const CURRENCY_KEY = 'selected_currency';
    const CURRENCY_RATE_KEY = 'currency_rate';
    const LANGUAGE_KEY = 'selected_language';
    const TIMEZONE_KEY = 'selected_timezone';
    const BASE_CURRENCY = 'BDT';

    async function getCurrencies() {
        return ApiClient.get('/i18n/currencies/', {}, { useCache: true, cacheTTL: 3600000 });
    }

    async function getLanguages() {
        return ApiClient.get('/i18n/languages/', {}, { useCache: true, cacheTTL: 3600000 });
    }

    async function getTimezones() {
        return ApiClient.get('/i18n/timezones/', {}, { useCache: true, cacheTTL: 3600000 });
    }

    async function getCountries() {
        return ApiClient.get('/i18n/countries/', {}, { useCache: true, cacheTTL: 3600000 });
    }

    async function getDivisions(countryCode) {
        return ApiClient.get(`/i18n/countries/${countryCode}/divisions/`, {}, { useCache: true, cacheTTL: 3600000 });
    }

    async function convertCurrency(amount, from, to) {
        return ApiClient.get('/i18n/convert/', { amount, from, to });
    }
    
    async function getExchangeRate(from, to) {
        // Client-side exchange rate fetching disabled; always return 1 (no conversion)
        return 1;
    }

    async function setCurrency(code) {
        // No-op preference setter for single-currency mode; keep global meta updated for compatibility
        try {
            window.BUNORAA_CURRENCY = Object.assign({}, window.BUNORAA_CURRENCY || {}, { code });
        } catch (e) {
            // ignore
        }
        // Currency change event dispatch removed (single-currency mode).
    }

    function getCurrency() {
        return (window.BUNORAA_CURRENCY && window.BUNORAA_CURRENCY.code) || 'BDT';
    }

    async function getCurrentCurrency() {
        try {
            const resp = await ApiClient.get('/i18n/currencies/current/');
            if (resp && resp.data && resp.data.code) return resp.data.code;
        } catch (err) {
            // ignore
        }
        return getCurrency();
    }
    
    function getStoredExchangeRate() {
        // Client-side exchange rates disabled - always return 1 (no conversion on client)
        return 1;
    }

    function setLanguage(code) {
        localStorage.setItem(LANGUAGE_KEY, code);
        window.dispatchEvent(new CustomEvent('language:changed', { detail: code }));
    }

    function getLanguage() {
        return localStorage.getItem(LANGUAGE_KEY) || 'en';
    }

    function setTimezone(tz) {
        localStorage.setItem(TIMEZONE_KEY, tz);
        window.dispatchEvent(new CustomEvent('timezone:changed', { detail: tz }));
    }

    function getTimezone() {
        return localStorage.getItem(TIMEZONE_KEY) || Intl.DateTimeFormat().resolvedOptions().timeZone;
    }

    return {
        getCurrencies,
        getLanguages,
        getTimezones,
        getCountries,
        getDivisions,
        convertCurrency,
        getExchangeRate,
        setCurrency,
        getCurrency,
        getCurrentCurrency,
        getStoredExchangeRate,
        setLanguage,
        getLanguage,
        setTimezone,
        getTimezone
    };
})();

window.LocalizationApi = LocalizationApi;
