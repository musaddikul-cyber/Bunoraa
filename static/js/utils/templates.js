/**
 * Template Utilities
 * @module utils/templates
 */

const Templates = (function() {
    'use strict';

    const cache = new Map();
    // Use a stable locale for date/time formatting (fallbacks to user agent)
    const locale = (window.BUNORAA_CURRENCY && window.BUNORAA_CURRENCY.locale) || navigator.language || 'en-US';
    // Price formatting now uses server-provided currency metadata (no client-side exchange conversions)

    /**
     * Format a price. If sourceCurrency is provided, the function assumes
     * `amount` is expressed in that currency and will avoid double-converting.
     * If no sourceCurrency is provided, the amount is treated as server/base currency
     * and will be converted to the user-selected currency when a stored rate exists.
     */
    function formatPrice(amount, sourceCurrency = null) {
        if (amount === null || amount === undefined) return '';

        // Use server-provided currency metadata only
        const baseMeta = (window.BUNORAA_CURRENCY && typeof window.BUNORAA_CURRENCY === 'object') ? window.BUNORAA_CURRENCY : { code: 'BDT', symbol: '৳', locale: 'en-BD', decimal_places: 2 };
        const targetCode = baseMeta.code || 'BDT';
        const targetLocale = baseMeta.locale || 'en-US';
        const targetDecimals = (typeof baseMeta.decimal_places === 'number') ? baseMeta.decimal_places : 2;

        let value = Number(amount);
        if (Number.isNaN(value)) return '';

        try {
            return new Intl.NumberFormat(targetLocale, {
                style: 'currency',
                currency: targetCode,
                minimumFractionDigits: targetDecimals,
                maximumFractionDigits: targetDecimals
            }).format(value);
        } catch (e) {
            const symbol = baseMeta.symbol || '৳';
            return symbol + Number(value).toFixed(targetDecimals);
        }
    }

    function formatDate(date, options = {}) {
        if (!date) return '';
        
        const d = date instanceof Date ? date : new Date(date);
        
        return d.toLocaleDateString(locale, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            ...options
        });
    }

    function formatDateTime(date, options = {}) {
        if (!date) return '';
        
        const d = date instanceof Date ? date : new Date(date);
        
        return d.toLocaleString(locale, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            ...options
        });
    }

    function formatRelativeTime(date) {
        if (!date) return '';
        
        const d = date instanceof Date ? date : new Date(date);
        const now = new Date();
        const diff = now - d;
        const seconds = Math.floor(diff / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        const weeks = Math.floor(days / 7);
        const months = Math.floor(days / 30);
        const years = Math.floor(days / 365);

        if (years > 0) return `${years} year${years > 1 ? 's' : ''} ago`;
        if (months > 0) return `${months} month${months > 1 ? 's' : ''} ago`;
        if (weeks > 0) return `${weeks} week${weeks > 1 ? 's' : ''} ago`;
        if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
        if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        return 'Just now';
    }

    function formatNumber(num, decimals = 0) {
        if (num === null || num === undefined) return '';
        
        return new Intl.NumberFormat(locale, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(num);
    }

    function truncate(text, length = 100, suffix = '...') {
        if (!text) return '';
        if (text.length <= length) return text;
        return text.substring(0, length).trim() + suffix;
    }

    function slugify(text) {
        return text
            .toLowerCase()
            .trim()
            .replace(/[^\w\s-]/g, '')
            .replace(/[\s_-]+/g, '-')
            .replace(/^-+|-+$/g, '');
    }

    function pluralize(count, singular, plural = null) {
        plural = plural || singular + 's';
        return count === 1 ? singular : plural;
    }

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function render(template, data = {}) {
        return template.replace(/\{\{(\w+(?:\.\w+)*)\}\}/g, (match, path) => {
            const value = path.split('.').reduce((obj, key) => obj?.[key], data);
            return value !== undefined ? escapeHtml(String(value)) : '';
        });
    }

    function compile(templateStr) {
        if (cache.has(templateStr)) {
            return cache.get(templateStr);
        }

        const fn = (data) => render(templateStr, data);
        cache.set(templateStr, fn);
        return fn;
    }

    function getTemplate(id) {
        const el = document.getElementById(id);
        if (!el) return null;
        return el.innerHTML.trim();
    }

    function generateStars(rating, maxRating = 5) {
        const fullStars = Math.floor(rating);
        const halfStar = rating % 1 >= 0.5;
        const emptyStars = maxRating - fullStars - (halfStar ? 1 : 0);
        
        let html = '';
        
        for (let i = 0; i < fullStars; i++) {
            html += '<svg class="w-4 h-4 text-yellow-400 fill-current" viewBox="0 0 20 20"><path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z"/></svg>';
        }
        
        if (halfStar) {
            html += '<svg class="w-4 h-4 text-yellow-400" viewBox="0 0 20 20"><defs><linearGradient id="half"><stop offset="50%" stop-color="currentColor"/><stop offset="50%" stop-color="#d1d5db"/></defs><path fill="url(#half)" d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z"/></svg>';
        }
        
        for (let i = 0; i < emptyStars; i++) {
            html += '<svg class="w-4 h-4 text-gray-300 fill-current" viewBox="0 0 20 20"><path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z"/></svg>';
        }
        
        return html;
    }

    function getStatusBadge(status) {
        const statusClasses = {
            pending: 'bg-yellow-100 text-yellow-800',
            processing: 'bg-blue-100 text-blue-800',
            shipped: 'bg-purple-100 text-purple-800',
            delivered: 'bg-green-100 text-green-800',
            cancelled: 'bg-red-100 text-red-800',
            refunded: 'bg-gray-100 text-gray-800',
            completed: 'bg-green-100 text-green-800',
            failed: 'bg-red-100 text-red-800'
        };
        
        const className = statusClasses[status?.toLowerCase()] || 'bg-gray-100 text-gray-800';
        return `<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${className}">${escapeHtml(status)}</span>`;
    }

    return {
        formatPrice,
        formatDate,
        formatDateTime,
        formatRelativeTime,
        formatNumber,
        truncate,
        slugify,
        pluralize,
        escapeHtml,
        render,
        compile,
        getTemplate,
        generateStars,
        getStatusBadge
    };
})();

window.Templates = Templates;
