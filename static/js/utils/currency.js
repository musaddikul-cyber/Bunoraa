/**
 * Comprehensive Currency Utility Module
 * 
 * Features:
 * - Client-side currency conversion with caching
 * - Smart formatting with Intl.NumberFormat fallback
 * - Exchange rate caching and batch conversion
 * - Currency detection and preference management
 */

// =============================================================================
// Constants & Cache
// =============================================================================

const RATE_CACHE = new Map();
const RATE_CACHE_TTL = 5 * 60 * 1000; // 5 minutes
const CURRENCY_DETAILS_CACHE = new Map();

// =============================================================================
// API Functions
// =============================================================================

/**
 * Convert amount between currencies via API
 * @param {number|string} amount - Amount to convert
 * @param {string} from - Source currency code
 * @param {string} to - Target currency code
 * @param {boolean} useCache - Whether to use cached rates
 * @returns {Promise<object|null>} Conversion result
 */
export async function convertCurrency(amount, from, to, useCache = true) {
    from = String(from).toUpperCase();
    to = String(to).toUpperCase();
    
    // Same currency
    if (from === to) {
        return {
            original_amount: amount,
            converted_amount: amount,
            from_currency: from,
            to_currency: to,
            rate: 1
        };
    }
    
    // Check cache first
    if (useCache) {
        const cachedRate = getCachedRate(from, to);
        if (cachedRate) {
            const converted = Number(amount) * cachedRate;
            return {
                original_amount: amount,
                converted_amount: converted.toFixed(2),
                from_currency: from,
                to_currency: to,
                rate: cachedRate,
                cached: true
            };
        }
    }
    
    try {
        const resp = await fetch('/api/v1/i18n/convert/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({
                amount: String(amount),
                from_currency: from,
                to_currency: to,
                round_result: true
            })
        });
        
        if (!resp.ok) throw new Error('Conversion failed');
        
        const data = await resp.json();
        if (data && data.success) {
            // Cache the rate
            if (data.data && data.data.rate) {
                cacheRate(from, to, Number(data.data.rate));
            }
            return data.data;
        }
    } catch (e) {
        console.warn('Currency conversion error:', e);
    }
    return null;
}

/**
 * Batch convert multiple amounts
 * @param {Array<{amount: number, from: string, to: string}>} conversions
 * @returns {Promise<Array>} Array of conversion results
 */
export async function batchConvert(conversions) {
    try {
        const resp = await fetch('/api/v1/i18n/batch-convert/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({ conversions })
        });
        
        if (!resp.ok) throw new Error('Batch conversion failed');
        
        const data = await resp.json();
        return data.success ? data.data : [];
    } catch (e) {
        // Fallback to individual conversions
        const results = [];
        for (const conv of conversions) {
            results.push(await convertCurrency(conv.amount, conv.from, conv.to));
        }
        return results;
    }
}

/**
 * Get current exchange rate
 * @param {string} from - Source currency code
 * @param {string} to - Target currency code
 * @returns {Promise<number|null>} Exchange rate
 */
export async function getExchangeRate(from, to) {
    from = String(from).toUpperCase();
    to = String(to).toUpperCase();
    
    if (from === to) return 1;
    
    // Check cache
    const cached = getCachedRate(from, to);
    if (cached) return cached;
    
    try {
        const resp = await fetch(`/api/v1/i18n/rates/?from=${from}&to=${to}`);
        if (!resp.ok) throw new Error('Failed to get rate');
        
        const data = await resp.json();
        if (data.success && data.data && data.data.rate) {
            cacheRate(from, to, Number(data.data.rate));
            return Number(data.data.rate);
        }
    } catch (e) {
        console.warn('Get exchange rate error:', e);
    }
    return null;
}

/**
 * Get available currencies
 * @returns {Promise<Array>} List of active currencies
 */
export async function getActiveCurrencies() {
    try {
        const resp = await fetch('/api/v1/i18n/currencies/');
        if (!resp.ok) throw new Error('Failed to fetch currencies');
        
        const data = await resp.json();
        return data.success ? data.data : [];
    } catch (e) {
        console.warn('Get currencies error:', e);
        return [];
    }
}

/**
 * Get currency details by code
 * @param {string} code - Currency code
 * @returns {Promise<object|null>} Currency details
 */
export async function getCurrencyDetails(code) {
    code = String(code).toUpperCase();
    
    if (CURRENCY_DETAILS_CACHE.has(code)) {
        return CURRENCY_DETAILS_CACHE.get(code);
    }
    
    try {
        const resp = await fetch(`/api/v1/i18n/currencies/${code}/`);
        if (!resp.ok) throw new Error('Currency not found');
        
        const data = await resp.json();
        if (data.success && data.data) {
            CURRENCY_DETAILS_CACHE.set(code, data.data);
            return data.data;
        }
    } catch (e) {
        console.warn('Get currency details error:', e);
    }
    return null;
}

// =============================================================================
// Formatting Functions
// =============================================================================

/**
 * Format amount with currency
 * @param {number|string} value - Amount to format
 * @param {string|object} currency - Currency code or config object
 * @param {string} locale - Locale for formatting
 * @returns {string} Formatted currency string
 */
export function formatCurrency(value, currency = null, locale = navigator.language) {
    // Use global currency if not specified
    if (!currency && typeof window !== 'undefined' && window.BUNORAA_CURRENCY) {
        currency = window.BUNORAA_CURRENCY;
    }

    if (!currency) {
        return String(value);
    }

    // Normalize value
    let num = Number(value);
    if (Number.isNaN(num)) return String(value);

    // If currency is a string code, use Intl.NumberFormat
    if (typeof currency === 'string') {
        try {
            return new Intl.NumberFormat(locale, {
                style: 'currency',
                currency: currency.toUpperCase(),
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            }).format(num);
        } catch (e) {
            return `${currency.toUpperCase()} ${num.toFixed(2)}`;
        }
    }

    // If currency is an object with formatting details
    return formatWithConfig(num, currency);
}

/**
 * Format with custom configuration
 * @param {number} num - Amount
 * @param {object} config - Currency configuration
 * @returns {string} Formatted string
 */
function formatWithConfig(num, config) {
    const symbol = config.symbol || config.code || '';
    const decimals = typeof config.decimal_places === 'number' ? config.decimal_places : 2;
    const thousand = config.thousand_separator || ',';
    const decSep = config.decimal_separator || '.';
    const symbolPos = config.symbol_position || 'before';
    const spacing = config.symbol_spacing !== false;

    const fixed = num.toFixed(decimals);
    const [intPart, decPart] = fixed.split('.');
    
    // Add thousand separator
    const formattedInt = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, thousand);
    const formattedNumber = decimals > 0 ? formattedInt + decSep + decPart : formattedInt;

    if (symbolPos === 'after') {
        return spacing ? `${formattedNumber} ${symbol}` : `${formattedNumber}${symbol}`;
    }
    return spacing ? `${symbol} ${formattedNumber}` : `${symbol}${formattedNumber}`;
}

/**
 * Format with compact notation for large numbers
 * @param {number} value - Amount
 * @param {string} currency - Currency code
 * @param {string} locale - Locale
 * @returns {string} Compact formatted string
 */
export function formatCompact(value, currency = 'BDT', locale = navigator.language) {
    const num = Number(value);
    if (Number.isNaN(num)) return String(value);
    
    try {
        return new Intl.NumberFormat(locale, {
            style: 'currency',
            currency: currency.toUpperCase(),
            notation: 'compact',
            compactDisplay: 'short'
        }).format(num);
    } catch (e) {
        // Fallback
        if (num >= 1000000) {
            return `${currency} ${(num / 1000000).toFixed(1)}M`;
        } else if (num >= 1000) {
            return `${currency} ${(num / 1000).toFixed(1)}K`;
        }
        return formatCurrency(num, currency, locale);
    }
}

/**
 * Format price range
 * @param {number} min - Minimum price
 * @param {number} max - Maximum price
 * @param {string} currency - Currency code
 * @returns {string} Formatted range
 */
export function formatPriceRange(min, max, currency = 'BDT') {
    if (min === max) {
        return formatCurrency(min, currency);
    }
    return `${formatCurrency(min, currency)} - ${formatCurrency(max, currency)}`;
}

// =============================================================================
// Cache Management
// =============================================================================

function getCacheKey(from, to) {
    return `${from}_${to}`;
}

function getCachedRate(from, to) {
    const key = getCacheKey(from, to);
    const cached = RATE_CACHE.get(key);
    
    if (cached && Date.now() - cached.timestamp < RATE_CACHE_TTL) {
        return cached.rate;
    }
    
    // Check inverse
    const inverseKey = getCacheKey(to, from);
    const inverseCached = RATE_CACHE.get(inverseKey);
    
    if (inverseCached && Date.now() - inverseCached.timestamp < RATE_CACHE_TTL) {
        return 1 / inverseCached.rate;
    }
    
    return null;
}

function cacheRate(from, to, rate) {
    const key = getCacheKey(from, to);
    RATE_CACHE.set(key, { rate, timestamp: Date.now() });
}

/**
 * Clear rate cache
 */
export function clearRateCache() {
    RATE_CACHE.clear();
}

/**
 * Preload exchange rates for common currencies
 * @param {string} baseCurrency - Base currency to load rates for
 */
export async function preloadRates(baseCurrency = 'BDT') {
    const commonCurrencies = ['USD', 'EUR', 'GBP', 'BDT', 'INR', 'CAD', 'AUD', 'JPY'];
    
    for (const currency of commonCurrencies) {
        if (currency !== baseCurrency) {
            await getExchangeRate(baseCurrency, currency);
        }
    }
}

// =============================================================================
// Client-side Conversion (with cached rates)
// =============================================================================

/**
 * Convert using cached rate (no API call)
 * @param {number} amount - Amount to convert
 * @param {string} from - Source currency
 * @param {string} to - Target currency
 * @returns {number|null} Converted amount or null if no cached rate
 */
export function convertCached(amount, from, to) {
    if (from === to) return Number(amount);
    
    const rate = getCachedRate(from, to);
    if (rate) {
        return Number(amount) * rate;
    }
    return null;
}

/**
 * Convert and format in one call
 * @param {number} amount - Amount to convert
 * @param {string} from - Source currency
 * @param {string} to - Target currency
 * @returns {Promise<string>} Formatted converted amount
 */
export async function convertAndFormat(amount, from, to) {
    const result = await convertCurrency(amount, from, to);
    if (result && result.converted_amount) {
        return formatCurrency(result.converted_amount, to);
    }
    return formatCurrency(amount, from);
}

// =============================================================================
// Currency Detection & Preferences
// =============================================================================

/**
 * Get user's current currency (from session/cookie/global)
 * @returns {string} Currency code
 */
export function getCurrentCurrency() {
    // Check global
    if (typeof window !== 'undefined') {
        if (window.BUNORAA_CURRENCY) {
            const curr = window.BUNORAA_CURRENCY;
            return typeof curr === 'string' ? curr : (curr.code || 'BDT');
        }
    }
    
    // Check cookie
    const cookies = document.cookie.split(';');
    for (const cookie of cookies) {
        const [name, value] = cookie.trim().split('=');
        if (name === 'currency') {
            return value.toUpperCase();
        }
    }
    
    return 'BDT';
}

/**
 * Set user's preferred currency
 * @param {string} code - Currency code
 * @param {boolean} persist - Save to cookie
 */
export function setCurrentCurrency(code, persist = true) {
    code = String(code).toUpperCase();
    
    if (typeof window !== 'undefined') {
        if (typeof window.BUNORAA_CURRENCY === 'object') {
            window.BUNORAA_CURRENCY.code = code;
        } else {
            window.BUNORAA_CURRENCY = code;
        }
    }
    
    if (persist) {
        document.cookie = `currency=${code};path=/;max-age=31536000;SameSite=Lax`;
    }
    
    // Dispatch event for components to react
    window.dispatchEvent(new CustomEvent('currencyChange', { detail: { currency: code } }));
}

// =============================================================================
// Price Display Helpers
// =============================================================================

/**
 * Display prices with original and converted amounts
 * @param {number} amount - Original amount
 * @param {string} originalCurrency - Original currency
 * @param {string} displayCurrency - Currency to display in
 * @returns {Promise<object>} Both formatted prices
 */
export async function dualPrice(amount, originalCurrency, displayCurrency = null) {
    displayCurrency = displayCurrency || getCurrentCurrency();
    
    const original = formatCurrency(amount, originalCurrency);
    
    if (originalCurrency === displayCurrency) {
        return { primary: original, secondary: null, same: true };
    }
    
    const converted = await convertCurrency(amount, originalCurrency, displayCurrency);
    
    return {
        primary: converted ? formatCurrency(converted.converted_amount, displayCurrency) : original,
        secondary: converted ? original : null,
        same: false,
        rate: converted ? converted.rate : null
    };
}

/**
 * Update all price elements on page
 * @param {string} targetCurrency - Target currency
 * @param {string} selector - CSS selector for price elements
 */
export async function updateAllPrices(targetCurrency, selector = '[data-price]') {
    const elements = document.querySelectorAll(selector);
    
    for (const el of elements) {
        const originalPrice = el.dataset.price;
        const originalCurrency = el.dataset.currency || 'BDT';
        
        if (!originalPrice) continue;
        
        if (originalCurrency === targetCurrency) {
            el.textContent = formatCurrency(originalPrice, originalCurrency);
        } else {
            const result = await convertCurrency(originalPrice, originalCurrency, targetCurrency);
            if (result) {
                el.textContent = formatCurrency(result.converted_amount, targetCurrency);
            }
        }
    }
}

// =============================================================================
// Export
// =============================================================================

export default {
    // Conversion
    convertCurrency,
    batchConvert,
    getExchangeRate,
    convertCached,
    convertAndFormat,
    
    // Formatting
    formatCurrency,
    formatCompact,
    formatPriceRange,
    
    // Currencies
    getActiveCurrencies,
    getCurrencyDetails,
    getCurrentCurrency,
    setCurrentCurrency,
    
    // Cache
    clearRateCache,
    preloadRates,
    
    // Display helpers
    dualPrice,
    updateAllPrices
};
