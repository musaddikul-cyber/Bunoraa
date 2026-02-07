/**
 * Storage Utility
 * @module utils/storage
 */

const Storage = (function() {
    'use strict';

    function isAvailable(type) {
        try {
            const storage = window[type];
            const test = '__storage_test__';
            storage.setItem(test, test);
            storage.removeItem(test);
            return true;
        } catch (e) {
            return false;
        }
    }

    const hasLocalStorage = isAvailable('localStorage');
    const hasSessionStorage = isAvailable('sessionStorage');

    function set(key, value, persistent = true) {
        const storage = persistent ? localStorage : sessionStorage;
        
        if (!isAvailable(persistent ? 'localStorage' : 'sessionStorage')) {
            return false;
        }

        try {
            const data = {
                value,
                timestamp: Date.now()
            };
            storage.setItem(key, JSON.stringify(data));
            return true;
        } catch (e) {
            console.error('Storage error:', e);
            return false;
        }
    }

    function get(key, defaultValue = null, persistent = true) {
        const storage = persistent ? localStorage : sessionStorage;
        
        if (!isAvailable(persistent ? 'localStorage' : 'sessionStorage')) {
            return defaultValue;
        }

        try {
            const item = storage.getItem(key);
            if (!item) return defaultValue;
            
            const data = JSON.parse(item);
            return data.value !== undefined ? data.value : data;
        } catch (e) {
            return defaultValue;
        }
    }

    function remove(key, persistent = true) {
        const storage = persistent ? localStorage : sessionStorage;
        
        if (isAvailable(persistent ? 'localStorage' : 'sessionStorage')) {
            storage.removeItem(key);
        }
    }

    function clear(persistent = true) {
        const storage = persistent ? localStorage : sessionStorage;
        
        if (isAvailable(persistent ? 'localStorage' : 'sessionStorage')) {
            storage.clear();
        }
    }

    function getWithExpiry(key, defaultValue = null) {
        const item = get(key, null);
        
        if (!item) return defaultValue;
        
        if (item.expiry && Date.now() > item.expiry) {
            remove(key);
            return defaultValue;
        }
        
        return item.value !== undefined ? item.value : item;
    }

    function setWithExpiry(key, value, ttl) {
        const data = {
            value,
            expiry: Date.now() + ttl,
            timestamp: Date.now()
        };
        
        return set(key, data);
    }

    function keys(persistent = true) {
        const storage = persistent ? localStorage : sessionStorage;
        
        if (!isAvailable(persistent ? 'localStorage' : 'sessionStorage')) {
            return [];
        }
        
        return Object.keys(storage);
    }

    function size(persistent = true) {
        const storage = persistent ? localStorage : sessionStorage;
        
        if (!isAvailable(persistent ? 'localStorage' : 'sessionStorage')) {
            return 0;
        }
        
        let total = 0;
        for (const key of Object.keys(storage)) {
            total += storage.getItem(key).length;
        }
        return total;
    }

    return {
        set,
        get,
        remove,
        clear,
        getWithExpiry,
        setWithExpiry,
        keys,
        size,
        hasLocalStorage,
        hasSessionStorage
    };
})();

window.Storage = Storage;
