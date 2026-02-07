/**
 * Debounce Utility
 * @module utils/debounce
 */

const Debounce = (function() {
    'use strict';

    function debounce(fn, delay = 300) {
        let timer = null;
        
        const debounced = function(...args) {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), delay);
        };
        
        debounced.cancel = function() {
            clearTimeout(timer);
        };
        
        debounced.flush = function(...args) {
            clearTimeout(timer);
            fn.apply(this, args);
        };
        
        return debounced;
    }

    function throttle(fn, limit = 300) {
        let inThrottle = false;
        let lastArgs = null;
        
        return function(...args) {
            if (!inThrottle) {
                fn.apply(this, args);
                inThrottle = true;
                
                setTimeout(() => {
                    inThrottle = false;
                    if (lastArgs) {
                        fn.apply(this, lastArgs);
                        lastArgs = null;
                    }
                }, limit);
            } else {
                lastArgs = args;
            }
        };
    }

    function rafThrottle(fn) {
        let rafId = null;
        
        return function(...args) {
            if (rafId) return;
            
            rafId = requestAnimationFrame(() => {
                fn.apply(this, args);
                rafId = null;
            });
        };
    }

    function once(fn) {
        let called = false;
        let result;
        
        return function(...args) {
            if (!called) {
                called = true;
                result = fn.apply(this, args);
            }
            return result;
        };
    }

    function defer(fn, ...args) {
        return setTimeout(() => fn.apply(this, args), 0);
    }

    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    return {
        debounce,
        throttle,
        rafThrottle,
        once,
        defer,
        delay
    };
})();

window.Debounce = Debounce;
