/**
 * DOM Utilities
 * @module utils/dom
 */

const DOM = (function() {
    'use strict';

    function $(selector, context = document) {
        return context.querySelector(selector);
    }

    function $$(selector, context = document) {
        return Array.from(context.querySelectorAll(selector));
    }

    function create(tag, attrs = {}, children = []) {
        const el = document.createElement(tag);
        
        Object.entries(attrs).forEach(([key, value]) => {
            if (key === 'className') {
                el.className = value;
            } else if (key === 'dataset') {
                Object.entries(value).forEach(([k, v]) => {
                    el.dataset[k] = v;
                });
            } else if (key === 'style' && typeof value === 'object') {
                Object.assign(el.style, value);
            } else if (key.startsWith('on') && typeof value === 'function') {
                el.addEventListener(key.slice(2).toLowerCase(), value);
            } else {
                el.setAttribute(key, value);
            }
        });

        children.forEach(child => {
            if (typeof child === 'string') {
                el.appendChild(document.createTextNode(child));
            } else if (child instanceof Node) {
                el.appendChild(child);
            }
        });

        return el;
    }

    function html(strings, ...values) {
        const template = document.createElement('template');
        template.innerHTML = strings.reduce((acc, str, i) => {
            const value = values[i - 1];
            if (Array.isArray(value)) {
                return acc + value.join('') + str;
            }
            return acc + (value ?? '') + str;
        });
        return template.content.cloneNode(true);
    }

    function empty(el) {
        while (el.firstChild) {
            el.removeChild(el.firstChild);
        }
        return el;
    }

    function show(el) {
        if (el) el.classList.remove('hidden');
    }

    function hide(el) {
        if (el) el.classList.add('hidden');
    }

    function toggle(el, force) {
        if (el) el.classList.toggle('hidden', force !== undefined ? !force : undefined);
    }

    function on(el, event, handler, options = {}) {
        if (typeof el === 'string') {
            el = $(el);
        }
        if (!el) return () => {};
        
        el.addEventListener(event, handler, options);
        return () => el.removeEventListener(event, handler, options);
    }

    function delegate(parent, event, selector, handler) {
        if (typeof parent === 'string') {
            parent = $(parent);
        }
        if (!parent) return () => {};

        const listener = (e) => {
            const target = e.target.closest(selector);
            if (target && parent.contains(target)) {
                handler.call(target, e, target);
            }
        };

        parent.addEventListener(event, listener);
        return () => parent.removeEventListener(event, listener);
    }

    function ready(fn) {
        if (document.readyState !== 'loading') {
            fn();
        } else {
            document.addEventListener('DOMContentLoaded', fn);
        }
    }

    function scrollTo(el, options = {}) {
        if (typeof el === 'string') {
            el = $(el);
        }
        if (!el) return;

        el.scrollIntoView({
            behavior: 'smooth',
            block: 'start',
            ...options
        });
    }

    function animate(el, keyframes, options = {}) {
        if (typeof el === 'string') {
            el = $(el);
        }
        if (!el) return Promise.resolve();

        return el.animate(keyframes, {
            duration: 300,
            easing: 'ease-in-out',
            fill: 'forwards',
            ...options
        }).finished;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function parseHtml(htmlString) {
        const template = document.createElement('template');
        template.innerHTML = htmlString.trim();
        return template.content.firstChild;
    }

    return {
        $,
        $$,
        create,
        html,
        empty,
        show,
        hide,
        toggle,
        on,
        delegate,
        ready,
        scrollTo,
        animate,
        escapeHtml,
        parseHtml
    };
})();

window.DOM = DOM;
