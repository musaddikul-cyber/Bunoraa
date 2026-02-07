/**
 * Base Component Class
 * Provides common functionality for all UI components
 */

import { clsx, createElement, addListener, generateId } from './utils.js';

export class BaseComponent {
  constructor(options = {}) {
    this.id = options.id || generateId();
    this.element = null;
    this.listeners = [];
    this.isInitialized = false;
    this.config = options;
  }

  /**
   * Create the component element
   */
  create(tag = 'div', { className = '', attrs = {}, html = '' } = {}) {
    this.element = createElement(tag, {
      id: this.id,
      className,
      attrs,
      html
    });
    return this.element;
  }

  /**
   * Mount component to DOM
   */
  mount(selector) {
    if (!this.element) {
      return false;
    }

    const target = typeof selector === 'string' 
      ? document.querySelector(selector) 
      : selector;

    if (!target) {
      return false;
    }

    target.appendChild(this.element);
    this.isInitialized = true;
    return true;
  }

  /**
   * Add event listener with cleanup
   */
  on(event, handler, options = {}) {
    if (!this.element) return;
    
    const cleanup = addListener(this.element, event, handler, options);
    this.listeners.push(cleanup);
    return cleanup;
  }

  /**
   * Add delegated event listener
   */
  delegate(selector, event, handler) {
    if (!this.element) return;

    const listener = (e) => {
      const target = e.target.closest(selector);
      if (target) handler.call(target, e);
    };

    this.element.addEventListener(event, listener);
    this.listeners.push(() => this.element.removeEventListener(event, listener));
  }

  /**
   * Add class to element
   */
  addClass(...classes) {
    if (this.element) {
      this.element.classList.add(...classes);
    }
  }

  /**
   * Remove class from element
   */
  removeClass(...classes) {
    if (this.element) {
      this.element.classList.remove(...classes);
    }
  }

  /**
   * Toggle class on element
   */
  toggleClass(className, force) {
    if (this.element) {
      this.element.classList.toggle(className, force);
    }
  }

  /**
   * Check if element has class
   */
  hasClass(className) {
    return this.element?.classList.contains(className) ?? false;
  }

  /**
   * Set attribute
   */
  attr(name, value) {
    if (!this.element) return;
    
    if (value === undefined) {
      return this.element.getAttribute(name);
    }
    
    if (value === null || value === false) {
      this.element.removeAttribute(name);
    } else if (value === true) {
      this.element.setAttribute(name, '');
    } else {
      this.element.setAttribute(name, value);
    }
  }

  /**
   * Set multiple attributes
   */
  attrs(attributes) {
    Object.entries(attributes).forEach(([key, value]) => {
      this.attr(key, value);
    });
  }

  /**
   * Set text content
   */
  text(content) {
    if (this.element) {
      this.element.textContent = content;
    }
  }

  /**
   * Set HTML content
   */
  html(content) {
    if (this.element) {
      this.element.innerHTML = content;
    }
  }

  /**
   * Append child element
   */
  append(element) {
    if (this.element && element) {
      this.element.appendChild(element instanceof BaseComponent ? element.element : element);
    }
  }

  /**
   * Prepend child element
   */
  prepend(element) {
    if (this.element && element) {
      this.element.prepend(element instanceof BaseComponent ? element.element : element);
    }
  }

  /**
   * Show element
   */
  show() {
    if (this.element) {
      this.element.style.display = '';
      this.element.removeAttribute('hidden');
    }
  }

  /**
   * Hide element
   */
  hide() {
    if (this.element) {
      this.element.style.display = 'none';
    }
  }

  /**
   * Toggle visibility
   */
  toggle(force) {
    if (this.element) {
      if (force === undefined) {
        force = this.element.style.display === 'none';
      }
      force ? this.show() : this.hide();
    }
  }

  /**
   * Get computed style
   */
  getStyle(property) {
    if (!this.element) return null;
    return window.getComputedStyle(this.element).getPropertyValue(property);
  }

  /**
   * Set style
   */
  setStyle(property, value) {
    if (this.element) {
      this.element.style[property] = value;
    }
  }

  /**
   * Set multiple styles
   */
  setStyles(styles) {
    Object.entries(styles).forEach(([key, value]) => {
      this.setStyle(key, value);
    });
  }

  /**
   * Focus element
   * By default prevents scrolling the page when focusing (avoid jumping to focused element on load).
   * Pass an options object to control behavior (e.g., { preventScroll: false }).
   */
  focus(options) {
    if (!this.element) return;

    try {
      if (typeof options === 'undefined') {
        // default: avoid changing scroll position when focusing
        this.element.focus({ preventScroll: true });
      } else {
        this.element.focus(options);
      }
    } catch (err) {
      // Fallback for older browsers that don't support focus options
      try { this.element.focus(); } catch (e) { /* ignore */ }
    }
  }

  /**
   * Blur element
   */
  blur() {
    if (this.element) {
      this.element.blur();
    }
  }

  /**
   * Get position relative to viewport
   */
  getPosition() {
    if (!this.element) return null;
    return this.element.getBoundingClientRect();
  }

  /**
   * Destroy component and cleanup
   */
  destroy() {
    // Remove all event listeners
    this.listeners.forEach(cleanup => cleanup?.());
    this.listeners = [];

    // Remove element from DOM
    if (this.element?.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }

    this.element = null;
    this.isInitialized = false;
  }

  /**
   * Initialize component (override in subclasses)
   */
  init() {
    if (this.element && !this.isInitialized) {
      this.isInitialized = true;
    }
  }

  /**
   * Render component (override in subclasses)
   */
  render() {
    return this.element;
  }
}
