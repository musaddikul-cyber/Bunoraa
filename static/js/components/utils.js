/**
 * Utilities and helpers for UI components
 */

// CSS utilities for Tailwind classes
const css = {
  // Layout
  flex: 'flex',
  inlineFlex: 'inline-flex',
  block: 'block',
  inlineBlock: 'inline-block',
  hidden: 'hidden',
  grid: 'grid',
  
  // Positioning
  absolute: 'absolute',
  relative: 'relative',
  fixed: 'fixed',
  sticky: 'sticky',
  
  // Sizing
  w: (val) => `w-${val}`,
  h: (val) => `h-${val}`,
  wFull: 'w-full',
  hFull: 'h-full',
  
  // Spacing
  p: (val) => `p-${val}`,
  m: (val) => `m-${val}`,
  gap: (val) => `gap-${val}`,
  
  // Colors
  textBlack: 'text-black',
  textWhite: 'text-white',
  textGray: (shade = '600') => `text-gray-${shade}`,
  bgWhite: 'bg-white',
  bgGray: (shade = '100') => `bg-gray-${shade}`,
  bgPrimary: 'bg-blue-600',
  
  // Borders
  border: 'border',
  borderGray: 'border-gray-200',
  rounded: 'rounded',
  roundedLg: 'rounded-lg',
  
  // Shadows and effects
  shadow: 'shadow',
  shadowLg: 'shadow-lg',
  
  // Opacity
  opacity50: 'opacity-50',
  
  // Transform
  scale: (val) => `scale-${val}`,
  
  // Transitions
  transition: 'transition',
  duration200: 'duration-200',
  easeOut: 'ease-out',
};

/**
 * Merge class names intelligently
 */
function clsx(...classes) {
  return classes
    .flat()
    .filter(c => c && typeof c === 'string')
    .join(' ');
}

/**
 * Create element with classes
 */
function createElement(tag = 'div', { id = '', className = '', attrs = {}, html = '', text = '' } = {}) {
  const el = document.createElement(tag);
  
  if (id) el.id = id;
  if (className) el.className = className;
  if (text) el.textContent = text;
  if (html) el.innerHTML = html;
  
  Object.entries(attrs).forEach(([key, value]) => {
    if (value === true) {
      el.setAttribute(key, '');
    } else if (value !== false && value !== null) {
      el.setAttribute(key, value);
    }
  });
  
  return el;
}

/**
 * Add event listeners with automatic cleanup
 */
function addListener(el, event, handler, options = {}) {
  if (!el) return;
  el.addEventListener(event, handler, options);
  
  return () => el.removeEventListener(event, handler, options);
}

/**
 * Debounce function
 */
function debounce(func, wait = 300) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

/**
 * Throttle function
 */
function throttle(func, limit = 300) {
  let inThrottle;
  return function(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

/**
 * Get computed style value
 */
function getComputedValue(el, property) {
  return window.getComputedStyle(el).getPropertyValue(property);
}

/**
 * Toggle attribute on element
 */
function toggleAttr(el, attr, value = '') {
  if (el.hasAttribute(attr)) {
    el.removeAttribute(attr);
  } else {
    el.setAttribute(attr, value);
  }
}

/**
 * Focus trap utility
 */
function createFocusTrap(container) {
  const focusableElements = container.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];
  
  return {
    init() {
      container.addEventListener('keydown', (e) => {
        if (e.key !== 'Tab') return;
        
        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      });
    },
    destroy() {
      // Cleanup if needed
    }
  };
}

/**
 * Keyboard event utilities
 */
const keyboard = {
  isEnter: (e) => e.key === 'Enter',
  isEscape: (e) => e.key === 'Escape',
  isArrowUp: (e) => e.key === 'ArrowUp',
  isArrowDown: (e) => e.key === 'ArrowDown',
  isArrowLeft: (e) => e.key === 'ArrowLeft',
  isArrowRight: (e) => e.key === 'ArrowRight',
  isSpace: (e) => e.key === ' ',
  isTab: (e) => e.key === 'Tab',
};

/**
 * Animation utilities
 */
function animate(el, keyframes, options = {}) {
  return el.animate(keyframes, {
    duration: 300,
    easing: 'ease-out',
    fill: 'forwards',
    ...options
  });
}

/**
 * Storage utilities
 */
const storage = {
  set: (key, value) => localStorage.setItem(key, JSON.stringify(value)),
  get: (key) => {
    try {
      return JSON.parse(localStorage.getItem(key));
    } catch {
      return null;
    }
  },
  remove: (key) => localStorage.removeItem(key),
  clear: () => localStorage.clear(),
};

/**
 * UUID generator
 */
function generateId() {
  return 'id-' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
}

/**
 * Check if element is in viewport
 */
function isInViewport(el) {
  const rect = el.getBoundingClientRect();
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
}

/**
 * Create modal backdrop
 */
function createBackdrop(className = '') {
  const backdrop = createElement('div', {
    className: clsx(
      'fixed inset-0 bg-black bg-opacity-50 z-40 transition-opacity duration-200',
      className
    ),
    attrs: { 'data-backdrop': 'true' }
  });
  return backdrop;
}

/**
 * Escape HTML
 */
function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}

/**
 * Format currency
 */
function formatCurrency(amount, currency = null, locale = navigator.language) {
  // If currency string code provided, use Intl
  if (typeof currency === 'string' && currency.length === 3) {
    try {
      return new Intl.NumberFormat(locale, { style: 'currency', currency }).format(Number(amount));
    } catch (e) {
      return Number(amount).toFixed(2);
    }
  }

  // If no currency provided, use global window configuration
  if (!currency && typeof window !== 'undefined' && window.BUNORAA_CURRENCY) {
    currency = window.BUNORAA_CURRENCY;
  }

  if (!currency) return String(amount);

  const symbol = currency.symbol || '';
  const decimals = typeof currency.decimal_places === 'number' ? currency.decimal_places : 2;
  const thousand = currency.thousand_separator || ',';
  const dec_sep = currency.decimal_separator || '.';
  const symbol_pos = currency.symbol_position || 'before';

  const num = Number(amount);
  if (Number.isNaN(num)) return String(amount);

  const fixed = num.toFixed(decimals);
  const parts = fixed.split('.');
  let intPart = parts[0];
  const decPart = parts[1] || '';

  intPart = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, thousand);
  const formattedNumber = decimals > 0 ? intPart + dec_sep + decPart : intPart;

  return symbol_pos === 'before' ? (symbol + formattedNumber) : (formattedNumber + ' ' + symbol);
}

/**
 * Format date
 */
function formatDate(date, format = 'short') {
  const options = {
    short: { year: 'numeric', month: 'short', day: 'numeric' },
    long: { year: 'numeric', month: 'long', day: 'numeric' },
    time: { hour: '2-digit', minute: '2-digit' }
  };
  return new Date(date).toLocaleDateString('en-US', options[format] || options.short);
}

export {
  css,
  clsx,
  createElement,
  addListener,
  debounce,
  throttle,
  getComputedValue,
  toggleAttr,
  createFocusTrap,
  keyboard,
  animate,
  storage,
  generateId,
  isInViewport,
  createBackdrop,
  escapeHtml,
  formatCurrency,
  formatDate
};
