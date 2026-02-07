/**
 * Toast/Sonner Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Toast extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.message = options.message || '';
    this.type = options.type || 'default'; // default, success, error, warning, info
    // On mobile, shorten toast duration slightly for faster UX
    if (typeof options.duration !== 'undefined') {
      this.duration = options.duration;
    } else {
      this.duration = (typeof window !== 'undefined' && window.innerWidth < 640) ? 2500 : 3000;
    }
    this.position = options.position || 'top-right'; // top-left, top-right, bottom-left, bottom-right, top-center
    this.className = options.className || '';
    this.onClose = options.onClose || null;
  }

  destroy() {
    if (!this.element) return;
    const el = this.element;
    // Play exit animation, then cleanup
    el.classList.add(this.getExitAnimationClass());
    const onDone = () => {
      el.removeEventListener('animationend', onDone);
      // Remove and cleanup via BaseComponent
      super.destroy();
      // If container becomes empty, remove it
      const container = Toast.getContainer(this.position);
      if (container && container.childElementCount === 0 && container.parentNode) {
        container.parentNode.removeChild(container);
        if (Toast._containers) {
          delete Toast._containers[this.position || 'top-right'];
        }
      }
    };
    el.addEventListener('animationend', onDone);
    // Fallback in case animationend doesn’t fire
    setTimeout(onDone, 320);
  }

  static getContainer(position) {
    const key = position || 'top-right';
    if (!this._containers) this._containers = {};
    if (this._containers[key] && document.body.contains(this._containers[key])) {
      return this._containers[key];
    }
    const container = createElement('div', {
      className: clsx(
        'fixed z-50 p-2 flex flex-col gap-2 pointer-events-none',
        this.getPositionClassesForContainer(key)
      )
    });
    document.body.appendChild(container);
    this._containers[key] = container;
    return container;
  }

  static getPositionClassesForContainer(position) {
    switch (position) {
      case 'top-left':
        return 'top-4 left-4 items-start';
      case 'top-right':
        return 'top-4 right-4 items-end';
      case 'bottom-left':
        return 'bottom-4 left-4 items-start';
      case 'bottom-right':
        return 'bottom-4 right-4 items-end';
      case 'top-center':
        return 'top-4 left-1/2 -translate-x-1/2 items-center transform';
      default:
        return 'top-4 right-4 items-end';
    }
  }

  create() {
    const containerElement = Toast.getContainer(this.position);

    const toastElement = createElement('div', {
      className: clsx(
        // Allow wrapping on small screens, limit max width for mobile, keep compact padding
        'rounded-lg shadow-lg p-2.5 flex items-center gap-2 min-w-0 max-w-[90vw] sm:max-w-sm bg-opacity-95',
        this.getEnterAnimationClass(),
        this.getTypeClasses(),
        this.className
      )
    });

    // Icon
    const iconElement = createElement('span', {
      className: 'text-base flex-shrink-0',
      text: this.getIcon()
    });
    toastElement.appendChild(iconElement);

    // Message
    const messageElement = createElement('span', {
      text: this.message,
      // Slightly smaller text on mobile, readable on larger screens
      className: 'flex-1 text-sm sm:text-base'
    });
    toastElement.appendChild(messageElement);

    // Add ARIA for assistive technologies
    toastElement.setAttribute('role', this.type === 'error' ? 'alert' : 'status');
    toastElement.setAttribute('aria-live', this.type === 'error' ? 'assertive' : 'polite');

    // Close button
    const closeBtn = createElement('button', {
      className: 'text-base hover:opacity-70 transition-opacity flex-shrink-0',
      text: '×'
    });

    closeBtn.setAttribute('aria-label', 'Dismiss notification');
    closeBtn.addEventListener('click', () => {
      this.destroy();
    });

    toastElement.appendChild(closeBtn);

    // Set element and append to shared container
    this.element = toastElement;
    containerElement.appendChild(this.element);

    // Enforce max visible toasts (3). Remove oldest if exceeded.
    while (containerElement.children.length > 3) {
      containerElement.removeChild(containerElement.firstElementChild);
    }

    // Auto close
    if (this.duration > 0) {
      setTimeout(() => {
        this.destroy();
      }, this.duration);
    }

    return this.element;
  }

  getEnterAnimationClass() {
    const pos = this.position || 'top-right';
    if (pos === 'top-right' || pos === 'bottom-right') return 'animate-slide-in-right transition-all duration-300 pointer-events-auto';
    if (pos === 'top-left' || pos === 'bottom-left') return 'animate-slide-in-left transition-all duration-300 pointer-events-auto';
    return 'animate-slide-in-top transition-all duration-300 pointer-events-auto';
  }

  getExitAnimationClass() {
    const pos = this.position || 'top-right';
    if (pos === 'top-right' || pos === 'bottom-right') return 'animate-slide-out-right';
    if (pos === 'top-left' || pos === 'bottom-left') return 'animate-slide-out-left';
    return 'animate-slide-out-top';
  }

  getPositionClasses() {
    const positions = {
      'top-left': 'top-4 left-4',
      'top-right': 'top-4 right-4',
      'bottom-left': 'bottom-4 left-4',
      'bottom-right': 'bottom-4 right-4',
      'top-center': 'top-4 left-1/2 -translate-x-1/2 transform'
    };
    return positions[this.position] || positions['bottom-right'];
  }

  getTypeClasses() {
    const types = {
      default: 'bg-gray-900 text-white',
      success: 'bg-green-600 text-white',
      error: 'bg-red-600 text-white',
      warning: 'bg-yellow-600 text-white',
      info: 'bg-blue-600 text-white'
    };
    return types[this.type] || types.default;
  }

  getIcon() {
    const icons = {
      default: 'ℹ',
      success: '✓',
      error: '✕',
      warning: '⚠',
      info: 'ℹ'
    };
    return icons[this.type] || icons.default;
  }

  static show(message, options = {}) {
    const toast = new Toast({ message, ...options });
    const element = toast.create();
    return toast;
  }

  // Convenience helpers
  static success(message, options = {}) { return this.show(message, { ...options, type: 'success' }); }
  static error(message, options = {}) { return this.show(message, { ...options, type: 'error', position: options.position || 'top-right' }); }
  static info(message, options = {}) { return this.show(message, { ...options, type: 'info' }); }
  static warning(message, options = {}) { return this.show(message, { ...options, type: 'warning' }); }
}

// Add CSS animation
if (!document.querySelector('style[data-toast]')) {
  const style = document.createElement('style');
  style.setAttribute('data-toast', 'true');
  style.textContent = `
    @keyframes slideInTop {
      from { transform: translateY(-12px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    .animate-slide-in-top { animation: slideInTop 0.25s ease-out; }

    @keyframes slideOutTop {
      from { transform: translateY(0); opacity: 1; }
      to { transform: translateY(-12px); opacity: 0; }
    }
    .animate-slide-out-top { animation: slideOutTop 0.2s ease-in forwards; }

    @keyframes slideInRight {
      from { transform: translateX(16px); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
    .animate-slide-in-right { animation: slideInRight 0.25s ease-out; }

    @keyframes slideOutRight {
      from { transform: translateX(0); opacity: 1; }
      to { transform: translateX(16px); opacity: 0; }
    }
    .animate-slide-out-right { animation: slideOutRight 0.2s ease-in forwards; }

    @keyframes slideInLeft {
      from { transform: translateX(-16px); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
    .animate-slide-in-left { animation: slideInLeft 0.25s ease-out; }

    @keyframes slideOutLeft {
      from { transform: translateX(0); opacity: 1; }
      to { transform: translateX(-16px); opacity: 0; }
    }
    .animate-slide-out-left { animation: slideOutLeft 0.2s ease-in forwards; }
  `;
  document.head.appendChild(style);
}
