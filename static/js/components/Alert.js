/**
 * Alert Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Alert extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.title = options.title || '';
    this.message = options.message || '';
    this.type = options.type || 'default'; // default, success, warning, error, info
    this.icon = options.icon || null;
    this.closeable = options.closeable || false;
    this.className = options.className || '';
  }

  create() {
    const typeConfig = {
      default: {
        bg: 'bg-blue-50',
        border: 'border-blue-200',
        title: 'text-blue-900',
        message: 'text-blue-800',
        icon: 'ⓘ'
      },
      success: {
        bg: 'bg-green-50',
        border: 'border-green-200',
        title: 'text-green-900',
        message: 'text-green-800',
        icon: '✓'
      },
      warning: {
        bg: 'bg-yellow-50',
        border: 'border-yellow-200',
        title: 'text-yellow-900',
        message: 'text-yellow-800',
        icon: '⚠'
      },
      error: {
        bg: 'bg-red-50',
        border: 'border-red-200',
        title: 'text-red-900',
        message: 'text-red-800',
        icon: '✕'
      },
      info: {
        bg: 'bg-cyan-50',
        border: 'border-cyan-200',
        title: 'text-cyan-900',
        message: 'text-cyan-800',
        icon: 'ℹ'
      }
    };

    const config = typeConfig[this.type] || typeConfig.default;
    const baseClasses = 'p-4 rounded-lg border-2';

    const alertElement = super.create('div', {
      className: clsx(baseClasses, config.bg, config.border, this.className),
      attrs: { role: 'alert' }
    });

    let html = '';
    
    // Icon and title
    if (this.title || this.icon) {
      html += `<div class="flex items-center gap-3 mb-2">`;
      if (this.icon || true) {
        html += `<span class="text-xl font-bold ${config.title}">${this.icon || config.icon}</span>`;
      }
      if (this.title) {
        html += `<h4 class="font-semibold ${config.title}">${this.title}</h4>`;
      }
      html += `</div>`;
    }

    // Message
    if (this.message) {
      html += `<p class="${config.message}">${this.message}</p>`;
    }

    // Close button
    if (this.closeable) {
      html += `<button class="absolute top-4 right-4 text-gray-400 hover:text-gray-600" aria-label="Close alert">×</button>`;
    }

    alertElement.innerHTML = html;

    if (this.closeable) {
      const closeBtn = alertElement.querySelector('button');
      closeBtn?.addEventListener('click', () => {
        this.destroy();
      });
    }

    return alertElement;
  }

  setMessage(message) {
    this.message = message;
    if (this.element) {
      const msgEl = this.element.querySelector('p');
      if (msgEl) {
        msgEl.textContent = message;
      }
    }
  }
}
