/**
 * Card Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Card extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.title = options.title || '';
    this.subtitle = options.subtitle || '';
    this.content = options.content || '';
    this.footer = options.footer || '';
    this.className = options.className || '';
    this.hoverable = options.hoverable !== false;
  }

  create() {
    const baseClasses = 'bg-white rounded-lg border border-gray-200 overflow-hidden';
    const hoverClass = this.hoverable ? 'hover:shadow-lg transition-shadow duration-300' : '';

    const cardElement = super.create('div', {
      className: clsx(baseClasses, hoverClass, this.className)
    });

    // Header
    if (this.title) {
      const headerElement = createElement('div', {
        className: 'px-6 py-4 border-b border-gray-200 bg-gray-50'
      });

      let html = `<h3 class="text-lg font-semibold text-gray-900">${this.title}</h3>`;
      if (this.subtitle) {
        html += `<p class="text-sm text-gray-600 mt-1">${this.subtitle}</p>`;
      }

      headerElement.innerHTML = html;
      cardElement.appendChild(headerElement);
    }

    // Content
    if (this.content) {
      const contentElement = createElement('div', {
        className: 'px-6 py-4',
        html: this.content
      });
      cardElement.appendChild(contentElement);
    }

    // Footer
    if (this.footer) {
      const footerElement = createElement('div', {
        className: 'px-6 py-4 border-t border-gray-200 bg-gray-50',
        html: this.footer
      });
      cardElement.appendChild(footerElement);
    }

    return cardElement;
  }

  setContent(content) {
    this.content = content;
    if (this.element) {
      const contentDiv = this.element.querySelector('.px-6.py-4:not(.border-b):not(.border-t)');
      if (contentDiv) {
        contentDiv.innerHTML = content;
      }
    }
  }

  addContent(element) {
    if (this.element) {
      const contentDiv = this.element.querySelector('.px-6.py-4:not(.border-b):not(.border-t)');
      if (contentDiv) {
        contentDiv.appendChild(element instanceof BaseComponent ? element.element : element);
      }
    }
  }
}
