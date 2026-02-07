/**
 * Breadcrumb Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Breadcrumb extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.items = options.items || []; // Array of {label, href}
    this.className = options.className || '';
  }

  create() {
    const baseClasses = 'flex items-center gap-2';

    const breadcrumbElement = super.create('nav', {
      className: clsx(baseClasses, this.className),
      attrs: { 'aria-label': 'Breadcrumb' }
    });

    this.items.forEach((item, index) => {
      if (index > 0) {
        const separator = createElement('span', {
          className: 'text-gray-400 mx-1',
          text: '/'
        });
        breadcrumbElement.appendChild(separator);
      }

      if (index === this.items.length - 1) {
        const span = createElement('span', {
          className: 'text-gray-700 font-medium',
          text: item.label,
          attrs: { 'aria-current': 'page' }
        });
        breadcrumbElement.appendChild(span);
      } else {
        const link = createElement('a', {
          className: 'text-blue-600 hover:text-blue-800 hover:underline transition-colors duration-200',
          text: item.label,
          attrs: { href: item.href || '#' }
        });
        breadcrumbElement.appendChild(link);
      }
    });

    return breadcrumbElement;
  }

  addItem(label, href = '#') {
    if (this.element) {
      if (this.element.children.length > 0) {
        const separator = createElement('span', {
          className: 'text-gray-400 mx-1',
          text: '/'
        });
        this.element.appendChild(separator);
      }

      const link = createElement('a', {
        className: 'text-blue-600 hover:text-blue-800 hover:underline transition-colors duration-200',
        text: label,
        attrs: { href }
      });
      this.element.appendChild(link);
      this.items.push({ label, href });
    }
  }
}
