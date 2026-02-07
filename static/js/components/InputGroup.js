/**
 * Input Group Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class InputGroup extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.prefix = options.prefix || null;
    this.suffix = options.suffix || null;
    this.input = options.input || null;
    this.className = options.className || '';
  }

  create() {
    const baseClasses = 'flex items-center border border-gray-300 rounded-md overflow-hidden focus-within:ring-2 focus-within:ring-blue-500';

    const groupElement = super.create('div', {
      className: clsx(baseClasses, this.className)
    });

    if (this.prefix) {
      const prefixEl = createElement('div', {
        className: 'px-3 py-2 bg-gray-50 text-gray-700 font-medium text-sm',
        html: this.prefix
      });
      groupElement.appendChild(prefixEl);
    }

    if (this.input) {
      const inputEl = this.input.element || this.input.create();
      inputEl.classList.remove('border', 'focus:ring-2', 'focus:ring-blue-500');
      groupElement.appendChild(inputEl);
    }

    if (this.suffix) {
      const suffixEl = createElement('div', {
        className: 'px-3 py-2 bg-gray-50 text-gray-700 font-medium text-sm',
        html: this.suffix
      });
      groupElement.appendChild(suffixEl);
    }

    return groupElement;
  }
}
