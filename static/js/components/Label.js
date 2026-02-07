/**
 * Label Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Label extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.text = options.text || '';
    this.htmlFor = options.htmlFor || '';
    this.required = options.required || false;
    this.className = options.className || '';
  }

  create() {
    const baseClasses = 'block text-sm font-medium text-gray-700 mb-1';

    const labelElement = super.create('label', {
      className: clsx(baseClasses, this.className),
      attrs: { for: this.htmlFor }
    });

    let html = this.text;
    if (this.required) {
      html += ' <span class="text-red-500 ml-1">*</span>';
    }

    labelElement.innerHTML = html;
    return labelElement;
  }

  setText(text) {
    this.text = text;
    if (this.element) {
      let html = text;
      if (this.required) {
        html += ' <span class="text-red-500 ml-1">*</span>';
      }
      this.element.innerHTML = html;
    }
  }

  setRequired(required) {
    this.required = required;
    if (this.element) {
      const requiredSpan = this.element.querySelector('[class*="text-red"]');
      if (required && !requiredSpan) {
        this.element.innerHTML += ' <span class="text-red-500 ml-1">*</span>';
      } else if (!required && requiredSpan) {
        requiredSpan.remove();
      }
    }
  }
}
