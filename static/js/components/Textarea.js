/**
 * Textarea Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Textarea extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.placeholder = options.placeholder || '';
    this.value = options.value || '';
    this.name = options.name || '';
    this.disabled = options.disabled || false;
    this.required = options.required || false;
    this.rows = options.rows || 4;
    this.className = options.className || '';
    this.onChange = options.onChange || null;
    this.maxLength = options.maxLength || null;
    this.resizable = options.resizable !== false; // true by default
  }

  create() {
    const baseClasses = 'w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900 placeholder-gray-500 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed font-sans';
    
    const resizeClass = this.resizable ? 'resize-vertical' : 'resize-none';

    const attrs = {
      placeholder: this.placeholder,
      name: this.name,
      disabled: this.disabled ? '' : null,
      required: this.required ? '' : null,
      rows: this.rows
    };

    if (this.maxLength) {
      attrs.maxlength = this.maxLength;
    }

    const textareaElement = super.create('textarea', {
      className: clsx(baseClasses, resizeClass, this.className),
      attrs
    });

    textareaElement.textContent = this.value;

    if (this.onChange) {
      this.on('change', this.onChange);
      this.on('input', this.onChange);
    }

    return textareaElement;
  }

  getValue() {
    return this.element?.value || '';
  }

  setValue(value) {
    this.value = value;
    if (this.element) {
      this.element.value = value;
    }
  }

  setDisabled(disabled) {
    this.disabled = disabled;
    this.attr('disabled', disabled ? '' : null);
  }

  clear() {
    this.setValue('');
  }

  setRows(rows) {
    this.rows = rows;
    if (this.element) {
      this.element.rows = rows;
    }
  }
}
