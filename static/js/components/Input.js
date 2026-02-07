/**
 * Input Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Input extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.type = options.type || 'text';
    this.placeholder = options.placeholder || '';
    this.value = options.value || '';
    this.name = options.name || '';
    this.disabled = options.disabled || false;
    this.required = options.required || false;
    this.className = options.className || '';
    this.onChange = options.onChange || null;
  }

  create() {
    const baseClasses = 'w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900 placeholder-gray-500 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed';

    const inputElement = super.create('input', {
      className: clsx(baseClasses, this.className),
      attrs: {
        type: this.type,
        placeholder: this.placeholder,
        value: this.value,
        name: this.name,
        disabled: this.disabled ? '' : null,
        required: this.required ? '' : null
      }
    });

    if (this.onChange) {
      this.on('change', this.onChange);
      this.on('input', this.onChange);
    }

    return inputElement;
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

  setPlaceholder(placeholder) {
    this.placeholder = placeholder;
    if (this.element) {
      this.element.placeholder = placeholder;
    }
  }

  focus() {
    super.focus();
  }

  clear() {
    this.setValue('');
  }
}
