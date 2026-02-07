/**
 * Native Select Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class NativeSelect extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.items = options.items || []; // Array of {label, value}
    this.selected = options.selected || '';
    this.placeholder = options.placeholder || 'Select...';
    this.disabled = options.disabled || false;
    this.required = options.required || false;
    this.name = options.name || '';
    this.className = options.className || '';
    this.onChange = options.onChange || null;
  }

  create() {
    const baseClasses = 'w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed bg-white appearance-none cursor-pointer';

    const selectElement = super.create('select', {
      className: clsx(baseClasses, this.className),
      attrs: {
        disabled: this.disabled ? '' : null,
        required: this.required ? '' : null,
        ...(this.name && { name: this.name })
      }
    });

    // Add placeholder option
    if (this.placeholder) {
      const placeholderOption = document.createElement('option');
      placeholderOption.value = '';
      placeholderOption.textContent = this.placeholder;
      placeholderOption.disabled = true;
      selectElement.appendChild(placeholderOption);
    }

    // Add items
    this.items.forEach(item => {
      const optionElement = document.createElement('option');
      optionElement.value = item.value;
      optionElement.textContent = item.label;
      if (item.value === this.selected) {
        optionElement.selected = true;
      }
      selectElement.appendChild(optionElement);
    });

    if (this.onChange) {
      this.on('change', this.onChange);
    }

    return selectElement;
  }

  getValue() {
    return this.element?.value || '';
  }

  setValue(value) {
    this.selected = value;
    if (this.element) {
      this.element.value = value;
    }
  }

  setDisabled(disabled) {
    this.disabled = disabled;
    this.attr('disabled', disabled ? '' : null);
  }

  addItem(label, value) {
    if (this.element) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      this.element.appendChild(option);
    }
  }

  removeItem(value) {
    if (this.element) {
      const option = this.element.querySelector(`option[value="${value}"]`);
      if (option) {
        option.remove();
      }
    }
  }
}
