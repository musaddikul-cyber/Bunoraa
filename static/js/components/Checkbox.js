/**
 * Checkbox Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Checkbox extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.label = options.label || '';
    this.checked = options.checked || false;
    this.disabled = options.disabled || false;
    this.required = options.required || false;
    this.name = options.name || '';
    this.className = options.className || '';
    this.onChange = options.onChange || null;
  }

  create() {
    const baseClasses = 'flex items-center gap-2';

    const containerElement = super.create('div', {
      className: clsx(baseClasses, this.className)
    });

    const checkboxClasses = 'w-5 h-5 border-2 border-gray-300 rounded cursor-pointer transition-colors duration-200 checked:bg-blue-600 checked:border-blue-600 disabled:opacity-50 disabled:cursor-not-allowed';

    const inputElement = document.createElement('input');
    inputElement.type = 'checkbox';
    inputElement.className = checkboxClasses;
    inputElement.checked = this.checked;
    inputElement.disabled = this.disabled;
    inputElement.required = this.required;
    if (this.name) inputElement.name = this.name;

    containerElement.appendChild(inputElement);

    if (this.label) {
      const labelElement = document.createElement('label');
      labelElement.className = 'cursor-pointer select-none';
      labelElement.textContent = this.label;
      containerElement.appendChild(labelElement);

      labelElement.addEventListener('click', () => {
        inputElement.click();
      });
    }

    if (this.onChange) {
      inputElement.addEventListener('change', this.onChange);
    }

    this.inputElement = inputElement;
    return containerElement;
  }

  isChecked() {
    return this.inputElement?.checked || false;
  }

  setChecked(checked) {
    this.checked = checked;
    if (this.inputElement) {
      this.inputElement.checked = checked;
    }
  }

  setDisabled(disabled) {
    this.disabled = disabled;
    if (this.inputElement) {
      this.inputElement.disabled = disabled;
    }
  }

  toggle() {
    this.setChecked(!this.isChecked());
  }
}
