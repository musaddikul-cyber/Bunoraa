/**
 * Date Picker Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class DatePicker extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.value = options.value || '';
    this.placeholder = options.placeholder || 'Select date...';
    this.format = options.format || 'yyyy-mm-dd';
    this.className = options.className || '';
    this.onChange = options.onChange || null;
    this.open = false;
  }

  create() {
    const containerElement = super.create('div', {
      className: 'relative w-full'
    });

    // Input
    const inputElement = document.createElement('input');
    inputElement.type = 'text';
    inputElement.placeholder = this.placeholder;
    inputElement.value = this.value;
    inputElement.className = clsx(
      'w-full px-3 py-2 border border-gray-300 rounded-md',
      'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent'
    );

    inputElement.addEventListener('click', () => {
      this.openPicker();
    });

    inputElement.addEventListener('change', (e) => {
      this.value = e.target.value;
      if (this.onChange) {
        this.onChange(this.value);
      }
    });

    containerElement.appendChild(inputElement);

    // Use native date input for simplicity
    const nativeInput = document.createElement('input');
    nativeInput.type = 'date';
    nativeInput.style.display = 'none';
    nativeInput.value = this.value;

    nativeInput.addEventListener('change', (e) => {
      const date = new Date(e.target.value);
      this.value = this.formatDate(date);
      inputElement.value = this.value;
      
      if (this.onChange) {
        this.onChange(this.value);
      }
    });

    containerElement.appendChild(nativeInput);

    this.inputElement = inputElement;
    this.nativeInput = nativeInput;

    return containerElement;
  }

  openPicker() {
    this.nativeInput.click();
  }

  formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    
    if (this.format === 'dd/mm/yyyy') {
      return `${day}/${month}/${year}`;
    }
    return `${year}-${month}-${day}`;
  }

  getValue() {
    return this.value;
  }

  setValue(value) {
    this.value = value;
    if (this.inputElement) {
      this.inputElement.value = value;
    }
  }
}
