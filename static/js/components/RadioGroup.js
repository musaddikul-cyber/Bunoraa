/**
 * Radio Group Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class RadioGroup extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.items = options.items || []; // Array of {label, value}
    this.selected = options.selected || null;
    this.name = options.name || 'radio-group';
    this.disabled = options.disabled || false;
    this.className = options.className || '';
    this.onChange = options.onChange || null;
    this.orientation = options.orientation || 'vertical'; // vertical, horizontal
  }

  create() {
    const baseClasses = this.orientation === 'horizontal' 
      ? 'flex items-center gap-6' 
      : 'flex flex-col gap-3';

    const groupElement = super.create('div', {
      className: clsx(baseClasses, this.className),
      attrs: { role: 'radiogroup' }
    });

    this.radioInputs = [];

    this.items.forEach((item, index) => {
      const containerClasses = 'flex items-center gap-2';
      const container = createElement('div', { className: containerClasses });

      const radioId = `${this.name}-${index}`;
      const inputClasses = 'w-4 h-4 border-2 border-gray-300 rounded-full cursor-pointer transition-colors duration-200 checked:bg-blue-600 checked:border-blue-600 disabled:opacity-50 disabled:cursor-not-allowed';

      const radioInput = document.createElement('input');
      radioInput.type = 'radio';
      radioInput.id = radioId;
      radioInput.name = this.name;
      radioInput.value = item.value;
      radioInput.className = inputClasses;
      radioInput.checked = item.value === this.selected;
      radioInput.disabled = this.disabled;

      container.appendChild(radioInput);

      const labelElement = createElement('label', {
        attrs: { for: radioId },
        text: item.label,
        className: 'cursor-pointer select-none'
      });

      container.appendChild(labelElement);
      groupElement.appendChild(container);

      radioInput.addEventListener('change', () => {
        this.selected = item.value;
        if (this.onChange) {
          this.onChange(item.value);
        }
      });

      this.radioInputs.push(radioInput);
    });

    return groupElement;
  }

  getValue() {
    return this.selected;
  }

  setValue(value) {
    this.selected = value;
    this.radioInputs.forEach(input => {
      input.checked = input.value === value;
    });
  }

  setDisabled(disabled) {
    this.disabled = disabled;
    this.radioInputs.forEach(input => {
      input.disabled = disabled;
    });
  }
}
