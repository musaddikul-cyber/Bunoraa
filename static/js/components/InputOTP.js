/**
 * Input OTP Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class InputOTP extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.length = options.length || 6;
    this.value = options.value || '';
    this.className = options.className || '';
    this.onChange = options.onChange || null;
    this.onComplete = options.onComplete || null;
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx('flex gap-2', this.className)
    });

    this.inputs = [];

    for (let i = 0; i < this.length; i++) {
      const input = document.createElement('input');
      input.type = 'text';
      input.maxLength = '1';
      input.inputMode = 'numeric';
      input.className = clsx(
        'w-12 h-12 text-center border-2 border-gray-300 rounded-lg font-semibold text-lg',
        'focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-200',
        'transition-colors duration-200'
      );

      if (this.value && this.value[i]) {
        input.value = this.value[i];
      }

      input.addEventListener('input', (e) => {
        const value = e.target.value;
        if (!/^\d*$/.test(value)) {
          e.target.value = '';
          return;
        }

        if (value && i < this.length - 1) {
          this.inputs[i + 1].focus();
        }

        this.updateValue();
      });

      input.addEventListener('keydown', (e) => {
        if (e.key === 'Backspace') {
          if (!input.value && i > 0) {
            this.inputs[i - 1].focus();
          }
        } else if (e.key === 'ArrowLeft' && i > 0) {
          this.inputs[i - 1].focus();
        } else if (e.key === 'ArrowRight' && i < this.length - 1) {
          this.inputs[i + 1].focus();
        }
      });

      this.inputs.push(input);
      containerElement.appendChild(input);
    }

    return containerElement;
  }

  updateValue() {
    this.value = this.inputs.map(input => input.value).join('');
    
    if (this.onChange) {
      this.onChange(this.value);
    }

    if (this.value.length === this.length && this.onComplete) {
      this.onComplete(this.value);
    }
  }

  getValue() {
    return this.value;
  }

  setValue(value) {
    this.value = value;
    for (let i = 0; i < this.length; i++) {
      this.inputs[i].value = value[i] || '';
    }
  }

  clear() {
    this.inputs.forEach(input => {
      input.value = '';
    });
    this.value = '';
  }

  focus() {
    if (this.inputs.length > 0) {
      this.inputs[0].focus();
    }
  }
}
