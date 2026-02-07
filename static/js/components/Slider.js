/**
 * Slider Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Slider extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.min = options.min || 0;
    this.max = options.max || 100;
    this.value = options.value || 50;
    this.step = options.step || 1;
    this.disabled = options.disabled || false;
    this.className = options.className || '';
    this.onChange = options.onChange || null;
    this.size = options.size || 'md';
  }

  create() {
    const baseClasses = 'w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed';

    const sliderElement = super.create('input', {
      className: clsx(baseClasses, 'slider-input', this.className),
      attrs: {
        type: 'range',
        min: this.min,
        max: this.max,
        value: this.value,
        step: this.step,
        disabled: this.disabled ? '' : null
      }
    });

    // Add custom slider styling
    const style = document.createElement('style');
    style.textContent = `
      .slider-input {
        -webkit-appearance: none;
        appearance: none;
      }
      
      .slider-input::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #2563eb;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transition: background-color 0.2s;
      }
      
      .slider-input::-webkit-slider-thumb:hover {
        background: #1d4ed8;
      }
      
      .slider-input::-moz-range-thumb {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #2563eb;
        cursor: pointer;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transition: background-color 0.2s;
      }
      
      .slider-input::-moz-range-thumb:hover {
        background: #1d4ed8;
      }
      
      .slider-input:disabled::-webkit-slider-thumb {
        background: #d1d5db;
        cursor: not-allowed;
      }
      
      .slider-input:disabled::-moz-range-thumb {
        background: #d1d5db;
        cursor: not-allowed;
      }
    `;
    document.head.appendChild(style);

    if (this.onChange) {
      this.on('input', this.onChange);
      this.on('change', this.onChange);
    }

    return sliderElement;
  }

  getValue() {
    return parseInt(this.element?.value || this.value);
  }

  setValue(value) {
    this.value = Math.min(Math.max(value, this.min), this.max);
    if (this.element) {
      this.element.value = this.value;
    }
  }

  setDisabled(disabled) {
    this.disabled = disabled;
    this.attr('disabled', disabled ? '' : null);
  }
}
