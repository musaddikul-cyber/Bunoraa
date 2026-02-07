/**
 * Progress Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Progress extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.value = options.value || 0; // 0-100
    this.max = options.max || 100;
    this.size = options.size || 'md'; // sm, md, lg
    this.variant = options.variant || 'default'; // default, success, warning, error
    this.showLabel = options.showLabel || false;
    this.className = options.className || '';
    this.animated = options.animated !== false;
  }

  create() {
    const sizeClasses = {
      sm: 'h-1',
      md: 'h-2',
      lg: 'h-4'
    };

    const variantClasses = {
      default: 'bg-blue-600',
      success: 'bg-green-600',
      warning: 'bg-yellow-600',
      error: 'bg-red-600'
    };

    const containerElement = super.create('div', {
      className: clsx('w-full bg-gray-200 rounded-full overflow-hidden', sizeClasses[this.size], this.className),
      attrs: { role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': this.max, 'aria-valuenow': this.value }
    });

    const percentage = (this.value / this.max) * 100;

    const barElement = document.createElement('div');
    barElement.className = clsx(
      variantClasses[this.variant],
      'h-full transition-all duration-500',
      this.animated ? 'animate-pulse' : ''
    );
    barElement.style.width = `${percentage}%`;

    containerElement.appendChild(barElement);

    if (this.showLabel) {
      const labelElement = document.createElement('span');
      labelElement.className = 'text-xs font-semibold text-gray-700 ml-2';
      labelElement.textContent = `${Math.round(percentage)}%`;
      containerElement.appendChild(labelElement);
    }

    this.barElement = barElement;
    return containerElement;
  }

  setValue(value) {
    this.value = Math.min(Math.max(value, 0), this.max);
    if (this.barElement) {
      const percentage = (this.value / this.max) * 100;
      this.barElement.style.width = `${percentage}%`;
      this.attr('aria-valuenow', this.value);
    }
  }

  increment(amount = 1) {
    this.setValue(this.value + amount);
  }

  decrement(amount = 1) {
    this.setValue(this.value - amount);
  }
}
