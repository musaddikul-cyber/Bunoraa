/**
 * Spinner Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Spinner extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.size = options.size || 'md'; // sm, md, lg, xl
    this.color = options.color || 'blue'; // blue, red, green, gray
    this.className = options.className || '';
  }

  create() {
    const sizeClasses = {
      sm: 'w-4 h-4',
      md: 'w-8 h-8',
      lg: 'w-12 h-12',
      xl: 'w-16 h-16'
    };

    const colorClasses = {
      blue: 'border-blue-400 border-t-blue-600',
      red: 'border-red-400 border-t-red-600',
      green: 'border-green-400 border-t-green-600',
      gray: 'border-gray-400 border-t-gray-600'
    };

    const spinnerElement = super.create('div', {
      className: clsx(
        'border-4 rounded-full animate-spin',
        sizeClasses[this.size],
        colorClasses[this.color],
        this.className
      ),
      attrs: { role: 'status', 'aria-label': 'Loading' }
    });

    return spinnerElement;
  }
}
