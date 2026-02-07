/**
 * Badge Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Badge extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.label = options.label || 'Badge';
    this.variant = options.variant || 'default'; // default, primary, success, warning, destructive, outline
    this.size = options.size || 'md'; // sm, md, lg
    this.className = options.className || '';
  }

  create() {
    const baseClasses = 'inline-flex items-center rounded font-semibold whitespace-nowrap';
    
    const variantClasses = {
      default: 'bg-gray-100 text-gray-800',
      primary: 'bg-blue-100 text-blue-800',
      success: 'bg-green-100 text-green-800',
      warning: 'bg-yellow-100 text-yellow-800',
      destructive: 'bg-red-100 text-red-800',
      outline: 'border border-gray-300 text-gray-700'
    };

    const sizeClasses = {
      sm: 'px-2 py-1 text-xs',
      md: 'px-3 py-1 text-sm',
      lg: 'px-4 py-2 text-base'
    };

    const badgeElement = super.create('span', {
      className: clsx(
        baseClasses,
        variantClasses[this.variant],
        sizeClasses[this.size],
        this.className
      )
    });

    badgeElement.textContent = this.label;
    return badgeElement;
  }

  setLabel(label) {
    this.label = label;
    if (this.element) {
      this.element.textContent = label;
    }
  }
}
