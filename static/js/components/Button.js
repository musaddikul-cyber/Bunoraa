/**
 * Button Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Button extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.label = options.label || 'Button';
    this.variant = options.variant || 'default'; // default, primary, secondary, destructive, outline, ghost
    this.size = options.size || 'md'; // sm, md, lg
    this.disabled = options.disabled || false;
    this.onClick = options.onClick || null;
    this.className = options.className || '';
  }

  create() {
    const baseClasses = 'px-4 py-2 font-medium rounded transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-offset-2';
    
    const variantClasses = {
      default: 'bg-gray-200 text-gray-900 hover:bg-gray-300',
      primary: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500',
      secondary: 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500',
      destructive: 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500',
      outline: 'border-2 border-gray-300 text-gray-900 hover:bg-gray-50',
      ghost: 'text-gray-900 hover:bg-gray-100'
    };

    const sizeClasses = {
      sm: 'px-2 py-1 text-sm',
      md: 'px-4 py-2 text-base',
      lg: 'px-6 py-3 text-lg'
    };

    const buttonElement = super.create('button', {
      className: clsx(
        baseClasses,
        variantClasses[this.variant],
        sizeClasses[this.size],
        this.className
      ),
      attrs: { disabled: this.disabled }
    });

    buttonElement.textContent = this.label;

    if (this.onClick) {
      this.on('click', this.onClick);
    }

    return buttonElement;
  }

  setLabel(label) {
    this.label = label;
    if (this.element) {
      this.element.textContent = label;
    }
  }

  setDisabled(disabled) {
    this.disabled = disabled;
    this.attr('disabled', disabled ? '' : null);
  }

  setLoading(loading) {
    this.setDisabled(loading);
    if (loading) {
      this.html('<span class="flex items-center gap-2"><span class="inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></span>Loading...</span>');
    } else {
      this.text(this.label);
    }
  }
}
