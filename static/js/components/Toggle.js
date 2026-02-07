/**
 * Toggle Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Toggle extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.label = options.label || '';
    this.pressed = options.pressed || false;
    this.disabled = options.disabled || false;
    this.variant = options.variant || 'default'; // default, outline
    this.size = options.size || 'md'; // sm, md, lg
    this.className = options.className || '';
    this.onChange = options.onChange || null;
  }

  create() {
    const baseClasses = 'px-4 py-2 font-medium rounded transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';
    
    const variantClasses = {
      default: this.pressed 
        ? 'bg-gray-900 text-white' 
        : 'bg-gray-100 text-gray-900 hover:bg-gray-200',
      outline: this.pressed
        ? 'border-2 border-gray-900 bg-gray-900 text-white'
        : 'border-2 border-gray-300 text-gray-900 hover:bg-gray-50'
    };

    const sizeClasses = {
      sm: 'px-2 py-1 text-sm',
      md: 'px-4 py-2 text-base',
      lg: 'px-6 py-3 text-lg'
    };

    const toggleElement = super.create('button', {
      className: clsx(
        baseClasses,
        variantClasses[this.variant],
        sizeClasses[this.size],
        this.className
      ),
      attrs: { 
        'aria-pressed': this.pressed,
        disabled: this.disabled ? '' : null
      }
    });

    toggleElement.textContent = this.label;

    this.on('click', () => {
      this.toggle();
    });

    return toggleElement;
  }

  isPressed() {
    return this.pressed;
  }

  setPressed(pressed) {
    this.pressed = pressed;
    if (this.element) {
      this.attr('aria-pressed', pressed);
      this.toggleClass('bg-gray-900 text-white', pressed);
      if (this.onChange) {
        this.onChange(pressed);
      }
    }
  }

  toggle() {
    this.setPressed(!this.pressed);
  }

  setDisabled(disabled) {
    this.disabled = disabled;
    this.attr('disabled', disabled ? '' : null);
  }
}
