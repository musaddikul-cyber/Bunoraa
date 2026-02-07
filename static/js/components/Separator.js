/**
 * Separator Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Separator extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.orientation = options.orientation || 'horizontal'; // horizontal, vertical
    this.className = options.className || '';
  }

  create() {
    const baseClasses = 'bg-gray-200';
    
    const orientationClasses = {
      horizontal: 'w-full h-px',
      vertical: 'h-full w-px'
    };

    const separatorElement = super.create('div', {
      className: clsx(
        baseClasses,
        orientationClasses[this.orientation],
        this.className
      ),
      attrs: { role: 'separator', 'aria-orientation': this.orientation }
    });

    return separatorElement;
  }
}
