/**
 * Kbd Component (Keyboard Key)
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Kbd extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.label = options.label || 'K';
    this.className = options.className || '';
  }

  create() {
    const baseClasses = 'px-2 py-1 bg-gray-100 border border-gray-300 rounded text-xs font-semibold text-gray-900 inline-block font-mono';

    const kbdElement = super.create('kbd', {
      className: clsx(baseClasses, this.className)
    });

    kbdElement.textContent = this.label;
    return kbdElement;
  }

  setLabel(label) {
    this.label = label;
    if (this.element) {
      this.element.textContent = label;
    }
  }
}
