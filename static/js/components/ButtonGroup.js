/**
 * Button Group Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class ButtonGroup extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.buttons = options.buttons || []; // Array of button objects
    this.orientation = options.orientation || 'horizontal'; // horizontal, vertical
    this.size = options.size || 'md';
    this.className = options.className || '';
  }

  create() {
    const orientationClass = this.orientation === 'vertical' ? 'flex-col' : 'flex-row';
    const baseClasses = 'inline-flex border border-gray-300 rounded-md overflow-hidden';

    const groupElement = super.create('div', {
      className: clsx('flex', orientationClass, baseClasses, this.className),
      attrs: { role: 'group' }
    });

    this.buttons.forEach((btnConfig, index) => {
      const btn = document.createElement('button');
      btn.textContent = btnConfig.label || 'Button';
      btn.className = clsx(
        'px-4 py-2 font-medium text-gray-700 hover:bg-gray-50 transition-colors duration-200',
        index > 0 ? (this.orientation === 'vertical' ? 'border-t border-gray-300' : 'border-l border-gray-300') : '',
        btnConfig.disabled ? 'opacity-50 cursor-not-allowed' : ''
      );
      btn.disabled = btnConfig.disabled || false;

      if (btnConfig.onClick) {
        btn.addEventListener('click', btnConfig.onClick);
      }

      groupElement.appendChild(btn);
    });

    return groupElement;
  }

  addButton(label, onClick) {
    if (this.element) {
      const btn = document.createElement('button');
      btn.textContent = label;
      btn.className = clsx(
        'px-4 py-2 font-medium text-gray-700 hover:bg-gray-50 transition-colors duration-200 border-l border-gray-300'
      );
      if (onClick) {
        btn.addEventListener('click', onClick);
      }
      this.element.appendChild(btn);
    }
  }
}
