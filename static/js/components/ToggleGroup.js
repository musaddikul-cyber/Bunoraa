/**
 * Toggle Group Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class ToggleGroup extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.items = options.items || []; // Array of {label, value}
    this.selected = options.selected || null;
    this.multiple = options.multiple || false;
    this.orientation = options.orientation || 'horizontal';
    this.className = options.className || '';
    this.onChange = options.onChange || null;
  }

  create() {
    const orientationClass = this.orientation === 'vertical' ? 'flex-col' : 'flex-row';
    const baseClasses = 'inline-flex border border-gray-300 rounded-md overflow-hidden';

    const groupElement = super.create('div', {
      className: clsx('flex', orientationClass, baseClasses, this.className),
      attrs: { role: 'group' }
    });

    this.toggleButtons = [];

    this.items.forEach((item, index) => {
      const isSelected = this.multiple 
        ? (Array.isArray(this.selected) && this.selected.includes(item.value))
        : item.value === this.selected;

      const btnClasses = clsx(
        'flex-1 px-4 py-2 font-medium transition-colors duration-200',
        isSelected ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50',
        index > 0 ? (this.orientation === 'vertical' ? 'border-t border-gray-300' : 'border-l border-gray-300') : ''
      );

      const btn = createElement('button', {
        className: btnClasses,
        text: item.label,
        attrs: {
          'data-value': item.value,
          'aria-pressed': isSelected,
          type: 'button'
        }
      });

      btn.addEventListener('click', () => {
        if (this.multiple) {
          if (!Array.isArray(this.selected)) this.selected = [];
          const index = this.selected.indexOf(item.value);
          if (index > -1) {
            this.selected.splice(index, 1);
          } else {
            this.selected.push(item.value);
          }
        } else {
          this.selected = item.value;
        }
        this.updateView();
        if (this.onChange) {
          this.onChange(this.selected);
        }
      });

      groupElement.appendChild(btn);
      this.toggleButtons.push(btn);
    });

    return groupElement;
  }

  updateView() {
    this.toggleButtons.forEach(btn => {
      const value = btn.getAttribute('data-value');
      const isSelected = this.multiple
        ? (Array.isArray(this.selected) && this.selected.includes(value))
        : value === this.selected;

      btn.setAttribute('aria-pressed', isSelected);
      btn.className = clsx(
        'flex-1 px-4 py-2 font-medium transition-colors duration-200',
        isSelected ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'
      );
    });
  }

  getValue() {
    return this.selected;
  }

  setValue(value) {
    this.selected = value;
    this.updateView();
  }
}
