/**
 * Item Component (Generic list item)
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Item extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.label = options.label || '';
    this.value = options.value || '';
    this.icon = options.icon || null;
    this.className = options.className || '';
    this.selected = options.selected || false;
    this.disabled = options.disabled || false;
  }

  create() {
    const baseClasses = 'flex items-center gap-2 px-3 py-2 rounded cursor-pointer transition-colors duration-200 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed';
    
    const stateClasses = this.selected ? 'bg-blue-50 text-blue-600' : 'text-gray-900';

    const itemElement = super.create('div', {
      className: clsx(baseClasses, stateClasses, this.className),
      attrs: { 
        role: 'option',
        'aria-selected': this.selected,
        disabled: this.disabled ? '' : null,
        'data-value': this.value
      }
    });

    let html = '';
    if (this.icon) {
      html += `<span class="flex-shrink-0">${this.icon}</span>`;
    }
    html += `<span>${this.label}</span>`;

    itemElement.innerHTML = html;
    return itemElement;
  }

  setSelected(selected) {
    this.selected = selected;
    if (this.element) {
      this.attr('aria-selected', selected);
      this.toggleClass('bg-blue-50 text-blue-600', selected);
    }
  }

  setDisabled(disabled) {
    this.disabled = disabled;
    this.attr('disabled', disabled ? '' : null);
  }
}
