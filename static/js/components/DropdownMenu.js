/**
 * Dropdown Menu Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class DropdownMenu extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.trigger = options.trigger || 'Menu';
    this.items = options.items || []; // Array of {label, onClick, icon, disabled, divider}
    this.position = options.position || 'bottom'; // top, bottom
    this.align = options.align || 'left'; // left, right
    this.className = options.className || '';
    this.open = false;
  }

  create() {
    const containerElement = super.create('div', {
      className: 'relative inline-block'
    });

    // Trigger button
    const triggerBtn = document.createElement('button');
    triggerBtn.className = 'px-4 py-2 bg-gray-100 text-gray-900 rounded hover:bg-gray-200 transition-colors duration-200 flex items-center gap-2';
    triggerBtn.innerHTML = `${this.trigger} <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path></svg>`;
    triggerBtn.addEventListener('click', () => this.toggle());

    containerElement.appendChild(triggerBtn);

    // Menu
    const positionClass = this.position === 'top' ? 'bottom-full mb-2' : 'top-full mt-2';
    const alignClass = this.align === 'right' ? 'right-0' : 'left-0';

    const menuElement = document.createElement('div');
    menuElement.className = clsx(
      'absolute hidden bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-50 min-w-max',
      positionClass,
      alignClass,
      this.className
    );

    this.items.forEach((item) => {
      if (item.divider) {
        const divider = document.createElement('div');
        divider.className = 'border-t border-gray-200 my-1';
        menuElement.appendChild(divider);
        return;
      }

      const itemElement = createElement('button', {
        className: clsx(
          'w-full text-left px-4 py-2 hover:bg-gray-100 transition-colors duration-200 flex items-center gap-2',
          item.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
        ),
        attrs: {
          disabled: item.disabled ? '' : null
        }
      });

      if (item.icon) {
        const iconSpan = document.createElement('span');
        iconSpan.innerHTML = item.icon;
        iconSpan.className = 'w-4 h-4';
        itemElement.appendChild(iconSpan);
      }

      const labelSpan = document.createElement('span');
      labelSpan.textContent = item.label;
      itemElement.appendChild(labelSpan);

      itemElement.addEventListener('click', () => {
        if (!item.disabled && item.onClick) {
          item.onClick();
        }
        this.close();
      });

      menuElement.appendChild(itemElement);
    });

    containerElement.appendChild(menuElement);

    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!containerElement.contains(e.target) && this.open) {
        this.close();
      }
    });

    this.triggerBtn = triggerBtn;
    this.menuElement = menuElement;

    return containerElement;
  }

  toggle() {
    this.open ? this.close() : this.open();
  }

  open() {
    this.open = true;
    this.menuElement.classList.remove('hidden');
  }

  close() {
    this.open = false;
    this.menuElement.classList.add('hidden');
  }

  addItem(label, onClick, icon = null) {
    const item = { label, onClick, icon };
    this.items.push(item);
    
    if (this.menuElement) {
      const itemElement = createElement('button', {
        className: 'w-full text-left px-4 py-2 hover:bg-gray-100 transition-colors duration-200 flex items-center gap-2'
      });

      if (icon) {
        const iconSpan = document.createElement('span');
        iconSpan.innerHTML = icon;
        iconSpan.className = 'w-4 h-4';
        itemElement.appendChild(iconSpan);
      }

      const labelSpan = document.createElement('span');
      labelSpan.textContent = label;
      itemElement.appendChild(labelSpan);

      itemElement.addEventListener('click', () => {
        onClick();
        this.close();
      });

      this.menuElement.appendChild(itemElement);
    }
  }
}
