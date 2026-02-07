/**
 * Context Menu Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class ContextMenu extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.items = options.items || []; // Array of {label, onClick, icon, disabled}
    this.className = options.className || '';
    this.visible = false;
  }

  create() {
    const containerElement = super.create('div', {
      className: 'relative inline-block w-full'
    });

    // Menu content
    const menuElement = document.createElement('div');
    menuElement.className = clsx(
      'absolute hidden bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-50 min-w-max',
      this.className
    );

    this.items.forEach((item) => {
      const itemElement = createElement('button', {
        className: clsx(
          'w-full text-left px-4 py-2 hover:bg-gray-100 transition-colors duration-200 flex items-center gap-2',
          item.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
        ),
        attrs: {
          disabled: item.disabled ? '' : null,
          'data-action': item.label
        }
      });

      if (item.icon) {
        const iconSpan = document.createElement('span');
        iconSpan.innerHTML = item.icon;
        itemElement.appendChild(iconSpan);
      }

      const labelSpan = document.createElement('span');
      labelSpan.textContent = item.label;
      itemElement.appendChild(labelSpan);

      itemElement.addEventListener('click', () => {
        if (!item.disabled && item.onClick) {
          item.onClick();
        }
        this.hide();
      });

      menuElement.appendChild(itemElement);
    });

    containerElement.appendChild(menuElement);

    // Show on right click
    containerElement.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this.showAt(e.clientX, e.clientY, menuElement);
    });

    // Hide on outside click
    document.addEventListener('click', () => {
      if (this.visible) {
        this.hide();
      }
    });

    this.menuElement = menuElement;
    return containerElement;
  }

  showAt(x, y, menuElement) {
    if (!menuElement) return;

    this.visible = true;
    menuElement.classList.remove('hidden');
    menuElement.style.position = 'fixed';
    menuElement.style.left = x + 'px';
    menuElement.style.top = y + 'px';
  }

  hide() {
    this.visible = false;
    if (this.menuElement) {
      this.menuElement.classList.add('hidden');
      this.menuElement.style.position = 'absolute';
    }
  }

  addItem(label, onClick, icon = null) {
    const item = { label, onClick, icon };
    this.items.push(item);
    
    if (this.menuElement) {
      const itemElement = createElement('button', {
        className: 'w-full text-left px-4 py-2 hover:bg-gray-100 transition-colors duration-200 flex items-center gap-2',
      });

      if (icon) {
        const iconSpan = document.createElement('span');
        iconSpan.innerHTML = icon;
        itemElement.appendChild(iconSpan);
      }

      itemElement.textContent = label;
      itemElement.addEventListener('click', () => {
        onClick();
        this.hide();
      });

      this.menuElement.appendChild(itemElement);
    }
  }
}
