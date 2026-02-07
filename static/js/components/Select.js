/**
 * Select Component (Custom)
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Select extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.items = options.items || [];
    this.selected = options.selected || '';
    this.placeholder = options.placeholder || 'Select...';
    this.searchable = options.searchable || true;
    this.clearable = options.clearable || true;
    this.multiple = options.multiple || false;
    this.className = options.className || '';
    this.onChange = options.onChange || null;
    this.open = false;
  }

  create() {
    const containerElement = super.create('div', {
      className: 'relative w-full'
    });

    // Select button
    const buttonElement = createElement('button', {
      className: clsx(
        'w-full px-3 py-2 border border-gray-300 rounded-md text-left text-gray-900',
        'hover:border-gray-400 transition-colors duration-200 flex items-center justify-between'
      ),
      html: `<span>${this.getDisplayText()}</span><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path></svg>`
    });

    containerElement.appendChild(buttonElement);

    // Dropdown menu
    const menuElement = createElement('div', {
      className: clsx(
        'absolute hidden top-full left-0 right-0 mt-1 bg-white border border-gray-300 rounded-md shadow-lg z-50',
        'max-h-64 overflow-y-auto'
      )
    });

    // Search input (if searchable)
    if (this.searchable) {
      const searchInput = document.createElement('input');
      searchInput.type = 'text';
      searchInput.placeholder = 'Search...';
      searchInput.className = 'w-full px-3 py-2 border-b border-gray-200 focus:outline-none';
      searchInput.addEventListener('input', (e) => {
        this.filterItems(e.target.value, menuElement);
      });
      menuElement.appendChild(searchInput);
    }

    // Items
    this.renderItems(menuElement);

    containerElement.appendChild(menuElement);

    buttonElement.addEventListener('click', () => {
      this.toggle(menuElement);
    });

    document.addEventListener('click', (e) => {
      if (!containerElement.contains(e.target) && this.open) {
        this.close(menuElement);
      }
    });

    this.buttonElement = buttonElement;
    this.menuElement = menuElement;

    return containerElement;
  }

  renderItems(menuElement) {
    const itemsContainer = menuElement.querySelector('[data-items]') || menuElement;
    if (menuElement.querySelector('[data-items]')) {
      menuElement.querySelector('[data-items]').innerHTML = '';
    }

    this.items.forEach(item => {
      const itemElement = createElement('div', {
        className: clsx(
          'px-3 py-2 cursor-pointer hover:bg-blue-50',
          this.isSelected(item.value) ? 'bg-blue-100' : ''
        ),
        text: item.label,
        attrs: { 'data-value': item.value }
      });

      itemElement.addEventListener('click', () => {
        if (this.multiple) {
          if (!Array.isArray(this.selected)) this.selected = [];
          const idx = this.selected.indexOf(item.value);
          if (idx > -1) {
            this.selected.splice(idx, 1);
          } else {
            this.selected.push(item.value);
          }
        } else {
          this.selected = item.value;
          this.close(menuElement);
        }
        this.updateDisplay();
        if (this.onChange) {
          this.onChange(this.selected);
        }
      });

      menuElement.appendChild(itemElement);
    });
  }

  filterItems(query, menuElement) {
    const items = menuElement.querySelectorAll('[data-value]');
    items.forEach(item => {
      const isMatch = item.textContent.toLowerCase().includes(query.toLowerCase());
      item.style.display = isMatch ? '' : 'none';
    });
  }

  getDisplayText() {
    if (this.multiple && Array.isArray(this.selected)) {
      if (this.selected.length === 0) return this.placeholder;
      return this.selected.map(v => this.items.find(i => i.value === v)?.label).join(', ');
    }
    const selected = this.items.find(i => i.value === this.selected);
    return selected ? selected.label : this.placeholder;
  }

  isSelected(value) {
    return this.multiple ? this.selected.includes(value) : this.selected === value;
  }

  updateDisplay() {
    if (this.buttonElement) {
      const span = this.buttonElement.querySelector('span');
      if (span) span.textContent = this.getDisplayText();
    }
  }

  toggle(menuElement) {
    this.open ? this.close(menuElement) : this.open(menuElement);
  }

  open(menuElement) {
    this.open = true;
    menuElement.classList.remove('hidden');
  }

  close(menuElement) {
    this.open = false;
    menuElement.classList.add('hidden');
  }
}
