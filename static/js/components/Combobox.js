/**
 * Combobox Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Combobox extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.items = options.items || [];
    this.value = options.value || '';
    this.placeholder = options.placeholder || 'Search...';
    this.className = options.className || '';
    this.onChange = options.onChange || null;
    this.open = false;
  }

  create() {
    const containerElement = super.create('div', {
      className: 'relative w-full'
    });

    // Input
    const inputElement = document.createElement('input');
    inputElement.type = 'text';
    inputElement.placeholder = this.placeholder;
    inputElement.value = this.value;
    inputElement.className = clsx(
      'w-full px-3 py-2 border border-gray-300 rounded-md',
      'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent'
    );

    // Dropdown list
    const listElement = createElement('div', {
      className: clsx(
        'absolute hidden top-full left-0 right-0 mt-1 bg-white border border-gray-300 rounded-md shadow-lg z-50',
        'max-h-64 overflow-y-auto'
      )
    });

    const renderList = (query = '') => {
      listElement.innerHTML = '';
      const filtered = this.items.filter(item =>
        item.label.toLowerCase().includes(query.toLowerCase())
      );

      if (filtered.length === 0) {
        const noResults = createElement('div', {
          className: 'px-3 py-2 text-gray-500',
          text: 'No results found'
        });
        listElement.appendChild(noResults);
        return;
      }

      filtered.forEach(item => {
        const option = createElement('div', {
          className: clsx(
            'px-3 py-2 cursor-pointer hover:bg-blue-50 transition-colors',
            item.value === this.value ? 'bg-blue-100' : ''
          ),
          text: item.label,
          attrs: { 'data-value': item.value }
        });

        option.addEventListener('click', () => {
          this.value = item.value;
          inputElement.value = item.label;
          listElement.classList.add('hidden');
          if (this.onChange) {
            this.onChange(this.value, item);
          }
        });

        listElement.appendChild(option);
      });
    };

    inputElement.addEventListener('input', (e) => {
      renderList(e.target.value);
      listElement.classList.remove('hidden');
    });

    inputElement.addEventListener('focus', () => {
      renderList(inputElement.value);
      listElement.classList.remove('hidden');
    });

    inputElement.addEventListener('blur', () => {
      setTimeout(() => {
        listElement.classList.add('hidden');
      }, 150);
    });

    containerElement.appendChild(inputElement);
    containerElement.appendChild(listElement);

    document.addEventListener('click', (e) => {
      if (!containerElement.contains(e.target)) {
        listElement.classList.add('hidden');
      }
    });

    this.inputElement = inputElement;
    this.listElement = listElement;
    renderList();

    return containerElement;
  }

  getValue() {
    return this.value;
  }

  setValue(value) {
    this.value = value;
    const item = this.items.find(i => i.value === value);
    if (item && this.inputElement) {
      this.inputElement.value = item.label;
    }
  }
}
