/**
 * Command Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Command extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.commands = options.commands || []; // {label, action, shortcut, category}
    this.placeholder = options.placeholder || 'Type a command...';
    this.className = options.className || '';
    this.open = false;
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx(
        'fixed inset-0 z-50 hidden flex items-start justify-center pt-20',
        this.open ? 'flex' : ''
      )
    });

    // Backdrop
    const backdrop = createElement('div', {
      className: 'absolute inset-0 bg-black bg-opacity-50'
    });
    containerElement.appendChild(backdrop);

    backdrop.addEventListener('click', () => this.close());

    // Command palette
    const paletteElement = createElement('div', {
      className: 'relative w-full max-w-md bg-white rounded-lg shadow-lg z-50'
    });

    // Input
    const inputElement = document.createElement('input');
    inputElement.type = 'text';
    inputElement.placeholder = this.placeholder;
    inputElement.className = 'w-full px-4 py-3 border-b border-gray-200 focus:outline-none';
    inputElement.autofocus = true;

    paletteElement.appendChild(inputElement);

    // Results
    const resultsElement = createElement('div', {
      className: 'max-h-96 overflow-y-auto'
    });

    const renderResults = (query = '') => {
      resultsElement.innerHTML = '';

      const filtered = query
        ? this.commands.filter(cmd =>
            cmd.label.toLowerCase().includes(query.toLowerCase())
          )
        : this.commands;

      if (filtered.length === 0) {
        const noResults = createElement('div', {
          className: 'px-4 py-3 text-sm text-gray-500',
          text: 'No commands found'
        });
        resultsElement.appendChild(noResults);
        return;
      }

      let currentCategory = '';

      filtered.forEach(cmd => {
        if (cmd.category && cmd.category !== currentCategory) {
          currentCategory = cmd.category;
          const categoryHeader = createElement('div', {
            className: 'px-4 py-2 text-xs font-semibold text-gray-600 bg-gray-50 uppercase',
            text: currentCategory
          });
          resultsElement.appendChild(categoryHeader);
        }

        const cmdElement = createElement('div', {
          className: clsx(
            'px-4 py-2 cursor-pointer hover:bg-blue-50 transition-colors flex items-center justify-between group'
          )
        });

        const label = createElement('span', {
          text: cmd.label,
          className: 'text-sm text-gray-900'
        });
        cmdElement.appendChild(label);

        if (cmd.shortcut) {
          const shortcut = createElement('span', {
            className: 'text-xs text-gray-500 group-hover:text-gray-700',
            text: cmd.shortcut
          });
          cmdElement.appendChild(shortcut);
        }

        cmdElement.addEventListener('click', () => {
          if (cmd.action) {
            cmd.action();
          }
          this.close();
        });

        resultsElement.appendChild(cmdElement);
      });
    };

    inputElement.addEventListener('input', (e) => {
      renderResults(e.target.value);
    });

    paletteElement.appendChild(resultsElement);
    containerElement.appendChild(paletteElement);

    inputElement.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.close();
      }
    });

    renderResults();
    this.containerElement = containerElement;

    return containerElement;
  }

  open() {
    if (!this.element) this.create();
    this.open = true;
    this.element.classList.remove('hidden');
    this.element.classList.add('flex');
  }

  close() {
    this.open = false;
    this.element?.classList.remove('flex');
    this.element?.classList.add('hidden');
  }

  toggle() {
    this.open ? this.close() : this.open();
  }
}
