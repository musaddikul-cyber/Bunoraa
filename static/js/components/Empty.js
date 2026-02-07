/**
 * Empty State Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Empty extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.icon = options.icon || '📦';
    this.title = options.title || 'No data';
    this.message = options.message || 'There is no data to display';
    this.action = options.action || null; // {label, onClick}
    this.className = options.className || '';
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx(
        'flex flex-col items-center justify-center p-8 text-center',
        this.className
      )
    });

    // Icon
    const iconElement = createElement('div', {
      className: 'text-6xl mb-4',
      text: this.icon
    });
    containerElement.appendChild(iconElement);

    // Title
    const titleElement = createElement('h3', {
      className: 'text-lg font-semibold text-gray-900 mb-2',
      text: this.title
    });
    containerElement.appendChild(titleElement);

    // Message
    const messageElement = createElement('p', {
      className: 'text-gray-500 mb-4',
      text: this.message
    });
    containerElement.appendChild(messageElement);

    // Action button
    if (this.action) {
      const actionBtn = createElement('button', {
        className: 'px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors',
        text: this.action.label
      });

      actionBtn.addEventListener('click', this.action.onClick);
      containerElement.appendChild(actionBtn);
    }

    return containerElement;
  }
}
