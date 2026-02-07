/**
 * Scroll Area Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class ScrollArea extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.content = options.content || '';
    this.height = options.height || '400px';
    this.className = options.className || '';
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx('relative border border-gray-200 rounded-lg', this.className)
    });

    // Scrollable content
    const scrollElement = createElement('div', {
      className: 'overflow-y-auto',
      html: this.content,
      attrs: {
        style: `height: ${this.height}`
      }
    });

    containerElement.appendChild(scrollElement);

    // Custom scrollbar styles
    const style = document.createElement('style');
    style.textContent = `
      div[class*="overflow-y-auto"]::-webkit-scrollbar {
        width: 8px;
      }
      div[class*="overflow-y-auto"]::-webkit-scrollbar-track {
        background: #f1f1f1;
      }
      div[class*="overflow-y-auto"]::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
      }
      div[class*="overflow-y-auto"]::-webkit-scrollbar-thumb:hover {
        background: #555;
      }
    `;
    document.head.appendChild(style);

    this.scrollElement = scrollElement;
    return containerElement;
  }

  setContent(content) {
    this.content = content;
    if (this.scrollElement) {
      this.scrollElement.innerHTML = content;
    }
  }

  scrollToTop() {
    if (this.scrollElement) {
      this.scrollElement.scrollTop = 0;
    }
  }

  scrollToBottom() {
    if (this.scrollElement) {
      this.scrollElement.scrollTop = this.scrollElement.scrollHeight;
    }
  }
}
