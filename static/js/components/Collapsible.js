/**
 * Collapsible Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Collapsible extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.title = options.title || '';
    this.content = options.content || '';
    this.open = options.open || false;
    this.className = options.className || '';
    this.onChange = options.onChange || null;
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx('border border-gray-200 rounded-lg overflow-hidden', this.className)
    });

    // Header/Trigger
    const triggerElement = createElement('button', {
      className: clsx(
        'w-full px-4 py-3 flex items-center justify-between',
        'hover:bg-gray-50 transition-colors duration-200 text-left'
      ),
      attrs: { 'aria-expanded': this.open }
    });

    const titleSpan = createElement('span', {
      className: 'font-semibold text-gray-900',
      text: this.title
    });

    const chevron = createElement('span', {
      className: clsx(
        'w-5 h-5 transition-transform duration-300',
        this.open ? 'rotate-180' : ''
      ),
      html: '▼'
    });

    triggerElement.appendChild(titleSpan);
    triggerElement.appendChild(chevron);

    triggerElement.addEventListener('click', () => {
      this.toggle();
    });

    containerElement.appendChild(triggerElement);

    // Content
    const contentElement = createElement('div', {
      className: clsx(
        'overflow-hidden transition-all duration-300',
        this.open ? 'max-h-96' : 'max-h-0',
        'border-t border-gray-200'
      )
    });

    const innerContent = createElement('div', {
      className: 'px-4 py-3',
      html: this.content
    });

    contentElement.appendChild(innerContent);
    containerElement.appendChild(contentElement);

    this.triggerElement = triggerElement;
    this.contentElement = contentElement;
    this.chevron = chevron;

    return containerElement;
  }

  toggle() {
    this.open = !this.open;
    this.updateUI();
    if (this.onChange) {
      this.onChange(this.open);
    }
  }

  open() {
    if (!this.open) {
      this.open = true;
      this.updateUI();
      if (this.onChange) {
        this.onChange(true);
      }
    }
  }

  close() {
    if (this.open) {
      this.open = false;
      this.updateUI();
      if (this.onChange) {
        this.onChange(false);
      }
    }
  }

  updateUI() {
    this.triggerElement.setAttribute('aria-expanded', this.open);
    this.contentElement.className = clsx(
      'overflow-hidden transition-all duration-300',
      this.open ? 'max-h-96' : 'max-h-0',
      'border-t border-gray-200'
    );
    this.chevron.className = clsx(
      'w-5 h-5 transition-transform duration-300',
      this.open ? 'rotate-180' : ''
    );
  }

  setContent(content) {
    this.content = content;
    if (this.contentElement) {
      const inner = this.contentElement.querySelector('div');
      if (inner) {
        inner.innerHTML = content;
      }
    }
  }
}
