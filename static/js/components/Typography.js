/**
 * Typography Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Typography extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.text = options.text || '';
    this.variant = options.variant || 'body'; // h1, h2, h3, h4, h5, h6, body, small, muted, lead, code
    this.className = options.className || '';
  }

  create() {
    const variantConfig = {
      h1: { tag: 'h1', class: 'text-4xl font-bold tracking-tight' },
      h2: { tag: 'h2', class: 'text-3xl font-bold tracking-tight' },
      h3: { tag: 'h3', class: 'text-2xl font-bold tracking-tight' },
      h4: { tag: 'h4', class: 'text-xl font-bold tracking-tight' },
      h5: { tag: 'h5', class: 'text-lg font-bold' },
      h6: { tag: 'h6', class: 'text-base font-bold' },
      body: { tag: 'p', class: 'text-base text-gray-700' },
      small: { tag: 'p', class: 'text-sm text-gray-600' },
      muted: { tag: 'p', class: 'text-sm text-gray-500' },
      lead: { tag: 'p', class: 'text-xl text-gray-600' },
      code: { tag: 'code', class: 'bg-gray-100 text-red-600 px-2 py-1 rounded font-mono text-sm' }
    };

    const config = variantConfig[this.variant] || variantConfig.body;

    const element = super.create(config.tag, {
      className: clsx(config.class, this.className),
      text: this.text
    });

    return element;
  }

  setText(text) {
    this.text = text;
    if (this.element) {
      this.element.textContent = text;
    }
  }
}
