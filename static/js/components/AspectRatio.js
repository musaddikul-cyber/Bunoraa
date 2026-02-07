/**
 * Aspect Ratio Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class AspectRatio extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.ratio = options.ratio || 16 / 9; // 16:9, 4:3, 1:1, etc
    this.content = options.content || '';
    this.className = options.className || '';
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx('relative w-full', this.className),
      attrs: {
        style: `aspect-ratio: ${this.ratio}`
      }
    });

    const innerElement = document.createElement('div');
    innerElement.className = 'w-full h-full';
    innerElement.innerHTML = this.content;

    containerElement.appendChild(innerElement);

    return containerElement;
  }

  setContent(content) {
    this.content = content;
    if (this.element) {
      const inner = this.element.querySelector('div');
      if (inner) {
        inner.innerHTML = content;
      }
    }
  }
}
