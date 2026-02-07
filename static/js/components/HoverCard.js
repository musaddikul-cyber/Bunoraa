/**
 * Hover Card Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class HoverCard extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.trigger = options.trigger || 'Hover me';
    this.content = options.content || '';
    this.position = options.position || 'bottom'; // top, bottom, left, right
    this.delay = options.delay || 200;
    this.className = options.className || '';
    this.visible = false;
    this.timeoutId = null;
  }

  create() {
    const containerElement = super.create('div', {
      className: 'relative inline-block'
    });

    // Trigger
    const triggerElement = document.createElement('div');
    triggerElement.className = 'cursor-pointer px-3 py-2 rounded hover:bg-gray-100 transition-colors duration-200';
    triggerElement.textContent = this.trigger;

    containerElement.appendChild(triggerElement);

    // Card content
    const cardElement = document.createElement('div');
    cardElement.className = clsx(
      'absolute hidden bg-white border border-gray-200 rounded-lg shadow-lg p-4 z-50',
      'min-w-max max-w-sm',
      this.getPositionClasses(),
      this.className
    );
    cardElement.innerHTML = this.content;

    containerElement.appendChild(cardElement);

    // Hover handlers
    containerElement.addEventListener('mouseenter', () => this.show(cardElement));
    containerElement.addEventListener('mouseleave', () => this.hide(cardElement));

    this.cardElement = cardElement;

    return containerElement;
  }

  getPositionClasses() {
    const positions = {
      top: 'bottom-full left-0 mb-2',
      bottom: 'top-full left-0 mt-2',
      left: 'right-full top-0 mr-2',
      right: 'left-full top-0 ml-2'
    };
    return positions[this.position] || positions.bottom;
  }

  show(cardElement = this.cardElement) {
    if (this.visible || !cardElement) return;

    this.timeoutId = setTimeout(() => {
      this.visible = true;
      cardElement.classList.remove('hidden');
      cardElement.classList.add('opacity-100', 'transition-opacity', 'duration-200');
    }, this.delay);
  }

  hide(cardElement = this.cardElement) {
    if (!this.visible || !cardElement) return;

    clearTimeout(this.timeoutId);
    this.visible = false;
    cardElement.classList.add('hidden');
    cardElement.classList.remove('opacity-100');
  }

  setContent(content) {
    this.content = content;
    if (this.cardElement) {
      this.cardElement.innerHTML = content;
    }
  }
}
