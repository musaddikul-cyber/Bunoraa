/**
 * Tooltip Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Tooltip extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.content = options.content || '';
    this.position = options.position || 'top'; // top, bottom, left, right
    this.delay = options.delay || 200;
    this.trigger = options.trigger || 'hover'; // hover, focus, manual
    this.className = options.className || '';
    this.visible = false;
    this.timeoutId = null;
  }

  create() {
    const containerElement = super.create('div', {
      className: 'relative inline-block'
    });

    // Tooltip content
    const tooltipElement = document.createElement('div');
    tooltipElement.className = clsx(
      'absolute hidden bg-gray-900 text-white px-3 py-2 rounded text-sm whitespace-nowrap z-50',
      'opacity-0 transition-opacity duration-200',
      this.getPositionClasses(),
      this.className
    );
    tooltipElement.textContent = this.content;

    // Arrow
    const arrow = document.createElement('div');
    arrow.className = clsx(
      'absolute w-2 h-2 bg-gray-900 transform rotate-45',
      this.getArrowClasses()
    );
    tooltipElement.appendChild(arrow);

    containerElement.appendChild(tooltipElement);

    this.tooltipElement = tooltipElement;

    // Handle trigger
    if (this.trigger === 'hover') {
      containerElement.addEventListener('mouseenter', () => this.show());
      containerElement.addEventListener('mouseleave', () => this.hide());
    } else if (this.trigger === 'focus') {
      containerElement.addEventListener('focus', () => this.show(), true);
      containerElement.addEventListener('blur', () => this.hide(), true);
    }

    return containerElement;
  }

  getPositionClasses() {
    const positions = {
      top: 'bottom-full left-1/2 transform -translate-x-1/2 mb-2',
      bottom: 'top-full left-1/2 transform -translate-x-1/2 mt-2',
      left: 'right-full top-1/2 transform -translate-y-1/2 mr-2',
      right: 'left-full top-1/2 transform -translate-y-1/2 ml-2'
    };
    return positions[this.position] || positions.top;
  }

  getArrowClasses() {
    const arrowPositions = {
      top: 'top-full left-1/2 transform -translate-x-1/2 -translate-y-1/2',
      bottom: 'bottom-full left-1/2 transform -translate-x-1/2 translate-y-1/2',
      left: 'left-full top-1/2 transform translate-x-1/2 -translate-y-1/2',
      right: 'right-full top-1/2 transform -translate-x-1/2 -translate-y-1/2'
    };
    return arrowPositions[this.position] || arrowPositions.top;
  }

  show() {
    if (this.visible) return;

    this.timeoutId = setTimeout(() => {
      this.visible = true;
      this.tooltipElement.classList.remove('hidden');
      this.tooltipElement.classList.add('opacity-100');
    }, this.delay);
  }

  hide() {
    if (!this.visible) return;

    clearTimeout(this.timeoutId);
    this.visible = false;
    this.tooltipElement.classList.remove('opacity-100');
    this.tooltipElement.classList.add('hidden');
  }

  setContent(content) {
    this.content = content;
    if (this.tooltipElement) {
      this.tooltipElement.textContent = content;
    }
  }
}
