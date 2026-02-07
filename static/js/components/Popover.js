/**
 * Popover Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Popover extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.trigger = options.trigger || 'Click to open';
    this.content = options.content || '';
    this.position = options.position || 'bottom'; // top, bottom, left, right
    this.className = options.className || '';
    this.open = false;
  }

  create() {
    const containerElement = super.create('div', {
      className: 'relative inline-block'
    });

    // Trigger button
    const triggerBtn = document.createElement('button');
    triggerBtn.className = 'px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors duration-200';
    triggerBtn.textContent = this.trigger;
    triggerBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggle();
    });

    containerElement.appendChild(triggerBtn);

    // Popover content
    const popoverElement = document.createElement('div');
    popoverElement.className = clsx(
      'absolute hidden bg-white border border-gray-200 rounded-lg shadow-lg p-4 z-50',
      'min-w-max max-w-sm',
      this.getPositionClasses(),
      this.className
    );
    popoverElement.innerHTML = this.content;

    containerElement.appendChild(popoverElement);

    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!containerElement.contains(e.target) && this.open) {
        this.close();
      }
    });

    this.popoverElement = popoverElement;
    this.triggerBtn = triggerBtn;

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

  toggle() {
    this.open ? this.close() : this.open();
  }

  open() {
    if (!this.popoverElement) return;
    
    this.open = true;
    this.popoverElement.classList.remove('hidden');
  }

  close() {
    if (!this.popoverElement) return;
    
    this.open = false;
    this.popoverElement.classList.add('hidden');
  }

  setContent(content) {
    this.content = content;
    if (this.popoverElement) {
      this.popoverElement.innerHTML = content;
    }
  }
}
