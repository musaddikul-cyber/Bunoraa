/**
 * Switch Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Switch extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.checked = options.checked || false;
    this.disabled = options.disabled || false;
    this.name = options.name || '';
    this.className = options.className || '';
    this.onChange = options.onChange || null;
  }

  create() {
    const containerClasses = 'flex items-center gap-2';
    const containerElement = super.create('div', {
      className: clsx(containerClasses, this.className)
    });

    const switchClasses = clsx(
      'relative inline-flex items-center h-6 w-11 rounded-full transition-colors duration-300 cursor-pointer',
      this.checked ? 'bg-blue-600' : 'bg-gray-300',
      this.disabled ? 'opacity-50 cursor-not-allowed' : ''
    );

    const switchElement = document.createElement('button');
    switchElement.className = switchClasses;
    switchElement.type = 'button';
    switchElement.setAttribute('role', 'switch');
    switchElement.setAttribute('aria-checked', this.checked);
    if (this.disabled) switchElement.disabled = true;

    // Create thumb
    const thumbClasses = clsx(
      'absolute w-5 h-5 bg-white rounded-full transition-transform duration-300 shadow',
      this.checked ? 'translate-x-5' : 'translate-x-0'
    );

    const thumbElement = document.createElement('span');
    thumbElement.className = thumbClasses;
    switchElement.appendChild(thumbElement);

    containerElement.appendChild(switchElement);

    switchElement.addEventListener('click', () => {
      if (!this.disabled) {
        this.toggle();
      }
    });

    switchElement.addEventListener('keydown', (e) => {
      if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        if (!this.disabled) {
          this.toggle();
        }
      }
    });

    this.switchElement = switchElement;
    this.thumbElement = thumbElement;

    return containerElement;
  }

  isChecked() {
    return this.checked;
  }

  setChecked(checked) {
    this.checked = checked;
    if (this.switchElement) {
      this.switchElement.setAttribute('aria-checked', checked);
      this.switchElement.className = clsx(
        'relative inline-flex items-center h-6 w-11 rounded-full transition-colors duration-300',
        checked ? 'bg-blue-600' : 'bg-gray-300',
        this.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
      );

      this.thumbElement.className = clsx(
        'absolute w-5 h-5 bg-white rounded-full transition-transform duration-300 shadow',
        checked ? 'translate-x-5' : 'translate-x-0'
      );

      if (this.onChange) {
        this.onChange(checked);
      }
    }
  }

  toggle() {
    this.setChecked(!this.checked);
  }

  setDisabled(disabled) {
    this.disabled = disabled;
    if (this.switchElement) {
      this.switchElement.disabled = disabled;
      this.switchElement.className = clsx(
        'relative inline-flex items-center h-6 w-11 rounded-full transition-colors duration-300',
        this.checked ? 'bg-blue-600' : 'bg-gray-300',
        disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
      );
    }
  }
}
