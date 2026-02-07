/**
 * Avatar Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Avatar extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.src = options.src || '';
    this.alt = options.alt || '';
    this.initials = options.initials || '';
    this.size = options.size || 'md'; // xs, sm, md, lg, xl
    this.className = options.className || '';
    this.fallbackBg = options.fallbackBg || 'bg-blue-600';
  }

  create() {
    const sizeClasses = {
      xs: 'w-6 h-6 text-xs',
      sm: 'w-8 h-8 text-sm',
      md: 'w-10 h-10 text-base',
      lg: 'w-12 h-12 text-lg',
      xl: 'w-16 h-16 text-xl'
    };

    const baseClasses = 'rounded-full overflow-hidden flex items-center justify-center flex-shrink-0 font-semibold';

    if (this.src) {
      const avatarElement = super.create('img', {
        className: clsx(baseClasses, sizeClasses[this.size], this.className),
        attrs: {
          src: this.src,
          alt: this.alt,
          role: 'img'
        }
      });
      return avatarElement;
    } else if (this.initials) {
      const avatarElement = super.create('div', {
        className: clsx(
          baseClasses,
          sizeClasses[this.size],
          'text-white',
          this.fallbackBg,
          this.className
        ),
        text: this.initials.toUpperCase()
      });
      return avatarElement;
    } else {
      const avatarElement = super.create('div', {
        className: clsx(baseClasses, sizeClasses[this.size], 'bg-gray-300', this.className),
        html: '<svg class="w-full h-full text-gray-500" fill="currentColor" viewBox="0 0 24 24"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>'
      });
      return avatarElement;
    }
  }

  setSrc(src, alt = '') {
    this.src = src;
    this.alt = alt;
    if (this.element && this.element.tagName === 'IMG') {
      this.element.src = src;
      this.element.alt = alt;
    }
  }

  setInitials(initials, bgColor = '') {
    this.initials = initials;
    if (bgColor) this.fallbackBg = bgColor;
    if (this.element) {
      this.element.textContent = initials.toUpperCase();
      if (bgColor && this.element.className.includes('bg-')) {
        const bgClass = this.element.className.match(/bg-\S+/)[0];
        this.element.classList.remove(bgClass);
        this.element.classList.add(bgColor);
      }
    }
  }
}
