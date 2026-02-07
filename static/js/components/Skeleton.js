/**
 * Skeleton Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx } from './utils.js';

export class Skeleton extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.width = options.width || '100%';
    this.height = options.height || '1rem';
    this.className = options.className || '';
    this.variant = options.variant || 'default'; // default, circle, rect
    this.count = options.count || 1;
  }

  create() {
    const baseClasses = 'bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200 rounded animate-pulse';

    const containerElement = super.create('div', {
      className: this.className
    });

    for (let i = 0; i < this.count; i++) {
      const skeletonElement = document.createElement('div');
      
      let skeletonClass = baseClasses;
      
      if (this.variant === 'circle') {
        skeletonClass = clsx('w-10 h-10 rounded-full', baseClasses);
      } else {
        skeletonClass = clsx(baseClasses);
      }

      skeletonElement.className = skeletonClass;
      skeletonElement.style.width = this.width;
      skeletonElement.style.height = this.height;

      if (this.count > 1) {
        skeletonElement.style.marginBottom = '0.5rem';
      }

      containerElement.appendChild(skeletonElement);
    }

    return containerElement;
  }
}
