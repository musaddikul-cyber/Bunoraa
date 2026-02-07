/**
 * Pagination Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Pagination extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.totalPages = options.totalPages || 1;
    this.currentPage = options.currentPage || 1;
    this.maxVisible = options.maxVisible || 5;
    this.className = options.className || '';
    this.onChange = options.onChange || null;
  }

  create() {
    const baseClasses = 'flex items-center gap-2';

    const paginationElement = super.create('nav', {
      className: clsx(baseClasses, this.className),
      attrs: { 'aria-label': 'Pagination' }
    });

    // Previous button
    const prevBtn = createElement('button', {
      className: clsx(
        'px-3 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed',
        this.currentPage === 1 ? 'opacity-50 cursor-not-allowed' : ''
      ),
      text: 'Previous',
      attrs: { disabled: this.currentPage === 1 ? '' : null }
    });

    prevBtn.addEventListener('click', () => {
      if (this.currentPage > 1) {
        this.goToPage(this.currentPage - 1);
      }
    });

    paginationElement.appendChild(prevBtn);

    // Page numbers
    const pageStart = Math.max(1, this.currentPage - Math.floor(this.maxVisible / 2));
    const pageEnd = Math.min(this.totalPages, pageStart + this.maxVisible - 1);

    if (pageStart > 1) {
      const firstBtn = this.createPageButton(1);
      paginationElement.appendChild(firstBtn);

      if (pageStart > 2) {
        const dots = createElement('span', { text: '...', className: 'px-2 py-2' });
        paginationElement.appendChild(dots);
      }
    }

    for (let i = pageStart; i <= pageEnd; i++) {
      const pageBtn = this.createPageButton(i);
      paginationElement.appendChild(pageBtn);
    }

    if (pageEnd < this.totalPages) {
      if (pageEnd < this.totalPages - 1) {
        const dots = createElement('span', { text: '...', className: 'px-2 py-2' });
        paginationElement.appendChild(dots);
      }

      const lastBtn = this.createPageButton(this.totalPages);
      paginationElement.appendChild(lastBtn);
    }

    // Next button
    const nextBtn = createElement('button', {
      className: clsx(
        'px-3 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed',
        this.currentPage === this.totalPages ? 'opacity-50 cursor-not-allowed' : ''
      ),
      text: 'Next',
      attrs: { disabled: this.currentPage === this.totalPages ? '' : null }
    });

    nextBtn.addEventListener('click', () => {
      if (this.currentPage < this.totalPages) {
        this.goToPage(this.currentPage + 1);
      }
    });

    paginationElement.appendChild(nextBtn);

    return paginationElement;
  }

  createPageButton(page) {
    const isActive = page === this.currentPage;
    const btn = createElement('button', {
      className: clsx(
        'px-3 py-2 rounded-md font-medium transition-colors duration-200',
        isActive
          ? 'bg-blue-600 text-white'
          : 'border border-gray-300 text-gray-700 hover:bg-gray-50'
      ),
      text: page.toString(),
      attrs: { 'aria-current': isActive ? 'page' : null }
    });

    btn.addEventListener('click', () => {
      this.goToPage(page);
    });

    return btn;
  }

  goToPage(page) {
    if (page < 1 || page > this.totalPages || page === this.currentPage) return;
    
    this.currentPage = page;
    
    if (this.onChange) {
      this.onChange(page);
    }

    // Recreate to update UI
    if (this.element?.parentNode) {
      const newElement = this.create();
      this.element.parentNode.replaceChild(newElement, this.element);
      this.element = newElement;
    }
  }
}
