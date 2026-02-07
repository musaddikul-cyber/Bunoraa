/**
 * Sheet Component (Bottom Sheet)
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createBackdrop } from './utils.js';

export class Sheet extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.title = options.title || '';
    this.content = options.content || '';
    this.open = options.open || false;
    this.onClose = options.onClose || null;
    this.closeButton = options.closeButton !== false;
    this.closeOnBackdrop = options.closeOnBackdrop !== false;
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx(
        'fixed inset-0 z-50',
        this.open ? '' : 'hidden'
      )
    });

    // Backdrop
    const backdrop = createBackdrop();
    containerElement.appendChild(backdrop);

    if (this.closeOnBackdrop) {
      backdrop.addEventListener('click', () => this.close());
    }

    // Sheet
    const sheetElement = document.createElement('div');
    sheetElement.className = clsx(
      'fixed bottom-0 left-0 right-0 bg-white rounded-t-2xl shadow-lg transition-transform duration-300 flex flex-col z-50',
      this.open ? 'translate-y-0' : 'translate-y-full'
    );

    // Header with handle
    const handle = document.createElement('div');
    handle.className = 'w-12 h-1 bg-gray-300 rounded-full mx-auto mt-3 mb-2';
    sheetElement.appendChild(handle);

    if (this.title) {
      const header = document.createElement('div');
      header.className = 'px-6 py-4 border-b border-gray-200 flex items-center justify-between';
      
      const title = document.createElement('h2');
      title.className = 'text-xl font-semibold text-gray-900';
      title.textContent = this.title;
      header.appendChild(title);

      if (this.closeButton) {
        const closeBtn = document.createElement('button');
        closeBtn.className = 'text-gray-500 hover:text-gray-700 transition-colors duration-200';
        closeBtn.innerHTML = '×';
        closeBtn.addEventListener('click', () => this.close());
        header.appendChild(closeBtn);
      }

      sheetElement.appendChild(header);
    }

    // Content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'flex-1 overflow-y-auto px-6 py-4 max-h-[80vh]';
    contentDiv.innerHTML = this.content;
    sheetElement.appendChild(contentDiv);

    containerElement.appendChild(sheetElement);

    this.backdrop = backdrop;
    this.sheetElement = sheetElement;

    return containerElement;
  }

  open() {
    if (!this.element) this.create();
    
    this.open = true;
    this.element.classList.remove('hidden');
    this.sheetElement.classList.remove('translate-y-full');
    this.sheetElement.classList.add('translate-y-0');
    document.body.style.overflow = 'hidden';
  }

  close() {
    this.open = false;
    this.sheetElement.classList.remove('translate-y-0');
    this.sheetElement.classList.add('translate-y-full');
    
    setTimeout(() => {
      if (this.element?.parentNode) {
        this.element.classList.add('hidden');
      }
      document.body.style.overflow = '';
    }, 300);
    
    if (this.onClose) {
      this.onClose();
    }
  }

  toggle() {
    this.open ? this.close() : this.open();
  }
}
