/**
 * Drawer Component (Side Sheet)
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createBackdrop, createFocusTrap, keyboard } from './utils.js';

export class Drawer extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.title = options.title || '';
    this.content = options.content || '';
    this.position = options.position || 'right'; // left, right
    this.open = options.open || false;
    this.onClose = options.onClose || null;
    this.closeButton = options.closeButton !== false;
    this.closeOnBackdrop = options.closeOnBackdrop !== false;
    this.closeOnEscape = options.closeOnEscape !== false;
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

    // Drawer
    const positionClasses = this.position === 'left' ? 'left-0' : 'right-0';
    const drawerElement = document.createElement('div');
    drawerElement.className = clsx(
      'absolute top-0 h-full w-96 bg-white shadow-lg transition-transform duration-300 flex flex-col z-50',
      positionClasses,
      this.open ? 'translate-x-0' : (this.position === 'left' ? '-translate-x-full' : 'translate-x-full')
    );

    // Header
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

      drawerElement.appendChild(header);
    }

    // Content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'flex-1 overflow-y-auto px-6 py-4';
    contentDiv.innerHTML = this.content;
    drawerElement.appendChild(contentDiv);

    containerElement.appendChild(drawerElement);

    // Handle keyboard
    this.on('keydown', (e) => {
      if (keyboard.isEscape(e) && this.closeOnEscape) {
        this.close();
      }
    }, { once: false });

    this.backdrop = backdrop;
    this.drawerElement = drawerElement;

    return containerElement;
  }

  open() {
    if (!this.element) this.create();
    
    this.open = true;
    this.element.classList.remove('hidden');
    this.drawerElement.classList.remove('-translate-x-full', 'translate-x-full');
    this.drawerElement.classList.add('translate-x-0');
    document.body.style.overflow = 'hidden';
  }

  close() {
    this.open = false;
    const translateClass = this.position === 'left' ? '-translate-x-full' : 'translate-x-full';
    this.drawerElement.classList.remove('translate-x-0');
    this.drawerElement.classList.add(translateClass);
    
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
