/**
 * Dialog Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createBackdrop, createFocusTrap, keyboard } from './utils.js';

export class Dialog extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.title = options.title || '';
    this.content = options.content || '';
    this.size = options.size || 'md'; // sm, md, lg, xl
    this.open = options.open || false;
    this.onClose = options.onClose || null;
    this.closeButton = options.closeButton !== false;
    this.closeOnBackdrop = options.closeOnBackdrop !== false;
    this.closeOnEscape = options.closeOnEscape !== false;
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx(
        'fixed inset-0 z-50 flex items-center justify-center',
        this.open ? '' : 'hidden'
      ),
      attrs: {
        role: 'dialog',
        'aria-modal': 'true',
        'aria-labelledby': `${this.id}-title`,
        'aria-describedby': `${this.id}-description`
      }
    });

    // Backdrop
    const backdrop = createBackdrop('dialog-backdrop');
    containerElement.appendChild(backdrop);

    if (this.closeOnBackdrop) {
      backdrop.addEventListener('click', () => this.close());
    }

    // Dialog content
    const sizeClasses = {
      sm: 'w-full max-w-sm',
      md: 'w-full max-w-md',
      lg: 'w-full max-w-lg',
      xl: 'w-full max-w-xl'
    };

    const dialogElement = document.createElement('div');
    dialogElement.className = clsx(
      'bg-white rounded-lg shadow-lg relative z-50',
      sizeClasses[this.size],
      'mx-4 max-h-[90vh] overflow-y-auto'
    );

    // Header
    if (this.title) {
      const header = document.createElement('div');
      header.className = 'px-6 py-4 border-b border-gray-200 flex items-center justify-between';
      
      const title = document.createElement('h2');
      title.id = `${this.id}-title`;
      title.className = 'text-xl font-semibold text-gray-900';
      title.textContent = this.title;
      header.appendChild(title);

      if (this.closeButton) {
        const closeBtn = document.createElement('button');
        closeBtn.className = 'text-gray-500 hover:text-gray-700 transition-colors duration-200';
        closeBtn.innerHTML = '×';
        closeBtn.setAttribute('aria-label', 'Close');
        closeBtn.addEventListener('click', () => this.close());
        header.appendChild(closeBtn);
      }

      dialogElement.appendChild(header);
    }

    // Content
    if (this.content) {
      const contentDiv = document.createElement('div');
      contentDiv.id = `${this.id}-description`;
      contentDiv.className = 'px-6 py-4';
      contentDiv.innerHTML = this.content;
      dialogElement.appendChild(contentDiv);
    }

    containerElement.appendChild(dialogElement);

    // Handle keyboard
    this.on('keydown', (e) => {
      if (keyboard.isEscape(e) && this.closeOnEscape) {
        this.close();
      }
    }, { once: false });

    this.backdrop = backdrop;
    this.dialogElement = dialogElement;
    this.focusTrap = createFocusTrap(dialogElement);

    return containerElement;
  }

  open() {
    if (!this.element) this.create();
    
    this.open = true;
    this.element.classList.remove('hidden');
    this.focusTrap.init();
    document.body.style.overflow = 'hidden';
  }

  close() {
    this.open = false;
    this.element?.classList.add('hidden');
    document.body.style.overflow = '';
    
    if (this.onClose) {
      this.onClose();
    }
  }

  toggle() {
    this.open ? this.close() : this.open();
  }

  setContent(content) {
    this.content = content;
    if (this.dialogElement) {
      const contentDiv = this.dialogElement.querySelector(`#${this.id}-description`);
      if (contentDiv) {
        contentDiv.innerHTML = content;
      }
    }
  }
}
