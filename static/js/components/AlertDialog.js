/**
 * Alert Dialog Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createBackdrop, createFocusTrap } from './utils.js';

export class AlertDialog extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.title = options.title || '';
    this.message = options.message || '';
    this.confirmText = options.confirmText || 'Confirm';
    this.cancelText = options.cancelText || 'Cancel';
    this.type = options.type || 'warning'; // warning, danger, info
    this.onConfirm = options.onConfirm || null;
    this.onCancel = options.onCancel || null;
    this.open = options.open || false;
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx(
        'fixed inset-0 z-50 flex items-center justify-center',
        this.open ? '' : 'hidden'
      ),
      attrs: {
        role: 'alertdialog',
        'aria-modal': 'true',
        'aria-labelledby': `${this.id}-title`,
        'aria-describedby': `${this.id}-message`
      }
    });

    // Backdrop
    const backdrop = createBackdrop();
    containerElement.appendChild(backdrop);

    // Dialog content
    const dialogElement = document.createElement('div');
    dialogElement.className = 'bg-white rounded-lg shadow-lg relative z-50 w-full max-w-md mx-4';

    // Header
    const header = document.createElement('div');
    header.className = 'px-6 py-4 border-b border-gray-200';

    const title = document.createElement('h2');
    title.id = `${this.id}-title`;
    title.className = 'text-lg font-semibold text-gray-900';
    title.textContent = this.title;
    header.appendChild(title);

    dialogElement.appendChild(header);

    // Message
    const messageDiv = document.createElement('div');
    messageDiv.id = `${this.id}-message`;
    messageDiv.className = 'px-6 py-4 text-gray-700';
    messageDiv.textContent = this.message;
    dialogElement.appendChild(messageDiv);

    // Footer with buttons
    const footer = document.createElement('div');
    footer.className = 'px-6 py-4 bg-gray-50 border-t border-gray-200 flex gap-3 justify-end rounded-b-lg';

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-100 transition-colors duration-200';
    cancelBtn.textContent = this.cancelText;
    cancelBtn.addEventListener('click', () => this.handleCancel());
    footer.appendChild(cancelBtn);

    const confirmBtn = document.createElement('button');
    const confirmBtnClasses = this.type === 'danger'
      ? 'bg-red-600 text-white hover:bg-red-700'
      : 'bg-blue-600 text-white hover:bg-blue-700';
    confirmBtn.className = clsx('px-4 py-2 rounded-md transition-colors duration-200', confirmBtnClasses);
    confirmBtn.textContent = this.confirmText;
    confirmBtn.addEventListener('click', () => this.handleConfirm());
    footer.appendChild(confirmBtn);

    dialogElement.appendChild(footer);
    containerElement.appendChild(dialogElement);

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
  }

  handleConfirm() {
    if (this.onConfirm) {
      this.onConfirm();
    }
    this.close();
  }

  handleCancel() {
    if (this.onCancel) {
      this.onCancel();
    }
    this.close();
  }
}
