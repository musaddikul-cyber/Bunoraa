/**
 * Form Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Form extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.fields = options.fields || []; // Array of field configs
    this.onSubmit = options.onSubmit || null;
    this.submitText = options.submitText || 'Submit';
    this.className = options.className || '';
  }

  create() {
    const formElement = super.create('form', {
      className: clsx('space-y-6', this.className)
    });

    formElement.addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleSubmit();
    });

    // Render fields
    this.fieldElements = {};

    this.fields.forEach(field => {
      const fieldContainer = createElement('div', {
        className: 'space-y-2'
      });

      // Label
      if (field.label) {
        const label = createElement('label', {
          className: 'block text-sm font-medium text-gray-700',
          text: field.label,
          attrs: { for: field.name }
        });
        fieldContainer.appendChild(label);
      }

      // Input
      const input = document.createElement(field.type === 'textarea' ? 'textarea' : 'input');
      input.id = field.name;
      input.name = field.name;
      input.className = clsx(
        'w-full px-3 py-2 border border-gray-300 rounded-md',
        'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent'
      );

      if (field.type !== 'textarea') {
        input.type = field.type || 'text';
      }

      if (field.placeholder) input.placeholder = field.placeholder;
      if (field.required) input.required = true;
      if (field.disabled) input.disabled = true;

      input.value = field.value || '';

      fieldContainer.appendChild(input);

      // Error message
      const errorDiv = createElement('div', {
        className: 'text-sm text-red-600 hidden',
        attrs: { 'data-error': field.name }
      });
      fieldContainer.appendChild(errorDiv);

      formElement.appendChild(fieldContainer);
      this.fieldElements[field.name] = input;
    });

    // Submit button
    const submitBtn = createElement('button', {
      className: clsx(
        'w-full px-4 py-2 bg-blue-600 text-white font-medium rounded',
        'hover:bg-blue-700 transition-colors duration-200'
      ),
      text: this.submitText,
      attrs: { type: 'submit' }
    });

    formElement.appendChild(submitBtn);

    return formElement;
  }

  handleSubmit() {
    const formData = {};

    Object.entries(this.fieldElements).forEach(([name, input]) => {
      formData[name] = input.value;
    });

    if (this.onSubmit) {
      this.onSubmit(formData);
    }
  }

  getValues() {
    const values = {};
    Object.entries(this.fieldElements).forEach(([name, input]) => {
      values[name] = input.value;
    });
    return values;
  }

  setValues(values) {
    Object.entries(values).forEach(([name, value]) => {
      if (this.fieldElements[name]) {
        this.fieldElements[name].value = value;
      }
    });
  }

  setError(fieldName, message) {
    const errorDiv = this.element.querySelector(`[data-error="${fieldName}"]`);
    if (errorDiv) {
      errorDiv.textContent = message;
      errorDiv.classList.remove('hidden');
    }
  }

  clearError(fieldName) {
    const errorDiv = this.element.querySelector(`[data-error="${fieldName}"]`);
    if (errorDiv) {
      errorDiv.textContent = '';
      errorDiv.classList.add('hidden');
    }
  }

  reset() {
    if (this.element) {
      this.element.reset();
    }
  }
}
