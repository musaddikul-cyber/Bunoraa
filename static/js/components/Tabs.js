/**
 * Tabs Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Tabs extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.tabs = options.tabs || []; // Array of {label, content, disabled}
    this.defaultTab = options.defaultTab || 0;
    this.activeTab = this.defaultTab;
    this.variant = options.variant || 'default'; // default, pills
    this.className = options.className || '';
    this.onChange = options.onChange || null;
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx('w-full', this.className)
    });

    // Tab buttons
    const tabsHeaderClasses = this.variant === 'pills' 
      ? 'flex gap-2 bg-gray-100 p-1 rounded-lg'
      : 'flex border-b border-gray-300';

    const headerElement = createElement('div', {
      className: tabsHeaderClasses,
      attrs: { role: 'tablist' }
    });

    this.tabs.forEach((tab, index) => {
      const isActive = index === this.activeTab;
      
      const tabButtonClasses = this.variant === 'pills'
        ? clsx(
            'px-4 py-2 rounded font-medium transition-colors duration-200',
            isActive ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-700 hover:text-gray-900',
            tab.disabled ? 'opacity-50 cursor-not-allowed' : ''
          )
        : clsx(
            'px-4 py-2 font-medium border-b-2 transition-colors duration-200 cursor-pointer',
            isActive ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-700 hover:text-gray-900',
            tab.disabled ? 'opacity-50 cursor-not-allowed' : ''
          );

      const tabButton = createElement('button', {
        className: tabButtonClasses,
        text: tab.label,
        attrs: {
          role: 'tab',
          'aria-selected': isActive,
          disabled: tab.disabled ? '' : null,
          'data-tab-index': index
        }
      });

      tabButton.addEventListener('click', () => {
        if (!tab.disabled) {
          this.selectTab(index);
        }
      });

      headerElement.appendChild(tabButton);
    });

    containerElement.appendChild(headerElement);

    // Tab content
    const contentElement = createElement('div', {
      className: 'mt-4',
      attrs: { role: 'tabpanel' }
    });

    this.tabs.forEach((tab, index) => {
      const panelClasses = index === this.activeTab ? 'block' : 'hidden';
      const panel = createElement('div', {
        className: panelClasses,
        html: tab.content,
        attrs: { 'data-panel-index': index }
      });
      contentElement.appendChild(panel);
    });

    containerElement.appendChild(contentElement);

    this.headerElement = headerElement;
    this.contentElement = contentElement;

    return containerElement;
  }

  selectTab(index) {
    if (index === this.activeTab) return;

    // Update buttons
    const buttons = this.headerElement.querySelectorAll('button');
    buttons.forEach((btn, i) => {
      const isActive = i === index;
      btn.setAttribute('aria-selected', isActive);
      
      if (this.variant === 'pills') {
        btn.className = clsx(
          'px-4 py-2 rounded font-medium transition-colors duration-200',
          isActive ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-700 hover:text-gray-900'
        );
      } else {
        btn.className = clsx(
          'px-4 py-2 font-medium border-b-2 transition-colors duration-200 cursor-pointer',
          isActive ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-700 hover:text-gray-900'
        );
      }
    });

    // Update panels
    const panels = this.contentElement.querySelectorAll('[data-panel-index]');
    panels.forEach((panel, i) => {
      panel.className = i === index ? 'block' : 'hidden';
    });

    this.activeTab = index;
    
    if (this.onChange) {
      this.onChange(index);
    }
  }

  getActiveTab() {
    return this.activeTab;
  }
}
