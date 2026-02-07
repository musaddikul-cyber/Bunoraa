/**
 * Sidebar Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Sidebar extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.items = options.items || []; // {label, href, icon, active, onClick}
    this.width = options.width || '256px';
    this.collapsible = options.collapsible || false;
    this.collapsed = options.collapsed || false;
    this.className = options.className || '';
  }

  create() {
    const containerElement = super.create('aside', {
      className: clsx(
        'bg-gray-900 text-white transition-all duration-300',
        this.collapsed ? 'w-20' : 'w-64',
        this.className
      ),
      attrs: {
        style: this.collapsible ? '' : `width: ${this.width}`
      }
    });

    // Header with toggle
    if (this.collapsible) {
      const header = createElement('div', {
        className: 'p-4 border-b border-gray-700 flex items-center justify-between'
      });

      const title = createElement('span', {
        className: clsx('text-lg font-bold transition-all', this.collapsed ? 'hidden' : ''),
        text: 'Menu'
      });

      const toggleBtn = createElement('button', {
        className: 'p-2 hover:bg-gray-800 rounded transition-colors',
        html: this.collapsed ? '→' : '←'
      });

      toggleBtn.addEventListener('click', () => {
        this.toggleCollapse();
      });

      header.appendChild(title);
      header.appendChild(toggleBtn);
      containerElement.appendChild(header);
    }

    // Navigation items
    const navElement = createElement('nav', {
      className: 'p-4 space-y-2'
    });

    this.items.forEach(item => {
      const isActive = item.active || false;

      const linkElement = createElement('a', {
        className: clsx(
          'flex items-center gap-3 px-4 py-2 rounded transition-colors',
          'hover:bg-gray-800',
          isActive ? 'bg-blue-600' : ''
        ),
        text: item.label,
        attrs: {
          href: item.href || '#',
          title: this.collapsed ? item.label : ''
        }
      });

      if (item.icon) {
        const iconSpan = createElement('span', {
          className: 'w-5 h-5 flex-shrink-0',
          html: item.icon
        });
        linkElement.insertBefore(iconSpan, linkElement.firstChild);
      }

      if (this.collapsed) {
        const labelSpan = linkElement.querySelector('a span:last-child');
        if (labelSpan) {
          labelSpan.className = 'hidden';
        }
      }

      if (item.onClick) {
        linkElement.addEventListener('click', (e) => {
          e.preventDefault();
          item.onClick();
        });
      }

      navElement.appendChild(linkElement);
    });

    containerElement.appendChild(navElement);

    this.navElement = navElement;
    return containerElement;
  }

  toggleCollapse() {
    this.collapsed = !this.collapsed;
    
    if (this.element) {
      if (this.collapsed) {
        this.element.className = clsx(this.element.className.replace('w-64', 'w-20'));
      } else {
        this.element.className = clsx(this.element.className.replace('w-20', 'w-64'));
      }

      // Toggle label visibility
      const labels = this.element.querySelectorAll('a span:last-child');
      labels.forEach(label => {
        label.classList.toggle('hidden');
      });
    }
  }

  addItem(label, href, icon = null, onClick = null) {
    const item = { label, href, icon, onClick };
    this.items.push(item);

    if (this.navElement) {
      const linkElement = createElement('a', {
        className: 'flex items-center gap-3 px-4 py-2 rounded transition-colors hover:bg-gray-800',
        text: label,
        attrs: { href: href || '#' }
      });

      if (icon) {
        const iconSpan = createElement('span', {
          className: 'w-5 h-5 flex-shrink-0',
          html: icon
        });
        linkElement.insertBefore(iconSpan, linkElement.firstChild);
      }

      if (onClick) {
        linkElement.addEventListener('click', (e) => {
          e.preventDefault();
          onClick();
        });
      }

      this.navElement.appendChild(linkElement);
    }
  }
}
