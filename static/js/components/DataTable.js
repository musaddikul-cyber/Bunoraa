/**
 * Data Table Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class DataTable extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.columns = options.columns || []; // {header, key, sortable}
    this.rows = options.rows || [];
    this.selectable = options.selectable || false;
    this.sortBy = options.sortBy || null;
    this.sortOrder = options.sortOrder || 'asc';
    this.className = options.className || '';
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx('w-full border border-gray-200 rounded-lg overflow-hidden', this.className)
    });

    const tableElement = document.createElement('table');
    tableElement.className = 'w-full';

    // Header
    const thead = document.createElement('thead');
    thead.className = 'bg-gray-50 border-b border-gray-200';

    const headerRow = document.createElement('tr');

    if (this.selectable) {
      const th = document.createElement('th');
      th.className = 'p-3 text-left';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.className = 'rounded';
      th.appendChild(checkbox);
      headerRow.appendChild(th);
    }

    this.columns.forEach(col => {
      const th = document.createElement('th');
      th.className = 'px-6 py-3 text-left text-sm font-semibold text-gray-900';

      let content = col.header;
      if (col.sortable) {
        const btn = document.createElement('button');
        btn.className = 'flex items-center gap-2 hover:text-gray-600';
        btn.textContent = col.header;
        btn.addEventListener('click', () => this.sort(col.key));
        content = btn;
      }

      if (typeof content === 'string') {
        th.textContent = content;
      } else {
        th.appendChild(content);
      }

      headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    tableElement.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');

    this.rows.forEach(row => {
      const tr = document.createElement('tr');
      tr.className = 'border-b border-gray-200 hover:bg-gray-50 transition-colors';

      if (this.selectable) {
        const td = document.createElement('td');
        td.className = 'p-3';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'rounded';
        td.appendChild(checkbox);
        tr.appendChild(td);
      }

      this.columns.forEach(col => {
        const td = document.createElement('td');
        td.className = 'px-6 py-3 text-sm text-gray-700';
        td.textContent = row[col.key] || '';
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });

    tableElement.appendChild(tbody);
    containerElement.appendChild(tableElement);

    this.tableElement = tableElement;

    return containerElement;
  }

  sort(key) {
    if (this.sortBy === key) {
      this.sortOrder = this.sortOrder === 'asc' ? 'desc' : 'asc';
    } else {
      this.sortBy = key;
      this.sortOrder = 'asc';
    }

    this.rows.sort((a, b) => {
      const aValue = a[key];
      const bValue = b[key];

      if (this.sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    // Refresh
    if (this.element && this.element.parentNode) {
      const newElement = this.create();
      this.element.parentNode.replaceChild(newElement, this.element);
      this.element = newElement;
    }
  }

  addRow(row) {
    this.rows.push(row);
    this.refresh();
  }

  removeRow(index) {
    this.rows.splice(index, 1);
    this.refresh();
  }

  refresh() {
    if (this.element && this.element.parentNode) {
      const newElement = this.create();
      this.element.parentNode.replaceChild(newElement, this.element);
      this.element = newElement;
    }
  }

  getSelectedRows() {
    const checkboxes = this.element.querySelectorAll('tbody input[type="checkbox"]:checked');
    return Array.from(checkboxes).map((cb, index) => this.rows[index]);
  }
}
