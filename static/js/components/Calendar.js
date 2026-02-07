/**
 * Calendar Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Calendar extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.selected = options.selected || new Date();
    this.month = this.selected.getMonth();
    this.year = this.selected.getFullYear();
    this.className = options.className || '';
    this.onChange = options.onChange || null;
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx('bg-white rounded-lg border border-gray-200 p-4', this.className)
    });

    // Header with month/year navigation
    const header = createElement('div', {
      className: 'flex items-center justify-between mb-4'
    });

    const prevBtn = createElement('button', {
      className: 'p-2 hover:bg-gray-100 rounded transition-colors',
      html: '←'
    });
    prevBtn.addEventListener('click', () => this.previousMonth());

    const monthYear = createElement('div', {
      className: 'text-lg font-semibold',
      text: this.getMonthYearString()
    });

    const nextBtn = createElement('button', {
      className: 'p-2 hover:bg-gray-100 rounded transition-colors',
      html: '→'
    });
    nextBtn.addEventListener('click', () => this.nextMonth());

    header.appendChild(prevBtn);
    header.appendChild(monthYear);
    header.appendChild(nextBtn);
    containerElement.appendChild(header);

    // Weekdays
    const weekdaysContainer = createElement('div', {
      className: 'grid grid-cols-7 gap-2 mb-2'
    });
    ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].forEach(day => {
      const dayElement = createElement('div', {
        className: 'text-center text-sm font-semibold text-gray-600',
        text: day
      });
      weekdaysContainer.appendChild(dayElement);
    });
    containerElement.appendChild(weekdaysContainer);

    // Days
    const daysContainer = createElement('div', {
      className: 'grid grid-cols-7 gap-2'
    });

    const firstDay = new Date(this.year, this.month, 1).getDay();
    const daysInMonth = new Date(this.year, this.month + 1, 0).getDate();

    // Empty cells before first day
    for (let i = 0; i < firstDay; i++) {
      daysContainer.appendChild(createElement('div'));
    }

    // Days
    for (let day = 1; day <= daysInMonth; day++) {
      const dayElement = createElement('button', {
        className: clsx(
          'p-2 rounded text-center hover:bg-blue-100 transition-colors',
          this.isSelectedDay(day) ? 'bg-blue-600 text-white' : 'hover:bg-gray-100'
        ),
        text: day.toString()
      });

      dayElement.addEventListener('click', () => {
        this.selectDay(day, monthYear);
      });

      daysContainer.appendChild(dayElement);
    }

    containerElement.appendChild(daysContainer);

    this.monthYearElement = monthYear;
    this.daysContainer = daysContainer;

    return containerElement;
  }

  getMonthYearString() {
    const months = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December'];
    return `${months[this.month]} ${this.year}`;
  }

  isSelectedDay(day) {
    return this.selected.getDate() === day && 
           this.selected.getMonth() === this.month && 
           this.selected.getFullYear() === this.year;
  }

  selectDay(day, monthYearElement) {
    this.selected = new Date(this.year, this.month, day);
    if (this.onChange) {
      this.onChange(this.selected);
    }
    // Refresh calendar
    if (this.element && this.element.parentNode) {
      const newElement = this.create();
      this.element.parentNode.replaceChild(newElement, this.element);
      this.element = newElement;
    }
  }

  previousMonth() {
    this.month--;
    if (this.month < 0) {
      this.month = 11;
      this.year--;
    }
    this.refresh();
  }

  nextMonth() {
    this.month++;
    if (this.month > 11) {
      this.month = 0;
      this.year++;
    }
    this.refresh();
  }

  refresh() {
    if (this.element && this.element.parentNode) {
      const newElement = this.create();
      this.element.parentNode.replaceChild(newElement, this.element);
      this.element = newElement;
    }
  }
}
