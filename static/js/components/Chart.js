/**
 * Chart Component (Simple bar/line chart)
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Chart extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.type = options.type || 'bar'; // bar, line
    this.data = options.data || []; // Array of {label, value}
    this.title = options.title || '';
    this.height = options.height || '300px';
    this.className = options.className || '';
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx('bg-white rounded-lg border border-gray-200 p-4', this.className)
    });

    // Title
    if (this.title) {
      const titleElement = createElement('h3', {
        className: 'text-lg font-semibold text-gray-900 mb-4',
        text: this.title
      });
      containerElement.appendChild(titleElement);
    }

    // Canvas or SVG
    const svgElement = this.createSVG();
    containerElement.appendChild(svgElement);

    return containerElement;
  }

  createSVG() {
    const maxValue = Math.max(...this.data.map(d => d.value));
    const width = 400;
    const height = 200;
    const padding = 40;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('class', 'w-full');

    if (this.type === 'bar') {
      const barWidth = (width - padding * 2) / this.data.length;

      this.data.forEach((d, i) => {
        const barHeight = (d.value / maxValue) * (height - padding * 2);
        const x = padding + i * barWidth + barWidth * 0.25;
        const y = height - padding - barHeight;

        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', barWidth * 0.5);
        rect.setAttribute('height', barHeight);
        rect.setAttribute('fill', '#3b82f6');
        rect.setAttribute('class', 'hover:fill-blue-700 transition-colors');

        svg.appendChild(rect);

        // Label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x + barWidth * 0.25);
        text.setAttribute('y', height - 10);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '12');
        text.setAttribute('fill', '#6b7280');
        text.textContent = d.label;
        svg.appendChild(text);
      });
    }

    return svg;
  }

  setData(data) {
    this.data = data;
    if (this.element && this.element.parentNode) {
      const newElement = this.create();
      this.element.parentNode.replaceChild(newElement, this.element);
      this.element = newElement;
    }
  }
}
