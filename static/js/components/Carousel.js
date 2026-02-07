/**
 * Carousel Component
 */

import { BaseComponent } from './BaseComponent.js';
import { clsx, createElement } from './utils.js';

export class Carousel extends BaseComponent {
  constructor(options = {}) {
    super(options);
    
    this.items = options.items || []; // Array of {src, alt, title}
    this.autoplay = options.autoplay || true;
    this.interval = options.interval || 5000;
    this.className = options.className || '';
    this.currentIndex = 0;
    this.autoplayTimer = null;
  }

  create() {
    const containerElement = super.create('div', {
      className: clsx('relative w-full bg-black rounded-lg overflow-hidden', this.className)
    });

    // Slides
    const slidesElement = createElement('div', {
      className: 'relative w-full h-96 overflow-hidden'
    });

    this.items.forEach((item, index) => {
      const slide = createElement('div', {
        className: clsx(
          'absolute inset-0 transition-opacity duration-500',
          index === this.currentIndex ? 'opacity-100' : 'opacity-0'
        )
      });

      const img = document.createElement('img');
      img.src = item.src;
      img.alt = item.alt || '';
      img.className = 'w-full h-full object-cover';

      slide.appendChild(img);

      if (item.title) {
        const titleElement = createElement('div', {
          className: 'absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-4',
          html: `<p class="text-white font-semibold">${item.title}</p>`
        });
        slide.appendChild(titleElement);
      }

      slidesElement.appendChild(slide);
    });

    containerElement.appendChild(slidesElement);

    // Navigation buttons
    const prevBtn = createElement('button', {
      className: 'absolute left-4 top-1/2 transform -translate-y-1/2 z-10 bg-black bg-opacity-50 text-white p-2 rounded-full hover:bg-opacity-75 transition-all',
      html: '❮'
    });

    prevBtn.addEventListener('click', () => {
      this.previous();
    });

    const nextBtn = createElement('button', {
      className: 'absolute right-4 top-1/2 transform -translate-y-1/2 z-10 bg-black bg-opacity-50 text-white p-2 rounded-full hover:bg-opacity-75 transition-all',
      html: '❯'
    });

    nextBtn.addEventListener('click', () => {
      this.next();
    });

    containerElement.appendChild(prevBtn);
    containerElement.appendChild(nextBtn);

    // Dots
    const dotsElement = createElement('div', {
      className: 'absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-2 z-10'
    });

    this.items.forEach((_, index) => {
      const dot = createElement('button', {
        className: clsx(
          'w-2 h-2 rounded-full transition-all',
          index === this.currentIndex ? 'bg-white w-8' : 'bg-gray-500'
        ),
        attrs: { 'data-index': index }
      });

      dot.addEventListener('click', () => {
        this.goTo(index);
      });

      dotsElement.appendChild(dot);
    });

    containerElement.appendChild(dotsElement);

    this.slidesElement = slidesElement;
    this.dotsElement = dotsElement;

    if (this.autoplay) {
      this.startAutoplay();
    }

    return containerElement;
  }

  next() {
    this.currentIndex = (this.currentIndex + 1) % this.items.length;
    this.updateSlides();
  }

  previous() {
    this.currentIndex = (this.currentIndex - 1 + this.items.length) % this.items.length;
    this.updateSlides();
  }

  goTo(index) {
    this.currentIndex = index;
    this.updateSlides();
    if (this.autoplay) {
      this.resetAutoplay();
    }
  }

  updateSlides() {
    const slides = this.slidesElement.querySelectorAll('div');
    const dots = this.dotsElement.querySelectorAll('button');

    slides.forEach((slide, index) => {
      slide.className = clsx(
        'absolute inset-0 transition-opacity duration-500',
        index === this.currentIndex ? 'opacity-100' : 'opacity-0'
      );
    });

    dots.forEach((dot, index) => {
      dot.className = clsx(
        'w-2 h-2 rounded-full transition-all',
        index === this.currentIndex ? 'bg-white w-8' : 'bg-gray-500'
      );
    });
  }

  startAutoplay() {
    this.autoplayTimer = setInterval(() => {
      this.next();
    }, this.interval);
  }

  stopAutoplay() {
    if (this.autoplayTimer) {
      clearInterval(this.autoplayTimer);
    }
  }

  resetAutoplay() {
    this.stopAutoplay();
    if (this.autoplay) {
      this.startAutoplay();
    }
  }

  destroy() {
    this.stopAutoplay();
    super.destroy();
  }
}
