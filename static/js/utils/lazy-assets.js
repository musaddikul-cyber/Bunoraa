/* Lazy asset loading + progressive rendering helpers */
(function () {
  'use strict';

  function setDefaultImageAttrs(img) {
    if (!img.hasAttribute('loading') && !img.dataset.eager) {
      img.setAttribute('loading', 'lazy');
    }
    if (!img.hasAttribute('decoding')) {
      img.setAttribute('decoding', 'async');
    }
  }

  function lazyLoadImages() {
    var hero = document.querySelector('main img');
    if (hero) {
      hero.dataset.eager = 'true';
      if (!hero.hasAttribute('loading')) hero.setAttribute('loading', 'eager');
      if (!hero.hasAttribute('fetchpriority')) hero.setAttribute('fetchpriority', 'high');
    }

    var imgs = document.querySelectorAll('img');
    imgs.forEach(setDefaultImageAttrs);

    document.querySelectorAll('iframe').forEach(function (frame) {
      if (!frame.hasAttribute('loading')) {
        frame.setAttribute('loading', 'lazy');
      }
    });

    var lazyNodes = document.querySelectorAll('img[data-src], img[data-srcset], source[data-srcset], [data-bg-src]');
    if (lazyNodes.length === 0) return;

    var onLoad = function (el) {
      if (el.tagName === 'IMG') {
        if (el.dataset.src) {
          el.src = el.dataset.src;
          el.removeAttribute('data-src');
        }
        if (el.dataset.srcset) {
          el.srcset = el.dataset.srcset;
          el.removeAttribute('data-srcset');
        }
        var reveal = function () {
          el.style.opacity = '1';
          el.classList.add('is-loaded');
        };
        if (el.complete) {
          reveal();
        } else {
          el.addEventListener('load', reveal, { once: true });
        }
      } else if (el.tagName === 'SOURCE') {
        if (el.dataset.srcset) {
          el.srcset = el.dataset.srcset;
          el.removeAttribute('data-srcset');
        }
      } else {
        var bg = el.getAttribute('data-bg-src');
        if (bg) {
          el.style.backgroundImage = 'url(' + bg + ')';
          el.removeAttribute('data-bg-src');
        }
      }
    };

    if (!('IntersectionObserver' in window)) {
      lazyNodes.forEach(onLoad);
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        onLoad(entry.target);
        observer.unobserve(entry.target);
      });
    }, { rootMargin: '200px 0px' });

    lazyNodes.forEach(function (node) { observer.observe(node); });
  }

  function lazyRevealSections() {
    var autoSections = document.querySelectorAll('main section, main article');
    autoSections.forEach(function (section) {
      if (section.getAttribute('data-lazy-section') === 'false') return;
      if (!section.hasAttribute('data-lazy-section')) {
        section.setAttribute('data-lazy-section', 'true');
      }
    });

    var sections = document.querySelectorAll('[data-lazy-section]');
    if (sections.length === 0 || !('IntersectionObserver' in window)) return;

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      });
    }, { rootMargin: '200px 0px' });

    sections.forEach(function (section) {
      section.classList.add('is-lazy');
      observer.observe(section);
    });
  }

  function loadScriptOnce(src) {
    if (!src) return;
    if (document.querySelector('script[data-lazy-src=\"' + src + '\"]')) return;
    var script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.setAttribute('data-lazy-src', src);
    document.body.appendChild(script);
  }

  function lazyLoadML() {
    var base = (window.BUNORAA_STATIC_URL || '/static/');
    if (base && base[base.length - 1] !== '/') base += '/';
    var recSrc = base + 'js/ml-recommendations.js';
    var trackSrc = base + 'js/ml-tracking.js';

    var needsML = document.querySelector('[data-ml-recommendations], #ml-cart-recommendations, #ml-similar-products, #ml-fbt');
    if (needsML) {
      loadScriptOnce(recSrc);
    }

    var loadTracking = function () { loadScriptOnce(trackSrc); };
    if ('requestIdleCallback' in window) {
      requestIdleCallback(loadTracking, { timeout: 3000 });
    } else {
      setTimeout(loadTracking, 2000);
    }
  }

  function init() {
    lazyLoadImages();
    lazyRevealSections();
    lazyLoadML();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
