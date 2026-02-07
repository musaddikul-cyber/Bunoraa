// Lazy hydration utility
// Usage: add `data-hydrate="components/Carousel.js#init"` to an element
// Module path is relative to static/js (e.g. components/Carousel.js)
// After the `#` is the exported initializer name (defaults to `init`).
const MODULE_REGISTRY = {
  // Components
  'components/Alert.js': () => import('../components/Alert.js'),
  'components/AlertDialog.js': () => import('../components/AlertDialog.js'),
  'components/Avatar.js': () => import('../components/Avatar.js'),
  'components/Badge.js': () => import('../components/Badge.js'),
  'components/Button.js': () => import('../components/Button.js'),
  'components/ButtonGroup.js': () => import('../components/ButtonGroup.js'),
  'components/Breadcrumb.js': () => import('../components/Breadcrumb.js'),
  'components/Card.js': () => import('../components/Card.js'),
  'components/Carousel.js': () => import('../components/Carousel.js'),
  'components/Chart.js': () => import('../components/Chart.js'),
  'components/Checkbox.js': () => import('../components/Checkbox.js'),
  'components/Collapsible.js': () => import('../components/Collapsible.js'),
  'components/Command.js': () => import('../components/Command.js'),
  'components/Combobox.js': () => import('../components/Combobox.js'),
  'components/ContextMenu.js': () => import('../components/ContextMenu.js'),
  'components/DatePicker.js': () => import('../components/DatePicker.js'),
  'components/Dialog.js': () => import('../components/Dialog.js'),
  'components/DropdownMenu.js': () => import('../components/DropdownMenu.js'),
  'components/Drawer.js': () => import('../components/Drawer.js'),
  'components/Empty.js': () => import('../components/Empty.js'),
  'components/Form.js': () => import('../components/Form.js'),
  'components/HoverCard.js': () => import('../components/HoverCard.js'),
  'components/Input.js': () => import('../components/Input.js'),
  'components/InputGroup.js': () => import('../components/InputGroup.js'),
  'components/InputOTP.js': () => import('../components/InputOTP.js'),
  'components/Item.js': () => import('../components/Item.js'),
  'components/Label.js': () => import('../components/Label.js'),
  'components/Kbd.js': () => import('../components/Kbd.js'),
  'components/NativeSelect.js': () => import('../components/NativeSelect.js'),
  'components/Tooltip.js': () => import('../components/Tooltip.js'),
  'components/Toggle.js': () => import('../components/Toggle.js'),
  'components/ToggleGroup.js': () => import('../components/ToggleGroup.js'),
  'components/Toast.js': () => import('../components/Toast.js'),

  // Add other components as needed
};

export function initLazyHydration({root = document, selector = '[data-hydrate]', threshold = 0.15} = {}) {
  if (typeof IntersectionObserver === 'undefined') return;
  const observer = new IntersectionObserver(async (entries, obs) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const el = entry.target;
      const spec = el.dataset.hydrate || '';
      const [modulePath, initName = 'init'] = spec.split('#');
      if (!modulePath) { obs.unobserve(el); continue; }
      try {
        let mod = null;
        const loader = MODULE_REGISTRY[modulePath];
        if (typeof loader === 'function') {
          mod = await loader();
        } else {
          // Not registered: skip importing arbitrary files to avoid bundler resolving non-JS assets
          throw new Error('Module not registered for lazy hydration: ' + modulePath);
        }
        const fn = mod[initName] || mod.default || null;
        if (typeof fn === 'function') {
          try { fn(el); } catch (e) { console.error('hydrate init failed', e); }
        }
      } catch (e) {
        console.error('lazy hydrate import failed for', modulePath, e);
      } finally {
        obs.unobserve(el);
      }
    }
  }, {threshold});
  root.querySelectorAll(selector).forEach(el => observer.observe(el));
}
