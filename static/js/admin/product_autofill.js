// Admin helper: request autofill suggestions for product metadata and optionally apply them to the form.
(function () {
  function $(sel, root) { return (root || document).querySelector(sel); }

  function createButton(text, cls) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = cls || 'button';
    btn.textContent = text;
    return btn;
  }

  function renderSuggestion(container, suggestion, fields) {
    container.innerHTML = '';

    const title = document.createElement('div');
    title.innerHTML = `<strong>Title suggestion:</strong> ${suggestion.name || ''}`;
    container.appendChild(title);

    const desc = document.createElement('div');
    desc.innerHTML = `<strong>Description:</strong> ${suggestion.short_description || ''}`;
    container.appendChild(desc);

    const tags = document.createElement('div');
    tags.innerHTML = `<strong>Tags:</strong> ${(suggestion.tags || []).join(', ')}`;
    container.appendChild(tags);

    const cat = document.createElement('div');
    const cs = (suggestion.category_suggestions || []).map(s => `${s.code} (${(s.confidence*100).toFixed(0)}%)`).join(', ');
    cat.innerHTML = `<strong>Category suggestions:</strong> ${cs}`;
    container.appendChild(cat);

    const applyBtn = createButton('Apply suggestion', 'button apply-suggestion-btn');
    applyBtn.addEventListener('click', function () {
      if (fields.name) fields.name.value = suggestion.name || fields.name.value;
      if (fields.short_description) fields.short_description.value = suggestion.short_description || '';
      if (fields.tags) fields.tags.value = (suggestion.tags || []).join(', ');
      // Optionally, we could set category selection if matched by slug/code
      if (suggestion.category_suggestions && suggestion.category_suggestions.length > 0 && fields.categories) {
        const code = suggestion.category_suggestions[0].code;
        // Try to match by option text or value
        for (const opt of fields.categories.options) {
          if (opt.textContent.trim().toLowerCase() === code.toLowerCase() || opt.value === code || opt.textContent.trim().toLowerCase() === (code.split('-').join(' '))) {
            opt.selected = true;
            fields.categories.dispatchEvent(new Event('change', {bubbles:true}));
            break;
          }
        }
      }
    });
    container.appendChild(applyBtn);
  }

  function init() {
    const form = document.querySelector('#product_form') || document.querySelector('form');
    if (!form) return;

    const nameInput = form.querySelector('[name="name"]');
    const descTextarea = form.querySelector('[name="description"]');
    const tagsInput = form.querySelector('[name="tags"]');
    const categoriesSelect = form.querySelector('[name="categories"]');
    const imagesInput = form.querySelector('[name="images"]'); // may be multiple inputs depending on admin

    if (!nameInput) return;

    const container = document.createElement('div');
    container.style = 'margin-top:8px;margin-bottom:12px;';
    const suggestBtn = createButton('Auto-fill from uploads', 'button btn-autofill');
    suggestBtn.style.marginRight = '8px';
    const suggestionsBox = document.createElement('div');
    suggestionsBox.className = 'autofill-suggestions';
    suggestionsBox.style = 'margin-top:8px;padding:8px;border:1px solid #e5e7eb;border-radius:6px;background:#fff;max-width:640px;';

    container.appendChild(suggestBtn);
    container.appendChild(suggestionsBox);

    // Insert after categories field (if present) or at top of form
    const categoriesWrapper = categoriesSelect ? (categoriesSelect.closest('.field-categories') || categoriesSelect.parentElement) : nameInput.parentElement;
    categoriesWrapper.parentNode.insertBefore(container, categoriesWrapper.nextSibling);

    suggestBtn.addEventListener('click', function () {
      suggestionsBox.innerHTML = '<em>Loading suggestions…</em>';
      // Collect data to send
      const data = new FormData();
      if (nameInput && nameInput.value) data.append('name', nameInput.value);
      if (descTextarea && descTextarea.value) data.append('description', descTextarea.value);
      // Add image filenames if file inputs are present
      const imageFiles = [];
      const fileInputs = form.querySelectorAll('input[type=file]');
      fileInputs.forEach(function (fi) {
        for (const f of fi.files) {
          imageFiles.push(f.name);
        }
      });
      // Also include any existing image filenames in DOM (admin may render previews with data attributes)
      const previewImgs = form.querySelectorAll('img.admin-image-preview');
      previewImgs.forEach(function (img) {
        const src = img.getAttribute('src');
        if (src) {
          const parts = src.split('/');
          imageFiles.push(parts[parts.length-1]);
        }
      });

      if (imageFiles.length > 0) {
        // attach as JSON array field
        data.append('image_filenames', JSON.stringify(imageFiles));
      }

      fetch('/api/v1/catalog/products/suggest/', {
        method: 'POST',
        headers: {
          'X-CSRFToken': (function(){const c=document.cookie.match(/(^|;)\s*csrftoken\s*=\s*([^;]+)/); return c?c.pop():''})()
        },
        body: data,
        credentials: 'same-origin'
      }).then(function (resp) {
        if (!resp.ok) throw new Error('Suggest failed');
        return resp.json();
      }).then(function (res) {
        if (res && res.success && Array.isArray(res.data) && res.data.length > 0) {
          // use first suggestion
          renderSuggestion(suggestionsBox, res.data[0], {name: nameInput, short_description: descTextarea, tags: tagsInput, categories: categoriesSelect});
        } else {
          suggestionsBox.innerHTML = '<em>No suggestions available</em>';
        }
      }).catch(function (err) {
        suggestionsBox.innerHTML = '<em>Failed to fetch suggestions.</em>';
        console.error(err);
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();