// Admin helper: fetch category suggestions from /api/v1/catalog/categories/classify/
// and let admin quickly add suggested categories to the Categories multi-select.

(function () {
  function $(sel, root) { return (root || document).querySelector(sel); }

  function findCategoryOptionsByName(selectEl, name) {
    const matches = [];
    for (const opt of selectEl.options) {
      if (opt.textContent.trim().toLowerCase() === name.trim().toLowerCase()) {
        matches.push(opt);
      }
    }
    return matches;
  }

  function createButton(text, cls) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = cls || 'button';
    btn.textContent = text;
    return btn;
  }

  function renderSuggestions(container, suggestions, categoriesSelect) {
    container.innerHTML = '';
    if (!suggestions || suggestions.length === 0) {
      const p = document.createElement('p');
      p.textContent = 'No suggestions.';
      container.appendChild(p);
      return;
    }

    suggestions.forEach(function (s) {
      const row = document.createElement('div');
      row.className = 'suggestion-row';
      row.style = 'display:flex;align-items:center;gap:8px;margin-bottom:6px;';

      const label = document.createElement('div');
      label.innerHTML = `<strong>${s.name}</strong> <span style="color: #666">(${(s.confidence*100).toFixed(0)}%)</span>`;
      row.appendChild(label);

      const addBtn = createButton('Add', 'button add-suggestion-btn');
      addBtn.addEventListener('click', function () {
        // Try to find matching option by name
        const matches = findCategoryOptionsByName(categoriesSelect, s.name);
        if (matches.length > 0) {
          matches.forEach(function (opt) { opt.selected = true; });
          // Trigger change event for any JS listeners
          const ev = new Event('change', { bubbles: true });
          categoriesSelect.dispatchEvent(ev);
        } else {
          // No option matched by name. Inform user to add manually.
          alert('No matching category option found for "' + s.name + '". You can search for it in the "Categories" widget.');
        }
      });

      row.appendChild(addBtn);
      container.appendChild(row);
    });
  }

  function init() {
    const form = document.querySelector('#product_form') || document.querySelector('form');
    if (!form) return;

    // Find inputs
    const nameInput = form.querySelector('[name="name"]');
    const descTextarea = form.querySelector('[name="description"]');
    const categoriesSelect = form.querySelector('[name="categories"]');
    if (!nameInput || !categoriesSelect) return;

    // Create UI
    const container = document.createElement('div');
    container.className = 'product-classifier-container';
    container.style = 'margin-top:8px;margin-bottom:12px;';

    const suggestBtn = createButton('Suggest categories', 'button btn-suggest-categories');
    suggestBtn.style.marginRight = '8px';

    const suggestionsBox = document.createElement('div');
    suggestionsBox.className = 'classifier-suggestions';
    suggestionsBox.style = 'margin-top:8px;padding:8px;border:1px solid #e5e7eb;border-radius:6px;background:#fff;max-width:640px;';

    container.appendChild(suggestBtn);
    container.appendChild(suggestionsBox);

    // Insert after categories field
    const categoriesWrapper = categoriesSelect.closest('.field-categories') || categoriesSelect.parentElement;
    if (categoriesWrapper && categoriesWrapper.parentNode) {
      categoriesWrapper.parentNode.insertBefore(container, categoriesWrapper.nextSibling);
    } else {
      form.appendChild(container);
    }

    suggestBtn.addEventListener('click', function () {
      const name = (nameInput.value || '').trim();
      const desc = (descTextarea && descTextarea.value) ? descTextarea.value.trim() : '';
      if (!name && !desc) {
        alert('Please provide a product name or description to get suggestions.');
        return;
      }
      suggestionsBox.innerHTML = '<em>Loading suggestions…</em>';
      const params = new URLSearchParams();
      if (name) params.append('name', name);
      if (desc) params.append('description', desc);
      fetch('/api/v1/catalog/categories/classify/?' + params.toString(), { credentials: 'same-origin' })
        .then(function (resp) {
          if (!resp.ok) throw new Error('Classification failed');
          return resp.json();
        })
        .then(function (data) {
          if (data && data.success && Array.isArray(data.data)) {
            renderSuggestions(suggestionsBox, data.data, categoriesSelect);
          } else {
            suggestionsBox.innerHTML = '<em>No suggestions.</em>';
          }
        })
        .catch(function (err) {
          suggestionsBox.innerHTML = '<em>Failed to fetch suggestions.</em>';
          console.error(err);
        });
    });
  }

  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
