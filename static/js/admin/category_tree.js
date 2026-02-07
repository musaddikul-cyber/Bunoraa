(function() {
  function buildTreeFromOptions(select) {
    const options = Array.from(select.querySelectorAll('option'));
    const nodes = {};

    options.forEach(opt => {
      const id = opt.value;
      const parent = opt.getAttribute('data-parent') || '';
      const depth = parseInt(opt.getAttribute('data-depth') || '0', 10);
      nodes[id] = { id, parent, label: opt.textContent.trim(), option: opt, children: [], depth, selected: opt.selected };
    });

    const root = [];
    Object.values(nodes).forEach(node => {
      if (node.parent && nodes[node.parent]) {
        nodes[node.parent].children.push(node);
      } else {
        root.push(node);
      }
    });

    function createList(items) {
      const ul = document.createElement('ul');
      ul.className = 'category-tree';
      items.forEach(item => {
        const li = document.createElement('li');
        li.className = 'category-item';
        li.dataset.id = item.id;
        li.dataset.depth = item.depth;
        li.classList.add('pl-1');

        const row = document.createElement('div');
        row.className = 'category-row';
        row.tabIndex = 0; // make row focusable for keyboard navigation
        row.classList.add('flex','items-center','gap-2','rounded','p-1','hover:bg-gray-50','dark:hover:bg-slate-800','focus:outline-none','focus:ring-2','focus:ring-indigo-400');

        // Toggle / caret
        if (item.children.length > 0) {
          const toggle = document.createElement('button');
          toggle.type = 'button';
          toggle.className = 'toggle-children';
          toggle.setAttribute('aria-expanded', 'false');
          toggle.innerHTML = '<span class="caret">&gt;</span>';
          toggle.classList.add('w-5','h-5','flex','items-center','justify-center','text-sm','text-gray-500','dark:text-slate-300');
          toggle.addEventListener('click', function(e) {
            e.stopPropagation();
            const expanded = toggle.getAttribute('aria-expanded') === 'true';
            toggle.setAttribute('aria-expanded', expanded ? 'false' : 'true');
            if (childUl) {
              childUl.classList.toggle('collapsed');
            }
            toggle.querySelector('.caret').textContent = expanded ? '>' : 'v';
          });
          row.appendChild(toggle);
        } else {
          const spacer = document.createElement('span');
          spacer.className = 'toggle-spacer';
          spacer.textContent = ' ';
          spacer.classList.add('w-5');
          row.appendChild(spacer);
        }

        // (No icon and no checkbox for single-select behavior)
        // Render label as simple text — clicking/selecting a row will select the category
        const label = document.createElement('div');
        label.className = 'category-label';
        label.textContent = item.label;
        label.classList.add('flex','items-center','gap-2','text-sm','text-gray-700','dark:text-slate-200');

        row.appendChild(label);

        // Mark selected if this option is currently selected
        if (item.selected) {
          row.classList.add('category-selected','bg-indigo-50','dark:bg-indigo-900/20');
        }
        li.appendChild(row);

        if (item.children.length > 0) {
          var childUl = createList(item.children);
          childUl.classList.add('collapsed');
          li.appendChild(childUl);
        }

        ul.appendChild(li);
      });
      return ul;
    }

    return { rootList: createList(root), nodes };
  }


  document.addEventListener('DOMContentLoaded', function() {
    const select = document.querySelector('select[name="categories"]');
    if (!select) return;

    // Hide original select but keep it in the DOM for form submission
    select.style.display = 'none';

    const wrapper = document.createElement('div');
    wrapper.className = 'category-tree-wrapper';
    wrapper.classList.add('bg-white','dark:bg-slate-900','rounded-md','p-2','border','border-gray-200','dark:border-slate-700','text-gray-700','dark:text-slate-200');

    const controls = document.createElement('div');
    controls.className = 'category-tree-controls';
    controls.classList.add('flex','gap-2','mb-2');

    const expandAll = document.createElement('button');
    expandAll.type = 'button';
    expandAll.textContent = 'Expand all';
    expandAll.classList.add('px-2','py-1','text-sm','rounded','bg-transparent','border','border-gray-200','dark:border-slate-700','text-gray-700','dark:text-slate-200','hover:bg-gray-100','dark:hover:bg-slate-800');
    expandAll.addEventListener('click', function() {
      wrapper.querySelectorAll('ul.category-tree ul').forEach(u => u.classList.remove('collapsed'));
      wrapper.querySelectorAll('button.toggle-children').forEach(b => b.setAttribute('aria-expanded', 'true'));
      wrapper.querySelectorAll('.toggle-children .caret').forEach(c => c.textContent = 'v');
    });

    const collapseAll = document.createElement('button');
    collapseAll.type = 'button';
    collapseAll.textContent = 'Collapse all';
    collapseAll.classList.add('px-2','py-1','text-sm','rounded','bg-transparent','border','border-gray-200','dark:border-slate-700','text-gray-700','dark:text-slate-200','hover:bg-gray-100','dark:hover:bg-slate-800');
    collapseAll.addEventListener('click', function() {
      wrapper.querySelectorAll('ul.category-tree ul').forEach(u => u.classList.add('collapsed'));
      wrapper.querySelectorAll('button.toggle-children').forEach(b => b.setAttribute('aria-expanded', 'false'));
      wrapper.querySelectorAll('.toggle-children .caret').forEach(c => c.textContent = '>');
    });

    controls.appendChild(expandAll);
    controls.appendChild(collapseAll);
    wrapper.appendChild(controls);

    const { rootList } = buildTreeFromOptions(select);
    wrapper.appendChild(rootList);
    select.parentNode.insertBefore(wrapper, select.nextSibling);

    // Single-select behavior: clicking a row selects it and deselects others
    function selectCategoryById(id) {
      // Deselect all options
      Array.from(select.options).forEach(opt => { opt.selected = false; });
      if (!id) return;
      const opt = select.querySelector(`option[value="${id}"]`);
      if (opt) opt.selected = true;

      // Update visual classes
      wrapper.querySelectorAll('li.category-item').forEach(li => {
        const r = li.querySelector('.category-row');
        const thisId = li.dataset.id;
        const isSelected = (thisId === id);
        r.classList.toggle('category-selected', isSelected);
        r.classList.toggle('bg-indigo-50', isSelected);
        r.classList.toggle('dark:bg-indigo-900/20', isSelected);
      });

      // Dispatch change event
      select.dispatchEvent(new Event('change', { bubbles: true }));
      // Update summary text (only if dropdownContainer is initialized)
      if (typeof dropdownContainer !== 'undefined' && dropdownContainer) {
        const summaryText = dropdownContainer.querySelector('.category-summary');
        if (summaryText) summaryText.textContent = buildSummary();
      }
    }

    // Initialize selected class states based on underlying <select>
    const initiallySelected = select.querySelector('option:checked');
    if (initiallySelected) {
      selectCategoryById(initiallySelected.value);
    }

    // Click / label handling and keyboard navigation
    wrapper.addEventListener('click', function(e) {
      const row = e.target.closest('.category-row');
      if (!row) return;
      const li = row.closest('li.category-item');
      const id = li.dataset.id;
      // If click target was toggle button or caret, ignore (handled elsewhere)
      if (e.target.closest('.toggle-children')) return;
      // Select the category (single select)
      selectCategoryById(id);
    });

    // Keyboard navigation: arrow keys & space/enter to select
    wrapper.addEventListener('keydown', function(e) {
      const row = e.target.closest('.category-row');
      if (!row) return;
      if (e.key === 'ArrowRight') {
        // expand
        const toggle = row.querySelector('.toggle-children');
        if (toggle) toggle.click();
        e.preventDefault();
      } else if (e.key === 'ArrowLeft') {
        const toggle = row.querySelector('.toggle-children');
        if (toggle && toggle.getAttribute('aria-expanded') === 'true') toggle.click();
        else {
          // move focus to parent
          const parentLi = row.closest('li.category-item').parentElement.closest('li.category-item');
          if (parentLi) parentLi.querySelector('.category-row').focus();
        }
        e.preventDefault();
      } else if (e.key === ' ' || e.key === 'Spacebar' || e.key === 'Enter') {
        const li = row.closest('li.category-item');
        const id = li.dataset.id;
        selectCategoryById(id);
        e.preventDefault();
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        // Move focus to next/previous visible row
        const visibleRows = Array.from(wrapper.querySelectorAll('.category-row')).filter(r => r.offsetParent !== null);
        const idx = visibleRows.indexOf(row);
        const next = e.key === 'ArrowDown' ? visibleRows[idx+1] : visibleRows[idx-1];
        if (next) next.focus();
        e.preventDefault();
      }
    });





    // --------------------------------------------------
    // Dropdown wrapper & summary button (pure JS)
    // --------------------------------------------------
    // Create a dropdown toggle that shows/hides the category tree
    const dropdownContainer = document.createElement('div');
    dropdownContainer.classList.add('category-dropdown','relative','w-full');

    const toggleBtn = document.createElement('button');
    toggleBtn.type = 'button';
    toggleBtn.classList.add('category-dropdown-toggle','w-full','text-left','px-3','py-2','rounded','border','border-gray-200','dark:border-slate-700','bg-white','dark:bg-slate-900','text-sm','flex','items-center','justify-between');
    toggleBtn.setAttribute('aria-haspopup', 'true');
    toggleBtn.setAttribute('aria-expanded', 'false');

    function buildSummary() {
      const selected = Array.from(select.querySelectorAll('option:checked'));
      if (!selected.length) return 'Select categories';
      if (selected.length === 1) return selected[0].textContent.trim();
      if (selected.length <= 3) return selected.map(o => o.textContent.trim()).join(', ');
      return `${selected.length} categories selected`;
    }

    const summaryText = document.createElement('span');
    summaryText.classList.add('category-summary','truncate');
    summaryText.textContent = buildSummary();

    const caret = document.createElement('span');
    caret.classList.add('caret','ml-2','text-gray-500');
    caret.textContent = 'v';

    toggleBtn.appendChild(summaryText);
    toggleBtn.appendChild(caret);

    const panel = document.createElement('div');
    panel.classList.add('category-dropdown-panel','hidden','absolute','z-50','mt-1','w-full','bg-white','dark:bg-slate-900','border','border-gray-200','dark:border-slate-700','rounded','shadow-lg','p-2','max-h-64','overflow-auto');

    // Move existing wrapper inside panel so it behaves like a dropdown
    const originalParent = wrapper.parentNode;
    originalParent.insertBefore(dropdownContainer, wrapper);
    panel.appendChild(wrapper);
    dropdownContainer.appendChild(toggleBtn);
    dropdownContainer.appendChild(panel);

    // Toggle panel visibility
    function openPanel() {
      panel.classList.remove('hidden');
      toggleBtn.setAttribute('aria-expanded', 'true');
      caret.textContent = '^';
    }
    function closePanel() {
      panel.classList.add('hidden');
      toggleBtn.setAttribute('aria-expanded', 'false');
      caret.textContent = 'v';
    }

    toggleBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      if (panel.classList.contains('hidden')) openPanel(); else closePanel();
    });

    // Close when clicking outside
    document.addEventListener('click', function(e) {
      if (!dropdownContainer.contains(e.target)) closePanel();
    });

    // Close with Escape
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') closePanel();
    });

    // Update summary when selection changes
    select.addEventListener('change', function() {
      summaryText.textContent = buildSummary();
    });

    // Ensure clicking inside panel doesn't propagate to document click handler
    panel.addEventListener('click', function(e) { e.stopPropagation(); });

  });

})();
