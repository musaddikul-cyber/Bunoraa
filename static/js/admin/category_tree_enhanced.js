/**
 * Enhanced Category Tree Widget for Product Admin
 * GitHub-style file tree with expand/collapse functionality
 * Primary category selection filters the categories dropdown
 */

(function($) {
    'use strict';

    // Configuration
    const CONFIG = {
        expandIcon: '▶',
        collapseIcon: '▼',
        fileIcon: '📄',
        folderIcon: '📁',
        indentSize: 20,
        animationDuration: 200
    };

    // State management
    const state = {
        expandedNodes: new Set(),
        selectedPrimary: null,
        categoryMap: new Map(),
        treeData: []
    };

    /**
     * Parse flat category list into hierarchical tree structure
     */
    function buildCategoryTree(categories) {
        const map = new Map();
        const roots = [];

        // First pass: create nodes
        categories.forEach(cat => {
            map.set(cat.id, {
                ...cat,
                children: [],
                level: 0
            });
        });

        // Second pass: build tree
        categories.forEach(cat => {
            const node = map.get(cat.id);
            if (cat.parent_id && map.has(cat.parent_id)) {
                const parent = map.get(cat.parent_id);
                parent.children.push(node);
                node.level = parent.level + 1;
            } else {
                roots.push(node);
            }
        });

        // Sort by sort_order and name
        const sortNodes = (nodes) => {
            nodes.sort((a, b) => {
                if (a.sort_order !== b.sort_order) {
                    return a.sort_order - b.sort_order;
                }
                return a.name.localeCompare(b.name);
            });
            nodes.forEach(node => sortNodes(node.children));
        };
        sortNodes(roots);

        return roots;
    }

    /**
     * Create tree HTML structure
     */
    function createTreeHTML(nodes, container, isSelectable = true, selectedId = null) {
        if (!nodes || nodes.length === 0) return;

        const ul = document.createElement('ul');
        ul.className = 'category-tree-list';

        nodes.forEach(node => {
            const li = document.createElement('li');
            li.className = 'category-tree-item';
            li.dataset.id = node.id;
            li.dataset.level = node.level;

            const hasChildren = node.children && node.children.length > 0;
            const isExpanded = state.expandedNodes.has(node.id);
            const isSelected = selectedId === node.id;

            // Create item content
            const content = document.createElement('div');
            content.className = 'category-tree-content' + (isSelected ? ' selected' : '');
            content.style.paddingLeft = `${node.level * CONFIG.indentSize}px`;

            // Toggle button for expandable nodes
            if (hasChildren) {
                const toggle = document.createElement('span');
                toggle.className = 'category-tree-toggle';
                toggle.dataset.id = node.id;
                toggle.innerHTML = isExpanded ? CONFIG.collapseIcon : CONFIG.expandIcon;
                content.appendChild(toggle);
            } else {
                const spacer = document.createElement('span');
                spacer.className = 'category-tree-spacer';
                content.appendChild(spacer);
            }

            // Icon
            const icon = document.createElement('span');
            icon.className = 'category-tree-icon';
            icon.innerHTML = hasChildren ? CONFIG.folderIcon : CONFIG.fileIcon;
            content.appendChild(icon);

            // Name (selectable)
            const name = document.createElement('span');
            name.className = 'category-tree-name';
            name.textContent = node.name;
            if (isSelectable) {
                name.dataset.id = node.id;
                name.style.cursor = 'pointer';
            }
            content.appendChild(name);

            li.appendChild(content);

            // Children container
            if (hasChildren) {
                const childrenContainer = document.createElement('div');
                childrenContainer.className = 'category-tree-children';
                childrenContainer.dataset.parentId = node.id;
                childrenContainer.style.display = isExpanded ? 'block' : 'none';
                createTreeHTML(node.children, childrenContainer, isSelectable, selectedId);
                li.appendChild(childrenContainer);
            }

            ul.appendChild(li);
        });

        container.appendChild(ul);
    }

    /**
     * Filter categories based on selected primary category
     */
    function filterCategoriesByPrimary(primaryId) {
        const $categoriesSelect = $('#id_categories');
        if (!$categoriesSelect.length) return;

        // Find primary category and its descendants
        const allowedIds = new Set();
        
        function collectDescendants(nodeId) {
            allowedIds.add(nodeId);
            const node = state.categoryMap.get(nodeId);
            if (node && node.children) {
                node.children.forEach(child => collectDescendants(child.id));
            }
        }

        if (primaryId) {
            collectDescendants(primaryId);
        } else {
            // If no primary selected, show all
            state.categoryMap.forEach((node, id) => allowedIds.add(id));
        }

        // Update select options visibility
        $categoriesSelect.find('option').each(function() {
            const $option = $(this);
            const value = $option.val();
            if (!value || allowedIds.has(value)) {
                $option.show();
            } else {
                $option.hide();
                $option.prop('selected', false);
            }
        });

        // If using Select2, refresh it
        if ($categoriesSelect.data('select2')) {
            $categoriesSelect.trigger('change.select2');
        }
    }

    /**
     * Update selected category display
     */
    function updateSelectedDisplay(containerId, categoryName, categoryId) {
        const $container = $(`#${containerId}-selected`);
        if ($container.length) {
            if (categoryId) {
                $container.html(`
                    <span class="selected-category-badge">
                        <span class="category-icon">📁</span>
                        ${categoryName}
                        <button type="button" class="clear-category" data-field="${containerId}">×</button>
                    </span>
                `);
            } else {
                $container.html('<span class="no-category-selected">No category selected</span>');
            }
        }
    }

    /**
     * Initialize category tree widget
     */
    function initCategoryTreeWidget() {
        const $primaryInput = $('#id_primary_category');
        const $categoriesInput = $('#id_categories');
        
        if (!$primaryInput.length && !$categoriesInput.length) return;

        // Get categories data from the source list
        const $sourceList = $('.category-tree-source');
        if (!$sourceList.length) return;

        const categories = [];
        $sourceList.find('li').each(function() {
            const $li = $(this);
            categories.push({
                id: $li.data('id'),
                name: $li.text().trim(),
                parent_id: $li.data('parent-id') || null,
                depth: parseInt($li.data('depth')) || 0,
                sort_order: parseInt($li.data('sort-order')) || 0
            });
        });

        // Build tree structure
        state.treeData = buildCategoryTree(categories);
        categories.forEach(cat => state.categoryMap.set(cat.id, cat));

        // Create primary category tree widget
        const $primaryContainer = $('#primary_category_tree_widget');
        if ($primaryContainer.length) {
            const selectedId = $primaryInput.val();
            $primaryContainer.empty();
            createTreeHTML(state.treeData, $primaryContainer[0], true, selectedId);
            
            // Initial display update
            if (selectedId) {
                const cat = state.categoryMap.get(selectedId);
                if (cat) {
                    updateSelectedDisplay('primary_category', cat.name, selectedId);
                    filterCategoriesByPrimary(selectedId);
                }
            }
        }

        // Create categories tree widget (for multi-select)
        const $categoriesContainer = $('#categories_tree_widget');
        if ($categoriesContainer.length) {
            const selectedIds = $categoriesInput.val() || [];
            $categoriesContainer.empty();
            createTreeHTML(state.treeData, $categoriesContainer[0], false, null);
            
            // Check selected items
            selectedIds.forEach(id => {
                $categoriesContainer.find(`[data-id="${id}"] .category-tree-content`).addClass('selected');
            });
        }
    }

    /**
     * Event handlers
     */
    function bindEvents() {
        // Toggle expand/collapse
        $(document).on('click', '.category-tree-toggle', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            const $toggle = $(this);
            const nodeId = $toggle.data('id');
            const $children = $(`.category-tree-children[data-parent-id="${nodeId}"]`);
            
            if (state.expandedNodes.has(nodeId)) {
                state.expandedNodes.delete(nodeId);
                $toggle.html(CONFIG.expandIcon);
                $children.slideUp(CONFIG.animationDuration);
            } else {
                state.expandedNodes.add(nodeId);
                $toggle.html(CONFIG.collapseIcon);
                $children.slideDown(CONFIG.animationDuration);
            }
        });

        // Primary category selection
        $(document).on('click', '#primary_category_tree_widget .category-tree-name', function(e) {
            e.preventDefault();
            const $name = $(this);
            const categoryId = $name.data('id');
            const categoryName = $name.text();
            
            // Update hidden input
            $('#id_primary_category').val(categoryId).trigger('change');
            
            // Update visual selection
            $('#primary_category_tree_widget .category-tree-content').removeClass('selected');
            $name.closest('.category-tree-content').addClass('selected');
            
            // Update display
            updateSelectedDisplay('primary_category', categoryName, categoryId);
            
            // Filter categories
            filterCategoriesByPrimary(categoryId);
            
            // Auto-expand ancestors
            let parentId = $name.closest('.category-tree-item').parent().closest('.category-tree-item').data('id');
            while (parentId) {
                state.expandedNodes.add(parentId);
                $(`.category-tree-toggle[data-id="${parentId}"]`).html(CONFIG.collapseIcon);
                $(`.category-tree-children[data-parent-id="${parentId}"]`).show();
                parentId = $(`.category-tree-item[data-id="${parentId}"]`).parent().closest('.category-tree-item').data('id');
            }
        });

        // Clear selection
        $(document).on('click', '.clear-category', function(e) {
            e.preventDefault();
            const field = $(this).data('field');
            $(`#id_${field}`).val('').trigger('change');
            $(`#${field}_tree_widget .category-tree-content`).removeClass('selected');
            updateSelectedDisplay(field, null, null);
            filterCategoriesByPrimary(null);
        });

        // Expand all / Collapse all buttons
        $(document).on('click', '.expand-all-categories', function(e) {
            e.preventDefault();
            $('.category-tree-children').slideDown(CONFIG.animationDuration);
            $('.category-tree-toggle').html(CONFIG.collapseIcon);
            state.treeData.forEach(node => collectAllIds(node, state.expandedNodes));
        });

        $(document).on('click', '.collapse-all-categories', function(e) {
            e.preventDefault();
            $('.category-tree-children').slideUp(CONFIG.animationDuration);
            $('.category-tree-toggle').html(CONFIG.expandIcon);
            state.expandedNodes.clear();
        });

        // Search/filter categories
        $(document).on('input', '.category-tree-search', function(e) {
            const searchTerm = $(this).val().toLowerCase();
            if (!searchTerm) {
                $('.category-tree-item').show();
                return;
            }

            $('.category-tree-item').each(function() {
                const $item = $(this);
                const name = $item.find('.category-tree-name').text().toLowerCase();
                if (name.includes(searchTerm)) {
                    $item.show();
                    // Expand parents to show match
                    let $parent = $item.parent().closest('.category-tree-item');
                    while ($parent.length) {
                        $parent.show();
                        const parentId = $parent.data('id');
                        state.expandedNodes.add(parentId);
                        $(`.category-tree-toggle[data-id="${parentId}"]`).html(CONFIG.collapseIcon);
                        $(`.category-tree-children[data-parent-id="${parentId}"]`).show();
                        $parent = $parent.parent().closest('.category-tree-item');
                    }
                } else {
                    $item.hide();
                }
            });
        });

        // Sync primary category change from original dropdown
        $('#id_primary_category').on('change', function() {
            const val = $(this).val();
            if (val) {
                const cat = state.categoryMap.get(val);
                if (cat) {
                    updateSelectedDisplay('primary_category', cat.name, val);
                    filterCategoriesByPrimary(val);
                }
            } else {
                updateSelectedDisplay('primary_category', null, null);
                filterCategoriesByPrimary(null);
            }
        });
    }

    function collectAllIds(node, set) {
        set.add(node.id);
        if (node.children) {
            node.children.forEach(child => collectAllIds(child, set));
        }
    }

    // Initialize on document ready
    $(document).ready(function() {
        initCategoryTreeWidget();
        bindEvents();
    });

    // Re-initialize on formset additions or dynamic content
    $(document).on('formset:added', initCategoryTreeWidget);

})(django.jQuery || jQuery);
