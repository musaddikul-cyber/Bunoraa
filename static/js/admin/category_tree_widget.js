(function($) {
    $(document).ready(function() {
        if ($('#category-tree-source').length === 0) {
            return;
        }

        const sourceList = $('#category-tree-source');
        const displayContainer = $('#category-tree-display');
        const hiddenInput = $('input[name="primary_category"]');

        let categories = {};
        let roots = [];

        function normalizeId(value) {
            return String(value || '').trim().toLowerCase();
        }

        sourceList.find('li').each(function() {
            const el = $(this);
            const id = normalizeId(el.attr('data-id'));
            if (!id) {
                return;
            }
            const parentId = normalizeId(el.attr('data-parent-id'));
            categories[id] = {
                id: id,
                parentId: parentId,
                name: el.text(),
                children: []
            };
        });

        for (const id in categories) {
            const category = categories[id];
            if (category.parentId && categories[category.parentId]) {
                categories[category.parentId].children.push(category);
            } else {
                roots.push(category);
            }
        }

        function buildTreeHtml(nodes, isRoot) {
            let html = '<ul class="' + (isRoot ? 'category-tree' : 'children') + '">';
            nodes.forEach(function(node) {
                html += '<li class="category-node" data-id="' + node.id + '">';
                html += '<div class="node-row">';
                html += '<span class="label">' + node.name + '</span>';

                if (node.children.length > 0) {
                    html += '<button type="button" class="toggle" aria-expanded="false" aria-label="Toggle subcategories"></button>';
                } else {
                    html += '<span class="toggle spacer" aria-hidden="true"></span>';
                }

                html += '</div>';

                if (node.children.length > 0) {
                    html += buildTreeHtml(node.children, false);
                }
                html += '</li>';
            });
            html += '</ul>';
            return html;
        }

        displayContainer.html(buildTreeHtml(roots, true));

        function selectLabel(label) {
            displayContainer.find('.label.selected').removeClass('selected');
            label.addClass('selected');
        }

        const initialValue = normalizeId(hiddenInput.val());
        if (initialValue) {
            const selectedLabel = displayContainer.find('.category-node[data-id="' + initialValue + '"] > .node-row > .label');
            if (selectedLabel.length) {
                selectLabel(selectedLabel);
                selectedLabel.parents('.category-node').addClass('expanded');
                selectedLabel.parents('.category-node').find('> .node-row > .toggle').attr('aria-expanded', 'true');
            }
        }

        function toggleNode(node, toggleButton) {
            if (node.find('> ul.children').length === 0) {
                return;
            }
            const isExpanded = node.toggleClass('expanded').hasClass('expanded');
            toggleButton.attr('aria-expanded', isExpanded ? 'true' : 'false');
        }

        displayContainer.on('click', '.toggle', function(e) {
            e.preventDefault();
            e.stopPropagation();
            const node = $(this).closest('.category-node');
            toggleNode(node, $(this));
        });

        displayContainer.on('click', '.node-row', function(e) {
            e.preventDefault();
            const row = $(this);
            const label = row.find('.label');
            const nodeId = label.closest('.category-node').data('id');

            selectLabel(label);
            hiddenInput.val(nodeId);

            const toggleButton = row.find('.toggle').first();
            if (toggleButton.length && !toggleButton.hasClass('spacer')) {
                toggleNode(row.closest('.category-node'), toggleButton);
            }
        });
    });
})(django.jQuery);
