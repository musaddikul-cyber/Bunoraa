/**
 * Sortable Inlines - Drag and Drop Reordering for Django Admin Inlines
 * Generic solution for any inline with ordering field
 */

(function($) {
    'use strict';
    
    class SortableInline {
        constructor(container) {
            this.container = $(container);
            this.tbody = this.container.find('tbody, .sortable-tbody').first();
            this.prefix = this.container.attr('id')?.replace('-group', '');
            
            if (!this.tbody.length || !this.prefix) {
                return;
            }
            
            this.init();
        }
        
        init() {
            this.initSortable();
            this.updateOrdering();
        }
        
        initSortable() {
            if (typeof Sortable === 'undefined') {
                console.warn('Sortable.js not available');
                return;
            }
            
            const self = this;
            
            this.sortable = new Sortable(this.tbody[0], {
                handle: '.drag-handle',
                animation: 150,
                ghostClass: 'sortable-ghost',
                chosenClass: 'sortable-chosen',
                dragClass: 'sortable-drag',
                onStart: function(evt) {
                    $(evt.from).addClass('sorting-active');
                    $(evt.item).addClass('dragging');
                },
                onEnd: function(evt) {
                    $(evt.from).removeClass('sorting-active');
                    $(evt.item).removeClass('dragging');
                    self.updateOrdering();
                    self.updateRowClasses();
                    self.triggerChange();
                }
            });
            
            // Add enabled class
            this.tbody.addClass('sortable-enabled');
        }
        
        updateOrdering() {
            const orderingInputs = this.tbody.find('input[name$="-ordering"]');
            
            if (!orderingInputs.length) {
                // Check for other ordering field names
                const orderInputs = this.tbody.find('input[name$="-order"]');
                if (!orderInputs.length) return;
            }
            
            const inputs = orderingInputs.length ? orderingInputs : this.tbody.find('input[name$="-order"]');
            
            // Update all ordering values based on current position
            inputs.each(function(index) {
                $(this).val(index);
            });
        }
        
        updateRowClasses() {
            // Update alternating row classes
            this.tbody.find('.form-row').each(function(index) {
                $(this)
                    .removeClass('row1 row2')
                    .addClass(index % 2 === 0 ? 'row1' : 'row2');
            });
        }
        
        triggerChange() {
            // Trigger change event for any listeners
            this.container.trigger('sortable:reordered');
        }
    }
    
    // Initialize on document ready
    $(document).ready(function() {
        // Find all inline groups with sortable data attribute
        $('[data-sortable="true"]').each(function() {
            new SortableInline(this);
        });
        
        // Also handle inline groups with ordering fields
        $('.inline-group').each(function() {
            const $group = $(this);
            if ($group.find('tbody .drag-handle').length && !$group.data('sortable-initialized')) {
                new SortableInline($group);
                $group.data('sortable-initialized', true);
            }
        });
    });
    
    // Handle dynamically added formsets
    $(document).on('formset:added', function(event, $row) {
        // Check if this inline should be sortable
        const $group = $row.closest('.inline-group');
        if ($group.find('tbody .drag-handle').length && !$group.data('sortable-initialized')) {
            new SortableInline($group);
            $group.data('sortable-initialized', true);
        }
    });
    
})(django.jQuery || jQuery);
