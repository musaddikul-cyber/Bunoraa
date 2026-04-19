/**
 * Image Manager
 * Enhanced image management for product admin
 */

(function($) {
    'use strict';
    
    class ImageManager {
        constructor(container) {
            this.container = $(container);
            this.tbody = this.container.find('.image-tbody, tbody').first();
            this.prefix = this.container.attr('id')?.replace('-group', '');
            
            this.init();
        }
        
        init() {
            this.initSortable();
            this.initTools();
            this.initPrimaryHandler();
            this.updateOrdering();
        }
        
        initSortable() {
            if (typeof Sortable === 'undefined') {
                return;
            }
            
            const self = this;
            
            new Sortable(this.tbody[0], {
                handle: '.drag-handle',
                animation: 150,
                ghostClass: 'sortable-ghost',
                onEnd: function() {
                    self.updateOrdering();
                    self.checkPrimaryImage();
                }
            });
            
            this.tbody.addClass('sortable-enabled');
        }
        
        initTools() {
            const self = this;
            
            // Auto set primary
            this.container.find('.auto-set-primary').on('click', function() {
                self.autoSetPrimary();
            });
            
            // Clear all alt text
            this.container.find('.clear-all-alt').on('click', function() {
                if (confirm('Clear all alt texts?')) {
                    self.container.find('input[name$="-alt_text"]').val('');
                }
            });
            
            // Add image link
            this.container.find('.add-image-link').on('click', function(e) {
                e.preventDefault();
                self.addEmptyImage();
            });
        }
        
        initPrimaryHandler() {
            const self = this;
            
            // When a checkbox is checked, uncheck all others
            this.tbody.on('change', 'input[name$="-is_primary"]', function() {
                if ($(this).is(':checked')) {
                    self.tbody.find('input[name$="-is_primary"]').not(this).prop('checked', false);
                    self.tbody.find('.form-row').removeClass('is-primary-row');
                    $(this).closest('.form-row').addClass('is-primary-row');
                }
            });
        }
        
        autoSetPrimary() {
            // Set first row as primary
            const firstRow = this.tbody.find('.form-row').first();
            if (firstRow.length) {
                this.tbody.find('input[name$="-is_primary"]').prop('checked', false);
                firstRow.find('input[name$="-is_primary"]').prop('checked', true);
                this.tbody.find('.form-row').removeClass('is-primary-row');
                firstRow.addClass('is-primary-row');
            }
        }
        
        checkPrimaryImage() {
            // Ensure at least one image is primary
            const hasPrimary = this.tbody.find('input[name$="-is_primary"]:checked').length > 0;
            if (!hasPrimary) {
                this.autoSetPrimary();
            }
        }
        
        updateOrdering() {
            this.tbody.find('.form-row').each(function(index) {
                $(this).find('input[name$="-ordering"]').val(index);
            });
        }
        
        addEmptyImage() {
            const totalFormsInput = $(`#id_${this.prefix}-TOTAL_FORMS`);
            const totalForms = parseInt(totalFormsInput.val());
            const maxForms = parseInt($(`#id_${this.prefix}-MAX_NUM_FORMS`).val()) || Infinity;
            
            if (totalForms >= maxForms) {
                alert('Maximum number of images reached');
                return;
            }
            
            const emptyRow = this.tbody.find('.empty-form').first();
            if (emptyRow.length === 0) return;
            
            const newRow = emptyRow.clone().removeClass('empty-form');
            const newIndex = totalForms;
            
            newRow.attr('id', `${this.prefix}-${newIndex}`);
            newRow.find('input, select, textarea').each(function() {
                const $input = $(this);
                const name = $input.attr('name');
                const id = $input.attr('id');
                
                if (name) {
                    $input.attr('name', name.replace('__prefix__', newIndex));
                    if ($input.attr('type') !== 'hidden') {
                        $input.val('');
                    }
                }
                if (id) {
                    $input.attr('id', id.replace('__prefix__', newIndex));
                }
            });
            
            newRow.insertBefore(emptyRow).hide().fadeIn(300);
            totalFormsInput.val(newIndex + 1);
            
            this.updateOrdering();
        }
    }
    
    // Initialize
    $(document).ready(function() {
        $('.image-group').each(function() {
            new ImageManager(this);
        });
    });
    
})(django.jQuery || jQuery);
