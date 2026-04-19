/**
 * Product Variant Manager
 * Enhanced variant management for Django admin with:
 * - Sortable drag-drop reordering
 * - Variant generator from option combinations
 * - Bulk pricing tools
 * - Visual stock indicators
 */

(function($) {
    'use strict';
    
    // =========================================================================
    // Variant Manager Class
    // =========================================================================
    
    class VariantManager {
        constructor(tableElement) {
            this.table = $(tableElement);
            this.tbody = this.table.find('.variant-tbody, tbody');
            this.prefix = this.table.closest('.inline-group').attr('id').replace('-group', '');
            this.totalFormsInput = $(`#id_${this.prefix}-TOTAL_FORMS`);
            this.maxFormsInput = $(`#id_${this.prefix}-MAX_NUM_FORMS`);
            this.emptyFormTemplate = $(`#${this.prefix}-empty`);
            
            this.init();
        }
        
        init() {
            this.initSortable();
            this.initGenerator();
            this.initBulkPricing();
            this.initAddButton();
            this.updateRowNumbers();
            this.bindEvents();
        }
        
        // =====================================================================
        // Sortable Drag-Drop
        // =====================================================================
        
        initSortable() {
            if (typeof Sortable === 'undefined') {
                console.warn('Sortable.js not loaded');
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
                    $(evt.item).addClass('dragging');
                },
                onEnd: function(evt) {
                    $(evt.item).removeClass('dragging');
                    self.updateRowNumbers();
                    self.updateOrdering();
                }
            });
            
            // Add visual feedback
            this.tbody.addClass('sortable-enabled');
        }
        
        // =====================================================================
        // Variant Generator
        // =====================================================================
        
        initGenerator() {
            const self = this;
            const panel = $('#variant-generator-panel');
            const trigger = $('.variant-generator-trigger');
            const optionSelector = panel.find('.option-selector');
            const generateBtn = panel.find('.generate-btn');
            
            // Toggle panel
            trigger.on('click', function(e) {
                e.preventDefault();
                panel.slideToggle(200);
                if (panel.is(':visible')) {
                    self.loadOptionsForGenerator(optionSelector);
                }
            });
            
            panel.find('.panel-close').on('click', function() {
                panel.slideUp(200);
            });
            
            // Handle option selection
            optionSelector.on('change', 'input[type="checkbox"]', function() {
                const selectedGroups = optionSelector.find('input:checked').length;
                generateBtn.prop('disabled', selectedGroups === 0);
            });
            
            // Generate variants
            generateBtn.on('click', function() {
                self.generateVariants(optionSelector);
            });
        }
        
        loadOptionsForGenerator(container) {
            // Load available options from the server or page data
            // For now, we'll use a simplified approach
            const options = this.getAvailableOptions();
            
            let html = '';
            Object.keys(options).forEach(optionName => {
                const values = options[optionName];
                html += `
                    <div class="option-group" data-option="${optionName}">
                        <label class="option-group-title">
                            <input type="checkbox" class="select-all-option"> 
                            <strong>${optionName}</strong>
                        </label>
                        <div class="option-values">
                            ${values.map(v => `
                                <label class="option-value-label">
                                    <input type="checkbox" value="${v.id}" data-value="${v.value}">
                                    ${v.value}
                                </label>
                            `).join('')}
                        </div>
                    </div>
                `;
            });
            
            container.html(html || '<p class="no-options">No options configured. Configure options in the database first.</p>');
        }
        
        getAvailableOptions() {
            // This should be populated with actual data from the server
            // For demonstration, we'll check if there's data on the page
            const options = {};
            
            // Try to extract from existing variant rows
            this.tbody.find('.option-value-select').first().find('optgroup').each(function() {
                const optionName = $(this).attr('label');
                options[optionName] = [];
                $(this).find('option').each(function() {
                    options[optionName].push({
                        id: $(this).val(),
                        value: $(this).text().split(': ')[1] || $(this).text()
                    });
                });
            });
            
            return options;
        }
        
        generateVariants(optionSelector) {
            const selectedOptions = {};
            
            optionSelector.find('.option-group').each(function() {
                const checkedValues = $(this).find('.option-values input:checked').map(function() {
                    return {
                        id: $(this).val(),
                        value: $(this).data('value'),
                        option: $(this).closest('.option-group').data('option')
                    };
                }).get();
                
                if (checkedValues.length > 0) {
                    selectedOptions[$(this).data('option')] = checkedValues;
                }
            });
            
            // Generate combinations
            const combinations = this.cartesianProduct(selectedOptions);
            
            // Create variant rows
            combinations.forEach(combo => {
                this.addVariantFromOptions(combo);
            });
            
            // Close panel and show success
            $('#variant-generator-panel').slideUp(200);
            this.showNotification(`Generated ${combinations.length} variant(s)`, 'success');
        }
        
        cartesianProduct(options) {
            const keys = Object.keys(options);
            if (keys.length === 0) return [];
            
            let result = [[]];
            
            keys.forEach(key => {
                const values = options[key];
                const newResult = [];
                
                result.forEach(existing => {
                    values.forEach(value => {
                        newResult.push([...existing, { option: key, ...value }]);
                    });
                });
                
                result = newResult;
            });
            
            return result;
        }
        
        addVariantFromOptions(optionValues) {
            const totalForms = parseInt(this.totalFormsInput.val());
            const maxForms = parseInt(this.maxFormsInput.val()) || Infinity;
            
            if (totalForms >= maxForms) {
                this.showNotification('Maximum number of variants reached', 'error');
                return;
            }
            
            // Get empty form
            const emptyRow = this.tbody.find('.empty-form').first();
            if (emptyRow.length === 0) {
                console.error('No empty form template found');
                return;
            }
            
            // Clone and prepare new row
            const newRow = emptyRow.clone().removeClass('empty-form');
            const newIndex = totalForms;
            
            // Update IDs and names
            newRow.attr('id', `${this.prefix}-${newIndex}`);
            newRow.find('input, select, textarea').each(function() {
                const $input = $(this);
                const name = $input.attr('name');
                const id = $input.attr('id');
                
                if (name) {
                    $input.attr('name', name.replace('__prefix__', newIndex));
                }
                if (id) {
                    $input.attr('id', id.replace('__prefix__', newIndex));
                }
            });
            
            // Auto-generate SKU from options
            const skuParts = optionValues.map(ov => ov.value.toString().replace(/\s+/g, '-'));
            const baseSku = $('input[name="sku"]').val() || 'PROD';
            const generatedSku = `${baseSku}-${skuParts.join('-')}`.toUpperCase();
            
            // Pre-fill fields
            newRow.find('input[name$="-sku"]').val(generatedSku);
            
            // Set option values in the multi-select
            const optionIds = optionValues.map(ov => ov.id);
            newRow.find('.option-value-select').val(optionIds);
            
            // Insert before empty row
            newRow.insertBefore(emptyRow);
            
            // Update form count
            this.totalFormsInput.val(newIndex + 1);
            
            // Animate in
            newRow.hide().fadeIn(300);
            
            this.updateRowNumbers();
        }
        
        // =====================================================================
        // Bulk Pricing Tools
        // =====================================================================
        
        initBulkPricing() {
            const self = this;
            const panel = $('#bulk-pricing-panel');
            const trigger = $('.variant-bulk-pricing');
            
            trigger.on('click', function(e) {
                e.preventDefault();
                $('#variant-generator-panel').hide();
                panel.slideToggle(200);
            });
            
            panel.find('.panel-close').on('click', function() {
                panel.slideUp(200);
            });
            
            // Apply fixed price
            panel.find('.apply-bulk-price').on('click', function() {
                const price = parseFloat(panel.find('.bulk-price-input').val());
                if (!isNaN(price)) {
                    self.setAllVariantPrices(price);
                    panel.slideUp(200);
                }
            });
            
            // Apply percentage adjustment
            panel.find('.apply-bulk-percent').on('click', function() {
                const percent = parseFloat(panel.find('.bulk-percent-input').val());
                if (!isNaN(percent)) {
                    self.adjustAllVariantPrices(percent);
                    panel.slideUp(200);
                }
            });
            
            // Apply stock
            panel.find('.apply-bulk-stock').on('click', function() {
                const stock = parseInt(panel.find('.bulk-stock-input').val());
                if (!isNaN(stock)) {
                    self.setAllVariantStock(stock);
                    panel.slideUp(200);
                }
            });
        }
        
        setAllVariantPrices(price) {
            this.tbody.find('input[name$="-price"]').not('[name*="__prefix__"]').val(price.toFixed(2));
            this.showNotification(`Set all prices to $${price.toFixed(2)}`, 'success');
        }
        
        adjustAllVariantPrices(percent) {
            this.tbody.find('input[name$="-price"]').not('[name*="__prefix__"]').each(function() {
                const current = parseFloat($(this).val()) || 0;
                const adjusted = current * (1 + percent / 100);
                $(this).val(adjusted.toFixed(2));
            });
            this.showNotification(`Adjusted all prices by ${percent}%`, 'success');
        }
        
        setAllVariantStock(quantity) {
            this.tbody.find('input[name$="-stock_quantity"]').not('[name*="__prefix__"]').val(quantity);
            this.updateStockBadges();
            this.showNotification(`Set all stock to ${quantity}`, 'success');
        }
        
        // =====================================================================
        // Add New Variant Button
        // =====================================================================
        
        initAddButton() {
            const self = this;
            
            this.table.closest('.inline-group').find('.add-variant-link').on('click', function(e) {
                e.preventDefault();
                self.addEmptyVariant();
            });
        }
        
        addEmptyVariant() {
            const totalForms = parseInt(this.totalFormsInput.val());
            const maxForms = parseInt(this.maxFormsInput.val()) || Infinity;
            
            if (totalForms >= maxForms) {
                this.showNotification('Maximum number of variants reached', 'error');
                return;
            }
            
            const emptyRow = this.tbody.find('.empty-form').first();
            if (emptyRow.length === 0) return;
            
            const newRow = emptyRow.clone().removeClass('empty-form');
            const newIndex = totalForms;
            
            // Update IDs and names
            newRow.attr('id', `${this.prefix}-${newIndex}`);
            newRow.find('input, select, textarea').each(function() {
                const $input = $(this);
                const name = $input.attr('name');
                const id = $input.attr('id');
                
                if (name) {
                    $input.attr('name', name.replace('__prefix__', newIndex));
                    // Clear values
                    if ($input.attr('type') !== 'hidden') {
                        $input.val('');
                    }
                }
                if (id) {
                    $input.attr('id', id.replace('__prefix__', newIndex));
                }
            });
            
            newRow.insertBefore(emptyRow).hide().fadeIn(300);
            this.totalFormsInput.val(newIndex + 1);
            this.updateRowNumbers();
        }
        
        // =====================================================================
        // Helper Methods
        // =====================================================================
        
        updateRowNumbers() {
            // Update row styling
            this.tbody.find('.form-row').each(function(index) {
                $(this).removeClass('row1 row2').addClass(index % 2 === 0 ? 'row1' : 'row2');
            });
        }
        
        updateOrdering() {
            // Update hidden ordering fields if they exist
            this.tbody.find('.form-row').each(function(index) {
                $(this).find('input[name$="-ordering"]').val(index);
            });
        }
        
        updateStockBadges() {
            // Re-render stock badges based on input values
            // This would require server-side rendering or client-side templating
        }
        
        showNotification(message, type = 'info') {
            // Simple notification
            const notification = $(`<div class="variant-notification ${type}">${message}</div>`);
            $('body').append(notification);
            
            setTimeout(() => {
                notification.fadeOut(() => notification.remove());
            }, 3000);
        }
        
        // =====================================================================
        // Event Binding
        // =====================================================================
        
        bindEvents() {
            const self = this;
            
            // Handle default variant selection (radio-like behavior)
            this.tbody.on('change', 'input[name$="-is_default"]', function() {
                if ($(this).is(':checked')) {
                    // Uncheck other defaults
                    self.tbody.find('input[name$="-is_default"]').not(this).prop('checked', false);
                }
            });
            
            // Handle stock quantity changes for live badge updates
            this.tbody.on('change', 'input[name$="-stock_quantity"]', function() {
                // Could trigger badge update here
            });
        }
    }
    
    // =========================================================================
    // Initialize on document ready
    // =========================================================================
    
    $(document).ready(function() {
        $('.variant-group').each(function() {
            new VariantManager(this);
        });
        
        // Re-initialize for dynamically added formsets (Django admin inlines)
        $(document).on('formset:added', function(event, $row, formsetName) {
            if ($row.closest('.variant-group').length) {
                // Re-initialize for this formset
                $row.closest('.variant-group').each(function() {
                    // Destroy existing if any
                    // Re-create
                    new VariantManager(this);
                });
            }
        });
    });
    
    // =========================================================================
    // CSS for notifications and helper classes
    // =========================================================================
    
    const style = document.createElement('style');
    style.textContent = `
        .variant-notification {
            position: fixed;
            top: 60px;
            right: 20px;
            padding: 12px 20px;
            background: #333;
            color: white;
            border-radius: 4px;
            z-index: 9999;
            animation: slideIn 0.3s ease;
        }
        .variant-notification.success { background: #16a34a; }
        .variant-notification.error { background: #dc2626; }
        .variant-notification.info { background: #3b82f6; }
        
        .sortable-ghost {
            opacity: 0.4;
            background: #f0f9ff;
        }
        .sortable-chosen {
            background: #e0f2fe;
        }
        .sortable-drag {
            opacity: 1;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    `;
    document.head.appendChild(style);
    
})(django.jQuery || jQuery);
