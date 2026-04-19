/**
 * Product Attribute Manager
 * Enhanced attribute value selection with category-aware filtering
 */

(function($) {
    'use strict';
    
    class AttributeManager {
        constructor(container) {
            this.container = $(container);
            this.categorySelect = $('#id_primary_category');
            this.init();
        }
        
        init() {
            this.bindEvents();
            this.filterByCategory();
        }
        
        bindEvents() {
            const self = this;
            
            // Refresh when category changes
            if (this.categorySelect.length) {
                this.categorySelect.on('change', function() {
                    self.filterByCategory();
                });
            }
            
            // Handle new row additions
            $(document).on('formset:added', function(event, $row) {
                if ($row.closest('.attribute-values-inline').length) {
                    self.filterNewRow($row);
                }
            });
        }
        
        filterByCategory() {
            let categoryId = this.categorySelect.val();
            // Handle list-like string values
            if (typeof categoryId === 'string' && categoryId.startsWith('[') && categoryId.endsWith(']')) {
                try {
                    categoryId = JSON.parse(categoryId.replace(/'/g, '"'))[0];
                } catch (e) {
                    categoryId = categoryId.replace(/[\[\]'"]/g, '');
                }
            }
            if (!categoryId || categoryId === '') return;

            // Get allowed facets for this category
            this.fetchCategoryFacets(categoryId).then(facets => {
                this.filterAttributeSelects(facets);
            });
        }
        
        async fetchCategoryFacets(categoryId) {
            // Fetch facets from API or page data
            try {
                const response = await fetch(`/api/catalog/categories/${categoryId}/facets/`);
                if (response.ok) {
                    return await response.json();
                }
            } catch (e) {
                console.log('Could not fetch facets, showing all attributes');
            }
            return null;
        }
        
        filterAttributeSelects(allowedFacets) {
            if (!allowedFacets) return;
            
            const facetSlugs = allowedFacets.map(f => f.slug);
            
            this.container.find('.attribute-value-select').each(function() {
                const $select = $(this);
                const currentValue = $select.val();
                
                // Filter options
                $select.find('option').each(function() {
                    const $option = $(this);
                    const attrName = $option.text().split(':')[0].toLowerCase().trim();
                    
                    if (facetSlugs.includes(attrName)) {
                        $option.prop('disabled', false).show();
                    } else {
                        $option.prop('disabled', true).hide();
                    }
                });
                
                // Restore value if still valid
                if (currentValue) {
                    const $current = $select.find(`option[value="${currentValue}"]`);
                    if ($current.is(':disabled')) {
                        $select.val('');
                    }
                }
            });
        }
        
        filterNewRow($row) {
            // Apply same filtering to newly added rows
            this.filterByCategory();
        }
    }
    
    // Initialize
    $(document).ready(function() {
        $('.attribute-values-inline').each(function() {
            new AttributeManager(this);
        });
    });
    
})(django.jQuery || jQuery);
