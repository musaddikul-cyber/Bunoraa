// Move ProductImage inline group to the top of the change form
document.addEventListener('DOMContentLoaded', function() {
  try {
    // The inline group id follows pattern: <modelname>_set-group; for ProductImage it's likely productimage_set-group
    var possibleIds = ['productimage_set-group', 'product_image_set-group', 'images_set-group'];
    var inline = null;
    for (var i = 0; i < possibleIds.length; i++) {
      inline = document.getElementById(possibleIds[i]);
      if (inline) break;
    }
    if (!inline) {
      // fallback: find any inline group that contains 'Product image' text
      var groups = document.querySelectorAll('.inline-group');
      for (var j = 0; j < groups.length; j++) {
        if (/product\s*image/i.test(groups[j].textContent || '')) {
          inline = groups[j];
          break;
        }
      }
    }

    if (!inline) return;

    // Remove any inline-local header (we want no 'Product images' heading)
    try {
      var inlineHeaders = inline.querySelectorAll('h1,h2,h3,h4,h5,h6,legend');
      inlineHeaders.forEach(function(h){ if (h && h.parentNode) h.parentNode.removeChild(h); });
    } catch (err) {
      // ignore
    }

    // Try to insert inline before the main 'Name' fieldset (not above the header)
    var nameField = document.querySelector('[name="name"], #id_name');
    var inserted = false;
    if (nameField) {
      // Attempt to find the containing fieldset/module for the name input
      var targetModule = nameField.closest('.module, .form-row, .form-row.related-widget-wrapper');
      if (targetModule && targetModule.parentNode) {
        targetModule.parentNode.insertBefore(inline, targetModule);
        inserted = true;
      }
    }

    if (!inserted) {
      // Fallback: insert near the top of the main change form, below the header
      var mainForm = document.querySelector('.change-form') || document.querySelector('#content-main');
      if (!mainForm) mainForm = document.body;
      var firstChild = mainForm.querySelector('.module') || mainForm.firstElementChild;
      if (firstChild && firstChild.parentNode) {
        firstChild.parentNode.insertBefore(inline, firstChild);
      }
    }

    // Improve UX: ensure only one primary checkbox can be selected (radio-like behavior)
    inline.addEventListener('change', function(e){
      var target = e.target;
      if (!target) return;
      // look for checkboxes named like 'is_primary'
      if (target.type === 'checkbox' && /is_primary/.test(target.name)) {
        var checkboxes = inline.querySelectorAll('input[type=checkbox][name*="is_primary"]');
        if (target.checked) {
          checkboxes.forEach(function(cb) { if (cb !== target) cb.checked = false; });
        }
      }
    });

    // Update add button text if present to a clearer label
    var addLink = inline.querySelector('.add-row a');
    if (addLink) {
      addLink.textContent = 'Add another Product image';
    }
  } catch (err) {
    // Do not crash admin; silently fail
    console && console.debug && console.debug('move_inlines_top:', err);
  }
});