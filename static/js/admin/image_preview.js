(function() {
    function attachPreview(inline) {
        if (!inline) return;
        // find file input in this inline
        var fileInput = inline.querySelector('input[type=file]');
        var img = inline.querySelector('img.admin-image-preview');
        if (!fileInput || !img) return;
        // Avoid attaching multiple times
        if (fileInput._previewAttached) return;
        fileInput._previewAttached = true;
        fileInput.addEventListener('change', function(e) {
            var file = fileInput.files && fileInput.files[0];
            if (file) {
                var reader = new FileReader();
                reader.onload = function(ev) {
                    img.src = ev.target.result;
                    img.style.display = '';
                };
                reader.readAsDataURL(file);
            } else {
                img.src = '';
                img.style.display = 'none';
            }
        });
    }

    function init() {
        // Attach to existing inlines
        var inlines = document.querySelectorAll('.inline-related');
        inlines.forEach(function(group) {
            var forms = group.querySelectorAll('.dynamic-' + (group.dataset.inlineName || ''));
            // fallback: select all .inline-related .form-row
            if (!forms.length) forms = group.querySelectorAll('.form-row');
            forms.forEach(function(form) {
                attachPreview(form);
            });
        });

        // Listen for formset:added event (triggered when adding an inline)
        document.addEventListener('formset:added', function(event) {
            var inline = event.target || event.detail && event.detail.form;
            if (inline) {
                // event.detail.form may be the form element
                attachPreview(inline);
            }
            // sometimes event.target is the whole inline group; try to attach to last form
            var group = document.querySelectorAll('.inline-related');
            if (group && group.length) {
                var last = group[group.length-1].querySelectorAll('.form-row');
                if (last && last.length) attachPreview(last[last.length-1]);
            }
        });

        // MutationObserver fallback
        var observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(m) {
                m.addedNodes && m.addedNodes.forEach(function(node) {
                    if (node.nodeType === 1) {
                        if (node.matches && node.matches('.inline-related')) {
                            var forms = node.querySelectorAll('.form-row');
                            forms.forEach(function(f) { attachPreview(f); });
                        } else if (node.matches && node.matches('.form-row')) {
                            attachPreview(node);
                        }
                    }
                });
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        init();
    } else {
        document.addEventListener('DOMContentLoaded', init);
    }
})();