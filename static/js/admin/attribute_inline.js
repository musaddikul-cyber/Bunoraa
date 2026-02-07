(function(){
    function attach(form) {
        if (!form) return;
        var attrSelect = form.querySelector('select[name$="-attribute"]');
        var valSelect = form.querySelector('select[name$="-attribute_value"]');
        if (!attrSelect || !valSelect) return;
        attrSelect.addEventListener('change', function(){
            var attrId = attrSelect.value;
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/products/attribute-values/?attribute=' + encodeURIComponent(attrId));
            xhr.onreadystatechange = function(){
                if (xhr.readyState === 4 && xhr.status === 200){
                    try{
                        var data = JSON.parse(xhr.responseText);
                        var current = valSelect.value;
                        valSelect.innerHTML = '';
                        var opt = document.createElement('option'); opt.value=''; opt.text='---------'; valSelect.appendChild(opt);
                        data.values.forEach(function(v){
                            var o = document.createElement('option'); o.value = v.id; o.text = v.value; valSelect.appendChild(o);
                        });
                        if (current) {
                            for (var i=0;i<valSelect.options.length;i++){
                                if (valSelect.options[i].value == current) { valSelect.selectedIndex = i; break; }
                            }
                        }
                    }catch(e){ }
                }
            };
            xhr.send();
        });
        if (attrSelect.value) {
            var evt = new Event('change'); attrSelect.dispatchEvent(evt);
        }
    }

    function init(){
        var forms = document.querySelectorAll('.inline-related .form-row');
        forms.forEach(function(f){ attach(f); });
        document.addEventListener('formset:added', function(event){ var f = event.target || event.detail && event.detail.form; if(f) attach(f); });
        var observer = new MutationObserver(function(muts){ muts.forEach(function(m){ m.addedNodes && m.addedNodes.forEach(function(node){ if(node.nodeType===1){ if(node.matches && node.matches('.inline-related')){ var fr = node.querySelectorAll('.form-row'); fr.forEach(attach); } else if(node.matches && node.matches('.form-row')){ attach(node); } } }); }); });
        observer.observe(document.body, { childList:true, subtree:true });
    }
    if (document.readyState === 'complete' || document.readyState === 'interactive') init(); else document.addEventListener('DOMContentLoaded', init);
})();