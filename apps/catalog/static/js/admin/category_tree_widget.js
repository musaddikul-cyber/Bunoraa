(function () {
  "use strict";

  function initCategoryTree(container) {
    if (!container || container.dataset.initialized === "1") {
      return;
    }

    var input = container.querySelector('input[type="hidden"]');
    var source = container.querySelector(".category-tree-source");
    var display = container.querySelector(".category-tree-display");

    if (!input || !source || !display) {
      return;
    }

    var selectedId = String(input.value || "");
    var nodes = Array.prototype.slice.call(source.querySelectorAll("li"));

    nodes.forEach(function (node) {
      var id = String(node.dataset.id || "");
      var depth = parseInt(node.dataset.depth || "0", 10);
      var label = node.textContent || "";
      var row = document.createElement("button");

      row.type = "button";
      row.className = "category-tree-node";
      row.dataset.value = id;
      row.style.paddingLeft = (8 + depth * 16) + "px";
      row.innerHTML =
        '<span class="category-tree-node-depth">&bull;</span><span>' + label + "</span>";

      if (id === selectedId) {
        row.classList.add("is-selected");
      }

      row.addEventListener("click", function () {
        var selected = display.querySelector(".category-tree-node.is-selected");
        if (selected) {
          selected.classList.remove("is-selected");
        }
        row.classList.add("is-selected");
        input.value = id;
        input.dispatchEvent(new Event("change", { bubbles: true }));
      });

      display.appendChild(row);
    });

    container.dataset.initialized = "1";
  }

  function bootstrap() {
    var widgets = document.querySelectorAll(".category-tree-widget-container");
    widgets.forEach(initCategoryTree);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }

  document.addEventListener("formset:added", bootstrap);
})();
