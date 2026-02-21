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
    bindPrimaryCategorySync();
  }

  function buildCategoryParentMap() {
    var nodes = document.querySelectorAll(".category-tree-source li");
    var parentMap = {};

    for (var i = 0; i < nodes.length; i += 1) {
      var node = nodes[i];
      var id = String(node.dataset.id || "");
      var parentId = String(node.dataset.parentId || "");
      if (id) {
        parentMap[id] = parentId;
      }
    }

    return parentMap;
  }

  function buildCategoryChain(categoryId, parentMap) {
    var chain = [];
    var visited = {};
    var current = String(categoryId || "");

    while (current && !visited[current]) {
      chain.push(current);
      visited[current] = true;
      current = String(parentMap[current] || "");
    }

    chain.reverse();
    return chain;
  }

  function addPrimaryCategoryToCategories(primaryCategoryId) {
    var categoryId = String(primaryCategoryId || "");
    if (!categoryId) {
      return;
    }

    // filter_horizontal widget mode (Django SelectFilter2)
    var fromId = "id_categories_from";
    var toId = "id_categories_to";
    var hasSelectFilter =
      typeof window.SelectBox !== "undefined" &&
      document.getElementById(fromId) &&
      document.getElementById(toId);

    if (hasSelectFilter) {
      var selectBox = window.SelectBox;
      if (!selectBox.cache[fromId] || !selectBox.cache[toId]) {
        return;
      }
      if (selectBox.cache_contains(toId, categoryId)) {
        return;
      }

      var sourceNode = null;
      var fromCache = selectBox.cache[fromId];
      for (var i = 0; i < fromCache.length; i += 1) {
        if (String(fromCache[i].value) === categoryId) {
          sourceNode = fromCache[i];
          break;
        }
      }

      if (!sourceNode) {
        return;
      }

      selectBox.add_to_cache(toId, {
        value: sourceNode.value,
        text: sourceNode.text,
        displayed: 1,
      });
      selectBox.delete_from_cache(fromId, categoryId);
      selectBox.redisplay(fromId);
      selectBox.redisplay(toId);

      if (typeof window.SelectFilter !== "undefined") {
        window.SelectFilter.refresh_icons("id_categories");
        if (typeof window.SelectFilter.refresh_filtered_selects === "function") {
          window.SelectFilter.refresh_filtered_selects("id_categories");
        }
        if (typeof window.SelectFilter.refresh_filtered_warning === "function") {
          window.SelectFilter.refresh_filtered_warning("id_categories");
        }
      }

      var toBox = document.getElementById(toId);
      if (toBox) {
        for (var j = 0; j < toBox.options.length; j += 1) {
          if (String(toBox.options[j].value) === categoryId) {
            toBox.options[j].selected = true;
            break;
          }
        }
        toBox.dispatchEvent(new Event("change", { bubbles: true }));
      }
      return;
    }

    // fallback for uninitialized/default multiple-select widget
    var categoriesSelect = document.getElementById("id_categories");
    if (!categoriesSelect) {
      return;
    }

    for (var k = 0; k < categoriesSelect.options.length; k += 1) {
      var option = categoriesSelect.options[k];
      if (String(option.value) === categoryId) {
        if (!option.selected) {
          option.selected = true;
          categoriesSelect.dispatchEvent(new Event("change", { bubbles: true }));
        }
        break;
      }
    }
  }

  function addPrimaryCategoryWithAncestorsToCategories(primaryCategoryId, parentMap) {
    var chain = buildCategoryChain(primaryCategoryId, parentMap || {});

    for (var i = 0; i < chain.length; i += 1) {
      addPrimaryCategoryToCategories(chain[i]);
    }
  }

  function bindPrimaryCategorySync() {
    var primaryCategoryInput = document.getElementById("id_primary_category");
    if (!primaryCategoryInput) {
      return;
    }
    if (primaryCategoryInput.dataset.primaryCategorySyncBound === "1") {
      return;
    }

    var parentMap = buildCategoryParentMap();

    primaryCategoryInput.addEventListener("change", function () {
      addPrimaryCategoryWithAncestorsToCategories(primaryCategoryInput.value, parentMap);
    });
    primaryCategoryInput.dataset.primaryCategorySyncBound = "1";

    // Initial sync for edit page and preserved form state after validation errors.
    addPrimaryCategoryWithAncestorsToCategories(primaryCategoryInput.value, parentMap);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }

  document.addEventListener("formset:added", bootstrap);
})();
