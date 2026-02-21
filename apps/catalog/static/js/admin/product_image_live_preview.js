(function () {
  "use strict";

  function isLikelyImageInput(input) {
    if (!input || input.tagName !== "INPUT" || input.type !== "file") {
      return false;
    }
    var name = String(input.name || "");
    var id = String(input.id || "");
    var accept = String(input.getAttribute("accept") || "");
    return (
      /-image$/.test(name) ||
      /-image$/.test(id) ||
      accept.indexOf("image/") !== -1
    );
  }

  function findRow(element) {
    if (!element) {
      return null;
    }
    return (
      element.closest("tr.form-row") ||
      element.closest(".form-row") ||
      element.closest(".inline-related")
    );
  }

  function getPreviewTarget(fileInput) {
    var row = findRow(fileInput);
    if (!row) {
      return null;
    }

    var previewCell = row.querySelector(".field-thumbnail_preview");
    if (previewCell) {
      return { kind: "cell", node: previewCell };
    }

    var fallback = row.querySelector(".js-inline-image-preview-fallback");
    if (!fallback) {
      fallback = document.createElement("div");
      fallback.className = "js-inline-image-preview-fallback";
      fallback.style.marginTop = "6px";
      fileInput.insertAdjacentElement("afterend", fallback);
    }
    return { kind: "fallback", node: fallback };
  }

  function preserveOriginal(node) {
    if (!node || node.dataset.originalPreviewHtml !== undefined) {
      return;
    }
    node.dataset.originalPreviewHtml = node.innerHTML;
  }

  function revokeObjectUrl(node) {
    if (!node || !node.dataset.previewObjectUrl) {
      return;
    }
    try {
      window.URL.revokeObjectURL(node.dataset.previewObjectUrl);
    } catch (_err) {
      // Ignore browser URL revocation failures.
    }
    delete node.dataset.previewObjectUrl;
  }

  function restoreOriginalPreview(target) {
    if (!target || !target.node) {
      return;
    }
    revokeObjectUrl(target.node);
    if (target.kind === "cell" && target.node.dataset.originalPreviewHtml !== undefined) {
      target.node.innerHTML = target.node.dataset.originalPreviewHtml;
      return;
    }
    target.node.innerHTML = "";
  }

  function renderPreview(target, objectUrl) {
    if (!target || !target.node || !objectUrl) {
      return;
    }
    preserveOriginal(target.node);
    revokeObjectUrl(target.node);
    target.node.dataset.previewObjectUrl = objectUrl;
    target.node.innerHTML =
      '<img src="' +
      objectUrl +
      '" alt="Preview" style="max-height: 56px; max-width: 92px; object-fit: cover; border-radius: 4px; border: 1px solid #d1d5db;" />';
  }

  function findClearCheckbox(fileInput) {
    if (!fileInput || !fileInput.name) {
      return null;
    }
    var row = findRow(fileInput);
    if (!row) {
      return null;
    }
    var clearName = fileInput.name + "-clear";
    return row.querySelector('input[type="checkbox"][name="' + clearName + '"]');
  }

  function handleFileSelection(fileInput) {
    if (!isLikelyImageInput(fileInput)) {
      return;
    }

    var target = getPreviewTarget(fileInput);
    if (!target) {
      return;
    }

    var file = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
    var clearCheckbox = findClearCheckbox(fileInput);
    if (file && String(file.type || "").indexOf("image/") === 0) {
      if (clearCheckbox && clearCheckbox.checked) {
        clearCheckbox.checked = false;
      }
      renderPreview(target, window.URL.createObjectURL(file));
      return;
    }

    if (clearCheckbox && clearCheckbox.checked) {
      preserveOriginal(target.node);
      revokeObjectUrl(target.node);
      target.node.innerHTML =
        '<span style="color: #6b7280; font-size: 12px;">No image selected</span>';
      return;
    }

    restoreOriginalPreview(target);
  }

  function handleClearToggle(clearCheckbox) {
    if (!clearCheckbox || clearCheckbox.type !== "checkbox" || !clearCheckbox.name) {
      return;
    }
    if (!/-image-clear$/.test(clearCheckbox.name)) {
      return;
    }

    var row = findRow(clearCheckbox);
    if (!row) {
      return;
    }

    var fileInputName = clearCheckbox.name.replace(/-clear$/, "");
    var fileInput = row.querySelector('input[type="file"][name="' + fileInputName + '"]');
    if (!fileInput) {
      return;
    }
    handleFileSelection(fileInput);
  }

  function bootstrap() {
    var inputs = document.querySelectorAll('input[type="file"]');
    inputs.forEach(function (input) {
      if (isLikelyImageInput(input)) {
        var target = getPreviewTarget(input);
        if (target && target.kind === "cell") {
          preserveOriginal(target.node);
        }
      }
    });
  }

  function bindListeners() {
    document.addEventListener("change", function (event) {
      var target = event.target;
      if (!target) {
        return;
      }
      if (target.matches && target.matches('input[type="file"]')) {
        handleFileSelection(target);
        return;
      }
      if (target.matches && target.matches('input[type="checkbox"]')) {
        handleClearToggle(target);
      }
    });

    document.addEventListener("formset:added", bootstrap);
    window.addEventListener("beforeunload", function () {
      var previewNodes = document.querySelectorAll("[data-preview-object-url]");
      previewNodes.forEach(revokeObjectUrl);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      bootstrap();
      bindListeners();
    });
  } else {
    bootstrap();
    bindListeners();
  }
})();
