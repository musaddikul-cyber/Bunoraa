(function () {
  "use strict";

  function getCSRFToken() {
    var input = document.querySelector('input[name="csrfmiddlewaretoken"]');
    if (input && input.value) {
      return input.value;
    }
    var cookie = document.cookie
      .split(";")
      .map(function (c) {
        return c.trim();
      })
      .find(function (c) {
        return c.indexOf("csrftoken=") === 0;
      });
    return cookie ? decodeURIComponent(cookie.split("=")[1]) : "";
  }

  function formatJobUrl(template, jobId) {
    return template.replace(/[0-9a-fA-F-]{36}/, jobId);
  }

  function collectImageFiles(maxImages) {
    var inputs = Array.prototype.slice.call(
      document.querySelectorAll('input[type="file"]')
    );
    var files = [];
    inputs.forEach(function (input) {
      if (!input.files || !input.files.length) {
        return;
      }
      Array.prototype.forEach.call(input.files, function (file) {
        if (files.length < maxImages) {
          files.push(file);
        }
      });
    });
    return files.slice(0, maxImages);
  }

  function selectedOptionTexts(selectEl, limit) {
    if (!selectEl) return [];
    var labels = [];
    Array.prototype.forEach.call(selectEl.options || [], function (option) {
      if (option.selected && option.text) {
        labels.push(String(option.text).trim());
      }
    });
    return labels.filter(Boolean).slice(0, limit || 12);
  }

  function selectedOptionValues(selectEl, limit) {
    if (!selectEl) return [];
    var values = [];
    Array.prototype.forEach.call(selectEl.options || [], function (option) {
      if (option.selected && option.value) {
        values.push(String(option.value).trim());
      }
    });
    return values.filter(Boolean).slice(0, limit || 12);
  }

  function getMultiSelectElement(fieldName) {
    return (
      document.getElementById("id_" + fieldName + "_to") ||
      document.getElementById("id_" + fieldName)
    );
  }

  function selectedPrimaryCategoryLabel() {
    var selectedNode = document.querySelector(
      ".category-tree-widget-container .category-tree-node.is-selected span:last-child"
    );
    if (!selectedNode || !selectedNode.textContent) {
      return "";
    }
    return selectedNode.textContent.trim();
  }

  function collectContextHints(files, maxImages) {
    var hints = {};

    var nameInput = document.getElementById("id_name");
    if (nameInput && nameInput.value && String(nameInput.value).trim()) {
      hints.name = String(nameInput.value).trim();
    }

    var shortDescriptionInput = document.getElementById("id_short_description");
    if (
      shortDescriptionInput &&
      shortDescriptionInput.value &&
      String(shortDescriptionInput.value).trim()
    ) {
      hints.short_description = String(shortDescriptionInput.value).trim().slice(0, 500);
    }

    var descriptionInput = document.getElementById("id_description");
    if (descriptionInput && descriptionInput.value && String(descriptionInput.value).trim()) {
      hints.description = String(descriptionInput.value).trim().slice(0, 1200);
    }

    var primaryCategoryInput = document.getElementById("id_primary_category");
    if (
      primaryCategoryInput &&
      primaryCategoryInput.value &&
      String(primaryCategoryInput.value).trim()
    ) {
      hints.primary_category_id = String(primaryCategoryInput.value).trim();
    }
    var primaryCategoryName = selectedPrimaryCategoryLabel();
    if (primaryCategoryName) {
      hints.primary_category_name = primaryCategoryName;
    }

    var categoriesSelect = getMultiSelectElement("categories");
    var categoryIds = selectedOptionValues(categoriesSelect, 12);
    var categoryNames = selectedOptionTexts(categoriesSelect, 12);
    if (categoryIds.length) {
      hints.category_ids = categoryIds;
    }
    if (categoryNames.length) {
      hints.category_names = categoryNames;
    }

    var tagsSelect = getMultiSelectElement("tags");
    var tagNames = selectedOptionTexts(tagsSelect, 12);
    if (tagNames.length) {
      hints.tag_names = tagNames;
    }

    var certSelect = getMultiSelectElement("eco_certifications");
    var certNames = selectedOptionTexts(certSelect, 8);
    if (certNames.length) {
      hints.eco_certification_names = certNames;
    }

    if (Array.isArray(files) && files.length) {
      hints.image_names = files
        .map(function (file) {
          return file && file.name ? String(file.name).trim() : "";
        })
        .filter(Boolean)
        .slice(0, maxImages || 4);
    }

    return hints;
  }

  function setStatus(message, isError) {
    var statusNode = document.getElementById("product-ai-status-text");
    if (!statusNode) return;
    statusNode.textContent = message || "";
    statusNode.style.color = isError ? "#b91c1c" : "#374151";
  }

  function renderSuggestions(suggestions) {
    var container = document.getElementById("product-ai-suggestions-container");
    if (!container) return;
    if (!suggestions || !suggestions.length) {
      container.innerHTML =
        '<p class="help">No suggestions yet. Run analysis first.</p>';
      return;
    }

    var html =
      '<table class="listing" style="margin-top:8px;width:100%;">' +
      "<thead><tr>" +
      "<th>Field</th><th>Value</th><th>Confidence</th><th>Rationale</th><th>Sources</th>" +
      "</tr></thead><tbody>";
    suggestions.forEach(function (item) {
      var value = item.value;
      if (Array.isArray(value)) {
        value = value.join(", ");
      } else if (value === null || value === undefined || value === "") {
        value = "<em>null</em>";
      } else {
        value = String(value);
      }
      var confidence = Number(item.confidence || 0).toFixed(2);
      var sources = (item.source_urls || [])
        .slice(0, 3)
        .map(function (url) {
          return '<a href="' + url + '" target="_blank" rel="noopener">link</a>';
        })
        .join(", ");
      html +=
        "<tr>" +
        "<td>" +
        item.field_name +
        (item.low_confidence ? ' <span style="color:#b45309;">(low)</span>' : "") +
        "</td>" +
        "<td>" +
        value +
        "</td>" +
        "<td>" +
        confidence +
        "</td>" +
        "<td>" +
        (item.rationale || "") +
        "</td>" +
        "<td>" +
        sources +
        "</td>" +
        "</tr>";
    });
    html += "</tbody></table>";
    container.innerHTML = html;
  }

  function setSelectValues(selectEl, values) {
    if (!selectEl) return;
    var selectedSet = new Set((values || []).map(String));
    Array.prototype.forEach.call(selectEl.options, function (option) {
      option.selected = selectedSet.has(option.value);
    });
    if (/_to$/.test(String(selectEl.id || "")) && typeof window.SelectFilter !== "undefined") {
      var fieldId = String(selectEl.id).replace(/_to$/, "");
      if (typeof window.SelectFilter.refresh_icons === "function") {
        window.SelectFilter.refresh_icons(fieldId);
      }
      if (typeof window.SelectFilter.refresh_filtered_selects === "function") {
        window.SelectFilter.refresh_filtered_selects(fieldId);
      }
      if (typeof window.SelectFilter.refresh_filtered_warning === "function") {
        window.SelectFilter.refresh_filtered_warning(fieldId);
      }
    }
    selectEl.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function applyToForm(fields) {
    Object.keys(fields || {}).forEach(function (fieldName) {
      var value = fields[fieldName];
      if (value === null || value === undefined) {
        return;
      }
      if (fieldName === "categories" || fieldName === "tags" || fieldName === "eco_certifications") {
        var select = getMultiSelectElement(fieldName);
        setSelectValues(select, value);
        return;
      }
      if (fieldName === "primary_category") {
        var hiddenInput = document.getElementById("id_primary_category");
        if (hiddenInput) {
          hiddenInput.value = value;
          hiddenInput.dispatchEvent(new Event("change", { bubbles: true }));
        }
        return;
      }
      if (fieldName === "shipping_material") {
        var shippingSelect = document.getElementById("id_shipping_material");
        if (shippingSelect) {
          shippingSelect.value = value;
          shippingSelect.dispatchEvent(new Event("change", { bubbles: true }));
        }
        return;
      }
      var input = document.getElementById("id_" + fieldName);
      if (!input) return;
      input.value = value;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }

  function initialize() {
    var config = document.getElementById("product-ai-config");
    if (!config) return;

    var startBtn = document.getElementById("product-ai-analyze-btn");
    var applyBtn = document.getElementById("product-ai-apply-btn");
    var forceCheckbox = document.getElementById("product-ai-force-overwrite");
    if (!startBtn || !applyBtn) return;

    var csrfToken = getCSRFToken();
    var productId = config.dataset.productId || "";
    var startUrl = config.dataset.startUrl;
    var statusTemplate = config.dataset.statusTemplate;
    var applyTemplate = config.dataset.applyTemplate;
    var maxImages = parseInt(config.dataset.maxImages || "4", 10);

    var currentJobId = null;
    var currentSuggestions = [];

    function pollJob(jobId) {
      var statusUrl = formatJobUrl(statusTemplate, jobId);
      fetch(statusUrl, { credentials: "same-origin" })
        .then(function (resp) {
          return resp.json();
        })
        .then(function (data) {
          if (!data.ok) {
            setStatus(data.error || "Unable to fetch AI status", true);
            return;
          }
          currentSuggestions = data.suggestions || [];
          renderSuggestions(currentSuggestions);
          setStatus("Status: " + data.status + " (" + data.progress + "%)", false);
          if (data.status === "completed") {
            applyBtn.disabled = false;
            return;
          }
          if (data.status === "failed" || data.status === "cancelled") {
            applyBtn.disabled = true;
            setStatus(data.error_message || "AI analysis failed.", true);
            return;
          }
          window.setTimeout(function () {
            pollJob(jobId);
          }, 2000);
        })
        .catch(function () {
          setStatus("Failed to poll AI status endpoint.", true);
        });
    }

    startBtn.addEventListener("click", function () {
      applyBtn.disabled = true;
      renderSuggestions([]);
      setStatus("Starting AI analysis...", false);

      var formData = new FormData();
      if (productId) {
        formData.append("product_id", productId);
      }
      var currencyField = document.getElementById("id_currency");
      if (currencyField && currencyField.value) {
        formData.append("currency", currencyField.value);
      }
      var locale = document.documentElement.getAttribute("lang") || "en";
      formData.append("locale", locale);
      formData.append("allow_external", "true");

      var files = collectImageFiles(maxImages);
      files.forEach(function (file) {
        formData.append("images", file);
      });
      var contextHints = collectContextHints(files, maxImages);
      if (Object.keys(contextHints).length) {
        formData.append("context_hints", JSON.stringify(contextHints));
      }

      fetch(startUrl, {
        method: "POST",
        body: formData,
        credentials: "same-origin",
        headers: {
          "X-CSRFToken": csrfToken,
        },
      })
        .then(function (resp) {
          return resp.json();
        })
        .then(function (data) {
          if (!data.ok) {
            setStatus(data.error || "Failed to start AI analysis", true);
            return;
          }
          currentJobId = data.job_id;
          setStatus("AI job created. Processing...", false);
          pollJob(currentJobId);
        })
        .catch(function () {
          setStatus("Failed to start AI analysis.", true);
        });
    });

    applyBtn.addEventListener("click", function () {
      if (!currentJobId) {
        setStatus("No completed AI job found to apply.", true);
        return;
      }
      var applyUrl = formatJobUrl(applyTemplate, currentJobId);
      var payload = {
        force_overwrite: Boolean(forceCheckbox && forceCheckbox.checked),
      };
      fetch(applyUrl, {
        method: "POST",
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": csrfToken,
        },
        body: JSON.stringify(payload),
      })
        .then(function (resp) {
          return resp.json();
        })
        .then(function (data) {
          if (!data.ok) {
            setStatus(data.error || "Failed to apply suggestions", true);
            return;
          }
          if (data.mode === "client_apply") {
            applyToForm(data.fields || {});
            setStatus("Suggestions applied to form fields.", false);
          } else {
            var count = (data.result && data.result.applied) || 0;
            setStatus("Applied " + count + " field suggestions.", false);
          }
        })
        .catch(function () {
          setStatus("Failed to apply suggestions.", true);
        });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();
