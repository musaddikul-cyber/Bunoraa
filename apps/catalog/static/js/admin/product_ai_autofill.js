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

  function parseJsonResponse(resp) {
    return resp
      .text()
      .then(function (body) {
        var payload = {};
        if (body) {
          try {
            payload = JSON.parse(body);
          } catch (error) {
            payload = {
              ok: false,
              error: "Unexpected response format from server.",
            };
          }
        }
        return {
          ok: resp.ok,
          status: resp.status,
          data: payload || {},
        };
      })
      .catch(function () {
        return {
          ok: false,
          status: 0,
          data: {
            ok: false,
            error: "Unable to read server response.",
          },
        };
      });
  }

  function shouldEnableDebug(config) {
    if (!config) return false;
    if (String(config.dataset.debug || "").toLowerCase() === "true") {
      return true;
    }
    try {
      if (window.localStorage && window.localStorage.getItem("product_ai_debug") === "1") {
        return true;
      }
    } catch (e) {}
    return /(?:\?|&)product_ai_debug=1(?:&|$)/.test(String(window.location.search || ""));
  }

  function createLogger(enabled) {
    function emit(level, args) {
      if (!enabled || !window.console) return;
      var prefix = "[ProductAI]";
      var fn =
        level === "error"
          ? console.error
          : level === "warn"
          ? console.warn
          : console.log;
      if (typeof fn === "function") {
        fn.apply(console, [prefix].concat(Array.prototype.slice.call(args)));
      }
    }
    return {
      info: function () {
        emit("info", arguments);
      },
      warn: function () {
        emit("warn", arguments);
      },
      error: function () {
        emit("error", arguments);
      },
      enabled: enabled,
    };
  }

  function buildClientDiagnostics(files) {
    return {
      page: "product_admin",
      url_path: String(window.location.pathname || ""),
      user_agent: String((window.navigator && window.navigator.userAgent) || "").slice(0, 200),
      timestamp: new Date().toISOString(),
      file_count: Array.isArray(files) ? files.length : 0,
      files: (files || []).map(function (file) {
        return {
          name: String((file && file.name) || "").slice(0, 120),
          size: Number((file && file.size) || 0),
          type: String((file && file.type) || "").slice(0, 80),
        };
      }),
    };
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

  function resolveSuggestionDisplayValue(item) {
    var metadata = (item && item.metadata) || {};

    if (metadata && typeof metadata.name === "string" && metadata.name.trim()) {
      return metadata.name.trim();
    }
    if (metadata && Array.isArray(metadata.names) && metadata.names.length) {
      return metadata.names
        .map(function (value) {
          return String(value || "").trim();
        })
        .filter(Boolean)
        .join(", ");
    }
    if (
      item &&
      item.display_value !== null &&
      item.display_value !== undefined &&
      String(item.display_value).trim() !== ""
    ) {
      return item.display_value;
    }
    return item ? item.value : "";
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
      var value = resolveSuggestionDisplayValue(item);
      if (Array.isArray(value)) {
        value = value.join(", ");
      } else if (value && typeof value === "object") {
        value = JSON.stringify(value);
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
    var debugLogger = createLogger(shouldEnableDebug(config));

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
    var aiEnabled = config.dataset.enabled === "true";
    var disabledReason = config.dataset.disabledReason || "AI autofill is disabled.";

    var currentJobId = null;
    var currentSuggestions = [];
    var applyInFlight = false;
    debugLogger.info("Initialized", {
      aiEnabled: aiEnabled,
      productIdPresent: Boolean(productId),
      maxImages: maxImages,
    });

    if (!aiEnabled) {
      startBtn.disabled = true;
      applyBtn.disabled = true;
      setStatus(disabledReason, true);
      renderSuggestions([]);
      debugLogger.warn("AI disabled", { reason: disabledReason });
      return;
    }

    function pollJob(jobId) {
      var statusUrl = formatJobUrl(statusTemplate, jobId);
      fetch(statusUrl, { credentials: "same-origin" })
        .then(function (resp) {
          return parseJsonResponse(resp);
        })
        .then(function (result) {
          var data = result.data || {};
          if (!data.ok) {
            debugLogger.warn("Status request returned error", {
              jobId: jobId,
              statusCode: result.status,
              payload: data,
            });
            setStatus(data.error || "Unable to fetch AI status.", true);
            return;
          }
          currentSuggestions = data.suggestions || [];
          renderSuggestions(currentSuggestions);
          var summary = data.summary || {};
          var nonNullSuggestions = Number(summary.non_null_suggestions || 0);
          var imagesAnalyzed = Number(summary.images_analyzed || 0);
          var lowCount = currentSuggestions.filter(function (item) {
            return Boolean(item && item.low_confidence);
          }).length;
          debugLogger.info("Job status", {
            jobId: jobId,
            status: data.status,
            progress: data.progress,
            suggestions: currentSuggestions.length,
            lowConfidence: lowCount,
            nonNullSuggestions: nonNullSuggestions,
            imagesAnalyzed: imagesAnalyzed,
          });
          setStatus("Status: " + data.status + " (" + data.progress + "%)", false);
          if (data.status === "completed") {
            applyBtn.disabled = false;
            if (!nonNullSuggestions) {
              setStatus(
                "Analysis completed, but no reliable fields were extracted. Check image quality and see console logs.",
                true
              );
              debugLogger.warn("Completed with zero non-null suggestions", {
                summary: summary,
                error: data.error_message || "",
              });
            }
            return;
          }
          if (data.status === "failed" || data.status === "cancelled") {
            applyBtn.disabled = true;
            setStatus(data.error_message || "AI analysis failed.", true);
            debugLogger.error("Job failed/cancelled", {
              jobId: jobId,
              status: data.status,
              error: data.error_message || "",
              summary: summary,
            });
            return;
          }
          window.setTimeout(function () {
            pollJob(jobId);
          }, 2000);
        })
        .catch(function (error) {
          debugLogger.error("Polling failed", { jobId: jobId, error: String(error || "") });
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
      if (!productId && !files.length) {
        setStatus("Upload at least one image before running analysis.", true);
        debugLogger.warn("Analyze blocked: new product has no selected image files");
        return;
      }

      var fileSummary = files.map(function (file) {
        return {
          name: file && file.name ? String(file.name) : "",
          size: Number((file && file.size) || 0),
          type: file && file.type ? String(file.type) : "",
        };
      });
      debugLogger.info("Starting analysis request", {
        productId: productId || "",
        files: fileSummary,
      });

      files.forEach(function (file) {
        formData.append("images", file);
      });
      var contextHints = collectContextHints(files, maxImages);
      if (Object.keys(contextHints).length) {
        formData.append("context_hints", JSON.stringify(contextHints));
      }
      var diagnostics = buildClientDiagnostics(files);
      diagnostics.context_hint_keys = Object.keys(contextHints || {});
      formData.append("client_diagnostics", JSON.stringify(diagnostics));
      debugLogger.info("Client diagnostics attached", diagnostics);

      fetch(startUrl, {
        method: "POST",
        body: formData,
        credentials: "same-origin",
        headers: {
          "X-CSRFToken": csrfToken,
        },
      })
        .then(function (resp) {
          return parseJsonResponse(resp);
        })
        .then(function (result) {
          var data = result.data || {};
          if (!data.ok) {
            debugLogger.warn("Start request failed", {
              statusCode: result.status,
              payload: data,
            });
            setStatus(data.error || "Failed to start AI analysis", true);
            return;
          }
          currentJobId = data.job_id;
          debugLogger.info("Analysis job created", {
            jobId: currentJobId,
            dispatchMode: data.dispatch_mode || "",
            imageCount: Number(data.image_count || 0),
          });
          setStatus("AI job created. Processing...", false);
          pollJob(currentJobId);
        })
        .catch(function (error) {
          debugLogger.error("Start request failed", { error: String(error || "") });
          setStatus("Failed to start AI analysis.", true);
        });
    });

    applyBtn.addEventListener("click", function () {
      if (applyInFlight) {
        return;
      }
      if (!currentJobId) {
        setStatus("No completed AI job found to apply.", true);
        return;
      }
      var applyUrl = formatJobUrl(applyTemplate, currentJobId);
      var payload = {
        force_overwrite: Boolean(forceCheckbox && forceCheckbox.checked),
      };
      applyInFlight = true;
      applyBtn.disabled = true;
      setStatus("Applying suggestions...", false);
      debugLogger.info("Applying suggestions", {
        jobId: currentJobId,
        forceOverwrite: payload.force_overwrite,
      });
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
          return parseJsonResponse(resp);
        })
        .then(function (result) {
          var data = result.data || {};
          if (!data.ok) {
            debugLogger.warn("Apply request failed", {
              jobId: currentJobId,
              statusCode: result.status,
              payload: data,
            });
            setStatus(data.error || "Failed to apply suggestions", true);
            return;
          }
          if (data.mode === "client_apply") {
            applyToForm(data.fields || {});
            setStatus("Suggestions applied to form fields.", false);
            debugLogger.info("Applied suggestions in client mode", {
              jobId: currentJobId,
              fieldCount: Object.keys(data.fields || {}).length,
            });
          } else {
            var count = (data.result && data.result.applied) || 0;
            var skipped = (data.result && data.result.skipped) || 0;
            if (count === 0 && !payload.force_overwrite) {
              setStatus(
                "No blank fields were eligible. Enable 'Overwrite existing values' and apply again.",
                true
              );
            } else {
              setStatus(
                "Applied " + count + " field suggestions" + (skipped ? " (" + skipped + " skipped)." : "."),
                false
              );
            }
            debugLogger.info("Apply completed in server mode", {
              jobId: currentJobId,
              applied: count,
              skipped: skipped,
            });
          }
        })
        .catch(function (error) {
          debugLogger.error("Apply request failed", {
            jobId: currentJobId,
            error: String(error || ""),
          });
          setStatus("Failed to apply suggestions.", true);
        })
        .finally(function () {
          applyInFlight = false;
          if (currentJobId) {
            applyBtn.disabled = false;
          }
        });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();
