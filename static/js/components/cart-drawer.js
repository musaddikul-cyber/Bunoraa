(function() {
  'use strict';

  function toNumber(value, fallback = 0) {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : fallback;
  }

  // Get configuration from DOM or window.BUNORAA_CURRENCY / window.BUNORAA_SHIPPING
  function getConfig() {
    const cartDrawer = document.getElementById('cart-drawer');
    const currencyMeta = window.BUNORAA_CURRENCY || {};
    const shippingMeta = window.BUNORAA_SHIPPING || {};

    return {
      FREE_SHIPPING_THRESHOLD: toNumber(
        cartDrawer?.dataset?.freeShippingThreshold,
        toNumber(shippingMeta.free_shipping_threshold, toNumber(currencyMeta.free_shipping_threshold, 2000))
      ),
      DELIVERY_TIMES: shippingMeta.delivery_times || {
        dhaka: '1-2 days',
        chittagong: '2-3 days',
        other: '3-5 days'
      },
      CURRENCY_SYMBOL: cartDrawer?.dataset?.currencySymbol || currencyMeta.symbol || '\u09F3',
      CURRENCY_CODE: cartDrawer?.dataset?.currencyCode || currencyMeta.code || 'BDT',
      CURRENCY_POSITION: cartDrawer?.dataset?.currencyPosition || currencyMeta.symbol_position || 'before',
      COD_FEE: toNumber(cartDrawer?.dataset?.codFee, toNumber(shippingMeta.cod_fee, 0)),
      TAX_RATE: toNumber(cartDrawer?.dataset?.taxRate, 10),
      USER_AUTHENTICATED: cartDrawer?.dataset?.userAuthenticated === 'true',
      USER_CITY: cartDrawer?.dataset?.userCity || 'Dhaka',
      USER_POSTAL: cartDrawer?.dataset?.userPostal || '1200'
    };
  }

  // Lazy-initialized config
  let CONFIG = null;
  function cfg() {
    if (!CONFIG) CONFIG = getConfig();
    return CONFIG;
  }

  function fmt(val, alreadyConverted = false) {
    try {
      if (window.Templates?.formatPrice) return window.Templates.formatPrice(val, null, alreadyConverted);
      const c = cfg();
      const formatted = Number(val || 0).toLocaleString('en-BD');
      return c.CURRENCY_POSITION === 'after' ? `${formatted}${c.CURRENCY_SYMBOL}` : `${c.CURRENCY_SYMBOL}${formatted}`;
    } catch { 
      const c = cfg();
      return `${c.CURRENCY_SYMBOL}${Number(val || 0).toFixed(2)}`; 
    }
  }
  
  function esc(s) {
    try { return window.Templates?.escapeHtml ? window.Templates.escapeHtml(s || '') : (s || ''); } catch { return s || ''; }
  }
  
  function getImageUrl(p) {
    if (!p) return '';
    // Prefer thumbnail for drawer performance
    if (typeof p.thumbnail === 'string' && p.thumbnail) return p.thumbnail;
    if (p.primary_image && typeof p.primary_image === 'object') {
      if (typeof p.primary_image.thumbnail === 'string' && p.primary_image.thumbnail) return p.primary_image.thumbnail;
      if (p.primary_image.image && typeof p.primary_image.image === 'object' && typeof p.primary_image.image.thumbnail === 'string' && p.primary_image.image.thumbnail) return p.primary_image.image.thumbnail;
    }
    // Fallbacks
    if (typeof p.primary_image === 'string' && p.primary_image) return p.primary_image;
    if (p.primary_image && typeof p.primary_image === 'object') {
      if (typeof p.primary_image.image === 'string' && p.primary_image.image) return p.primary_image.image;
      if (p.primary_image.image && typeof p.primary_image.image === 'object' && typeof p.primary_image.image.url === 'string') return p.primary_image.image.url;
      if (typeof p.primary_image.url === 'string' && p.primary_image.url) return p.primary_image.url;
    }
    if (typeof p.image === 'string' && p.image) return p.image;
    if (Array.isArray(p.images) && p.images.length) {
      const first = p.images[0];
      if (typeof first === 'string') return first;
      if (first && typeof first === 'object') {
        if (typeof first.thumbnail === 'string' && first.thumbnail) return first.thumbnail;
        if (typeof first.image === 'string' && first.image) return first.image;
        if (first.image && typeof first.image === 'object' && typeof first.image.url === 'string') return first.image.url;
        if (typeof first.url === 'string' && first.url) return first.url;
      }
    }
    if (typeof p.image_url === 'string' && p.image_url) return p.image_url;
    return '';
  }

  const SHIPPING_ADDRESS_MESSAGE_HTML = '<a href="/account/addresses/" class="text-xs font-medium text-amber-600 dark:text-amber-400 underline underline-offset-2 hover:text-amber-700 dark:hover:text-amber-300">Add shipping address to see shipping cost.</a>';
  let shippingAddressCache = null;
  let shippingAddressPromise = null;
  let shippingQuoteCache = { key: null, quote: null };

  function isAuthenticated() {
    return window.AuthApi?.isAuthenticated && AuthApi.isAuthenticated();
  }

  function getAddressesFromResponse(response) {
    if (!response) return [];
    if (Array.isArray(response)) return response;
    if (Array.isArray(response.data)) return response.data;
    if (Array.isArray(response.data?.results)) return response.data.results;
    if (Array.isArray(response.results)) return response.results;
    return [];
  }

  async function getDefaultShippingAddress() {
    if (!isAuthenticated()) return null;
    if (shippingAddressCache) return shippingAddressCache;
    if (!shippingAddressPromise) {
      shippingAddressPromise = (async () => {
        try {
          const resp = await AuthApi.getAddresses();
          const addresses = getAddressesFromResponse(resp);
          if (!addresses.length) return null;

          const shippingAddresses = addresses.filter(addr => {
            const type = String(addr.address_type || '').toLowerCase();
            return type === 'shipping' || type === 'both';
          });

          return (
            shippingAddresses.find(addr => addr.is_default) ||
            shippingAddresses[0] ||
            addresses.find(addr => addr.is_default) ||
            addresses[0] ||
            null
          );
        } catch (err) {
          return null;
        }
      })();
    }

    shippingAddressCache = await shippingAddressPromise;
    return shippingAddressCache;
  }

  function buildShippingPayload(address, subtotal, itemCount, productIds) {
    return {
      country: address?.country || 'BD',
      state: address?.state || address?.city || '',
      postal_code: address?.postal_code || '',
      subtotal: subtotal,
      weight: 0,
      item_count: itemCount,
      product_ids: productIds
    };
  }

  // Update free shipping progress bar
  function updateFreeShippingProgress(cartDrawer, subtotal) {
    const banner = cartDrawer?.querySelector('[data-free-shipping-banner]');
    const progressBar = cartDrawer?.querySelector('[data-shipping-progress]');
    const message = cartDrawer?.querySelector('[data-shipping-message]');
    
    if (!banner || !progressBar || !message) return;
    
    const c = cfg();
    const remaining = c.FREE_SHIPPING_THRESHOLD - subtotal;
    const progress = Math.min((subtotal / c.FREE_SHIPPING_THRESHOLD) * 100, 100);
    
    progressBar.style.width = `${progress}%`;
    
    if (remaining <= 0) {
      message.textContent = '🎉 You\'ve unlocked FREE shipping!';
      message.classList.add('text-green-600');
      banner.classList.add('bg-green-100', 'dark:bg-green-900/30');
    } else {
      message.textContent = `Add ${fmt(remaining)} more for FREE shipping!`;
      message.classList.remove('text-green-600');
      banner.classList.remove('bg-green-100', 'dark:bg-green-900/30');
    }
  }

  // Calculate tax based on subtotal and tax rate
  function calculateTax(subtotal) {
    const c = cfg();
    if (!c.TAX_RATE || c.TAX_RATE <= 0) return 0;
    return (subtotal * c.TAX_RATE / 100);
  }

  async function fetchShippingRate(subtotal, itemCount, productIds = [], address) {
    if (!address) return null;

    const payload = buildShippingPayload(address, subtotal, itemCount, productIds);
    const cacheKey = [
      address.id || '',
      payload.country,
      payload.state,
      payload.postal_code,
      subtotal,
      itemCount,
      productIds.join(',')
    ].join('|');

    if (shippingQuoteCache.key === cacheKey) {
      return shippingQuoteCache.quote;
    }

    try {
      let response;
      if (window.ShippingApi?.calculateShipping) {
        response = await ShippingApi.calculateShipping(payload);
      } else {
        const currencyCode = window.BUNORAA_CURRENCY?.code;
        response = await fetch('/api/v1/shipping/calculate/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': window.CSRF_TOKEN || document.querySelector('[name=csrfmiddlewaretoken]')?.value || '',
            ...(currencyCode ? { 'X-User-Currency': currencyCode } : {})
          },
          body: JSON.stringify(payload)
        }).then(res => res.json());
      }

      const methods = response?.data?.methods || response?.data?.data?.methods || response?.methods || [];
      if (!Array.isArray(methods) || methods.length === 0) return null;

      const cheapest = methods.reduce((min, m) => 
        parseFloat(m.rate) < parseFloat(min.rate) ? m : min, methods[0]);

      const cost = toNumber(cheapest.rate, 0);
      const quote = {
        cost,
        isFree: cheapest.is_free || cost <= 0,
        display: cheapest.rate_display || fmt(cost)
      };

      shippingQuoteCache = { key: cacheKey, quote };
      return quote;
    } catch (e) {
      console.warn('[Cart] Shipping API error:', e);
      return null;
    }
  }

  // Update cart summary with shipping and tax
  async function updateCartSummary(cartDrawer, cart) {
    const subtotalEl = cartDrawer?.querySelector('[data-cart-subtotal]');
    const shippingEl = cartDrawer?.querySelector('[data-cart-shipping]');
    const taxEl = cartDrawer?.querySelector('[data-cart-tax]');
    const taxLabelEl = cartDrawer?.querySelector('[data-cart-tax-label]');
    const totalEl = cartDrawer?.querySelector('[data-cart-total]');
    const savingsRow = cartDrawer?.querySelector('[data-cart-savings-row]');
    const savingsEl = cartDrawer?.querySelector('[data-cart-savings]');
    const countEl = cartDrawer?.querySelector('[data-cart-count]');
    
    const items = cart.items || [];
    const subtotal = Number(cart.subtotal ?? cart.summary?.subtotal ?? 0) || items.reduce((sum, it) => 
      sum + Number(it.total || it.line_total || it.unit_price || it.current_price || 0) * Number(it.quantity || 1), 0);
    
    // Calculate savings from original prices
    const totalSavings = items.reduce((sum, it) => {
      const original = Number(it.price_at_add || it.original_price || it.unit_price || it.current_price || 0);
      const current = Number(it.unit_price || it.current_price || 0);
      const qty = Number(it.quantity || 1);
      return sum + Math.max(0, (original - current) * qty);
    }, 0);
    
    if (taxLabelEl) {
      taxLabelEl.textContent = `Tax (${cfg().TAX_RATE}%)`;
    }

    // Calculate tax
    const tax = calculateTax(subtotal);
    
    // Get product IDs for shipping calculation
    const productIds = items.map(it => it.product?.id || it.product_id).filter(Boolean);
    const itemCount = items.reduce((sum, it) => sum + Number(it.quantity || 1), 0);
    
    const address = await getDefaultShippingAddress();
    if (!address) {
      if (subtotalEl) subtotalEl.textContent = cart.summary?.formatted_subtotal || fmt(subtotal);
      if (shippingEl) shippingEl.innerHTML = SHIPPING_ADDRESS_MESSAGE_HTML;
      if (taxEl) taxEl.textContent = fmt(tax);
      if (totalEl) totalEl.textContent = fmt(subtotal + tax);
      if (countEl) countEl.textContent = itemCount;
      updateFreeShippingProgress(cartDrawer, subtotal);
      return;
    }

    // Show loading state for shipping
    if (shippingEl) shippingEl.textContent = 'Calculating...';

    let shipping = null;
    if (items.length === 0) {
      shipping = { cost: 0, isFree: true, display: 'Free' };
    } else {
      shipping = await fetchShippingRate(subtotal, itemCount, productIds, address);
    }

    if (!shipping) {
      if (subtotalEl) subtotalEl.textContent = cart.summary?.formatted_subtotal || fmt(subtotal);
      if (shippingEl) shippingEl.innerHTML = SHIPPING_ADDRESS_MESSAGE_HTML;
      if (taxEl) taxEl.textContent = fmt(tax);
      if (totalEl) totalEl.textContent = fmt(subtotal + tax);
      if (countEl) countEl.textContent = itemCount;
      updateFreeShippingProgress(cartDrawer, subtotal);
      return;
    }

    const total = subtotal + shipping.cost + tax;

    // Update elements
    if (subtotalEl) subtotalEl.textContent = cart.summary?.formatted_subtotal || fmt(subtotal);
    if (shippingEl) shippingEl.textContent = shipping.isFree ? 'FREE' : (shipping.display || fmt(shipping.cost));
    if (taxEl) taxEl.textContent = fmt(tax);
    if (totalEl) totalEl.textContent = fmt(total);
    if (countEl) countEl.textContent = itemCount;
    
    // Show/hide savings
    if (savingsRow && savingsEl) {
      if (totalSavings > 0) {
        savingsRow.classList.remove('hidden');
        savingsEl.textContent = `-${fmt(totalSavings)}`;
      } else {
        savingsRow.classList.add('hidden');
      }
    }
    
    // Update free shipping progress
    updateFreeShippingProgress(cartDrawer, subtotal);
  }

  function render(cartDrawer, cart) {
    const cartContent = cartDrawer?.querySelector('[data-cart-content]');
    if (!cartContent || !cart) return;

    const items = cart.items || [];
    cartDrawer?.classList.toggle('cart-empty', items.length === 0);

    // Update summary (uses user's location from config or defaults to Dhaka for guests)
    updateCartSummary(cartDrawer, cart);

    if (items.length === 0) {
      cartContent.innerHTML = `
        <div class="p-8 text-center">
          <div class="mx-auto w-16 h-16 rounded-full bg-stone-100 dark:bg-stone-800 flex items-center justify-center mb-4">
            <svg class="w-8 h-8 text-stone-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z"/></svg>
          </div>
          <h3 class="text-lg font-semibold text-stone-900 dark:text-white mb-2">Your bag is empty</h3>
          <p class="text-stone-600 dark:text-stone-300 mb-4">Looks like you haven't added anything yet</p>
          <a href="/products/" class="inline-flex items-center px-6 py-3 bg-amber-600 hover:bg-amber-700 text-white font-medium rounded-xl transition-colors">
            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z"/></svg>
            Start Shopping
          </a>
        </div>`;
      return;
    }

    // Map summary items (if present) for pre-formatted values
    const summaryItemsMap = (cart.summary && Array.isArray(cart.summary.items)) 
      ? cart.summary.items.reduce((m, it) => { m[it.id] = it; return m; }, {}) 
      : {};

    const itemHtml = items.map(item => {
      const product = item.product || {};
      const variant = item.variant;
      const href = item.product_slug ? `/products/${item.product_slug}/` : (product.slug ? `/products/${product.slug}/` : '#');
      const img = item.product_image || getImageUrl(product);
      const unit = Number(item.unit_price ?? item.current_price ?? product.price ?? 0);
      const originalPrice = Number(item.price_at_add ?? item.original_price ?? unit);
      const qty = Number(item.quantity || 1);
      const line = Number(item.total ?? item.line_total ?? unit * qty);
      const hasDiscount = originalPrice > unit;
      const discountPercent = hasDiscount ? Math.round((1 - unit / originalPrice) * 100) : 0;
      const stockQuantity = item.stock_quantity || product.stock_quantity || 999;
      const isLowStock = stockQuantity <= 5 && stockQuantity > 0;

      const summaryItem = summaryItemsMap[item.id];
      const formattedUnit = summaryItem?.formatted_unit_price || fmt(unit, true);
      const formattedOriginal = fmt(originalPrice, true);
      const formattedLine = summaryItem?.formatted_total || fmt(line, true);

      return `
        <div class="group relative p-4 border-b border-stone-100 dark:border-stone-800 hover:bg-stone-50 dark:hover:bg-stone-800/50 transition-colors" data-cart-item-id="${item.id}">
          <div class="flex gap-4">
            <a href="${href}" class="w-20 h-20 rounded-xl overflow-hidden bg-stone-100 dark:bg-stone-800 flex-shrink-0 relative">
              <img src="${img}" alt="${esc(product.name || 'Product')}" class="w-full h-full object-cover" loading="lazy">
              ${hasDiscount ? `<span class="absolute top-1 left-1 px-1.5 py-0.5 text-[10px] font-bold bg-red-500 text-white rounded">-${discountPercent}%</span>` : ''}
            </a>
            <div class="flex-1 min-w-0">
              <div class="flex items-start justify-between gap-2">
                <div class="flex-1">
                  <a href="${href}" class="font-semibold text-stone-900 dark:text-white line-clamp-2 hover:text-amber-700 dark:hover:text-amber-400 text-sm">${esc(item.product_name || product.name || 'Product')}</a>
                  ${variant || item.variant_name ? `<p class="text-xs text-stone-500 dark:text-stone-400 mt-0.5">${esc(variant?.name || variant?.value || item.variant_name || '')}</p>` : ''}
                  ${isLowStock ? `<p class="text-xs text-amber-600 dark:text-amber-400 mt-0.5">Only ${stockQuantity} left</p>` : ''}
                </div>
                <div class="flex flex-col gap-1">
                  <button class="p-1 text-stone-400 hover:text-amber-600 transition-colors" data-cart-save="${item.id}" title="Save for later" aria-label="Save for later">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg>
                  </button>
                  <button class="p-1 text-stone-400 hover:text-red-500 transition-colors" data-cart-remove="${item.id}" aria-label="Remove">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                  </button>
                </div>
              </div>
              <div class="mt-2 flex items-end justify-between">
                <div class="flex items-center gap-1">
                  <button class="w-7 h-7 rounded-lg border border-stone-200 dark:border-stone-700 text-stone-700 dark:text-stone-200 hover:bg-stone-100 dark:hover:bg-stone-800 text-sm font-medium disabled:opacity-50" data-qty-minus="${item.id}" ${qty <= 1 ? 'disabled' : ''}>−</button>
                  <span class="min-w-[2rem] text-center font-medium text-sm" data-qty-value="${item.id}">${qty}</span>
                  <button class="w-7 h-7 rounded-lg border border-stone-200 dark:border-stone-700 text-stone-700 dark:text-stone-200 hover:bg-stone-100 dark:hover:bg-stone-800 text-sm font-medium disabled:opacity-50" data-qty-plus="${item.id}" ${qty >= stockQuantity ? 'disabled' : ''}>+</button>
                </div>
                <div class="text-right">
                  ${hasDiscount ? `<div class="text-xs text-stone-400 line-through">${formattedOriginal}</div>` : ''}
                  <div class="font-bold text-stone-900 dark:text-white" data-line-total="${item.id}">${formattedLine}</div>
                </div>
              </div>
            </div>
          </div>
        </div>`;
    }).join('');

    cartContent.innerHTML = itemHtml;

    // Bind event handlers
    bindItemEventHandlers(cartDrawer, cartContent);
  }

  function bindItemEventHandlers(cartDrawer, cartContent) {
    // Remove item
    cartContent.querySelectorAll('[data-cart-remove]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const id = btn.dataset.cartRemove;
        btn.disabled = true;
        try {
          const resp = await window.CartApi.removeItem(id);
          if (resp.success) {
            render(cartDrawer, resp.data?.cart);
            window.Toast?.success('Item removed from cart');
          }
        } catch (err) {
          console.error(err);
          window.Toast?.error('Failed to remove item');
        } finally {
          btn.disabled = false;
        }
      });
    });

    // Save for later
    cartContent.querySelectorAll('[data-cart-save]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const id = btn.dataset.cartSave;
        btn.disabled = true;
        try {
          // If SaveForLaterApi exists, use it; otherwise fall back to removing
          if (window.SaveForLaterApi?.saveItem) {
            const resp = await window.SaveForLaterApi.saveItem(id);
            if (resp.success) {
              render(cartDrawer, resp.data?.cart);
              window.Toast?.success('Item saved for later');
              updateSavedSection(cartDrawer);
            }
          } else {
            window.Toast?.info('Save for later coming soon!');
          }
        } catch (err) {
          console.error(err);
          window.Toast?.error('Failed to save item');
        } finally {
          btn.disabled = false;
        }
      });
    });

    // Quantity adjust
    const updateQty = async (id, nextQty) => {
      if (nextQty < 1) return;
      try {
        const resp = await window.CartApi.updateItem(id, nextQty);
        if (resp.success) {
          render(cartDrawer, resp.data?.cart);
        }
      } catch (err) {
        console.error(err);
        window.Toast?.error('Failed to update quantity');
      }
    };

    cartContent.querySelectorAll('[data-qty-minus]').forEach(btn => {
      btn.addEventListener('click', () => {
        const id = btn.dataset.qtyMinus;
        const valEl = cartContent.querySelector(`[data-qty-value="${id}"]`);
        const current = Number(valEl?.textContent || '1');
        if (current > 1) updateQty(id, current - 1);
      });
    });

    cartContent.querySelectorAll('[data-qty-plus]').forEach(btn => {
      btn.addEventListener('click', () => {
        const id = btn.dataset.qtyPlus;
        const valEl = cartContent.querySelector(`[data-qty-value="${id}"]`);
        const current = Number(valEl?.textContent || '1');
        updateQty(id, current + 1);
      });
    });
  }

  // Update saved for later section
  async function updateSavedSection(cartDrawer) {
    const savedSection = cartDrawer?.querySelector('[data-saved-section]');
    const savedContent = cartDrawer?.querySelector('[data-saved-content]');
    const savedCount = cartDrawer?.querySelector('[data-saved-count]');
    
    if (!savedSection || !savedContent) return;
    
    try {
      if (window.SaveForLaterApi?.getItems) {
        const resp = await window.SaveForLaterApi.getItems();
        if (resp.success && resp.data?.items?.length > 0) {
          savedSection.classList.remove('hidden');
          if (savedCount) savedCount.textContent = resp.data.items.length;
          
          savedContent.innerHTML = resp.data.items.map(item => `
            <div class="flex items-center gap-3 p-3 hover:bg-stone-50 dark:hover:bg-stone-800">
              <img src="${getImageUrl(item.product || {})}" alt="${esc(item.product?.name || '')}" class="w-10 h-10 rounded object-cover">
              <div class="flex-1 min-w-0">
                <p class="text-sm font-medium text-stone-900 dark:text-white truncate">${esc(item.product?.name || 'Product')}</p>
                <p class="text-xs text-stone-500">${fmt(item.current_price || 0)}</p>
              </div>
              <button class="text-xs text-amber-600 hover:text-amber-700 font-medium" data-move-to-cart="${item.id}">Add to Cart</button>
            </div>
          `).join('');
          
          // Bind move to cart handlers
          savedContent.querySelectorAll('[data-move-to-cart]').forEach(btn => {
            btn.addEventListener('click', async () => {
              const id = btn.dataset.moveToCart;
              try {
                if (window.SaveForLaterApi?.moveToCart) {
                  const resp = await window.SaveForLaterApi.moveToCart(id);
                  if (resp.success) {
                    render(cartDrawer, resp.data?.cart);
                    updateSavedSection(cartDrawer);
                    window.Toast?.success('Item moved to cart');
                  }
                }
              } catch (err) {
                console.error(err);
              }
            });
          });
        } else {
          savedSection.classList.add('hidden');
        }
      }
    } catch (err) {
      console.error('Failed to load saved items:', err);
    }
  }

  // Toggle saved section visibility
  function setupSavedToggle(cartDrawer) {
    const toggleBtn = cartDrawer?.querySelector('[data-toggle-saved]');
    const savedContent = cartDrawer?.querySelector('[data-saved-content]');
    const chevron = cartDrawer?.querySelector('[data-saved-chevron]');
    
    if (toggleBtn && savedContent) {
      toggleBtn.addEventListener('click', () => {
        savedContent.classList.toggle('hidden');
        chevron?.classList.toggle('rotate-180');
      });
    }
  }

  async function open(cartDrawer) {
    if (!cartDrawer) return;
    const cartContent = cartDrawer.querySelector('[data-cart-content]');
    if (cartContent) {
      cartContent.innerHTML = `
        <div class="p-6 text-center">
          <div class="inline-block animate-spin w-8 h-8 border-3 border-amber-600 border-t-transparent rounded-full"></div>
          <p class="mt-2 text-stone-500 text-sm dark:text-stone-400">Loading your cart...</p>
        </div>`;
    }
    try {
      const response = await window.CartApi.getCart();
      if (!response.success) throw new Error('Failed to load cart.');
      const cart = response.data;
      const items = cart?.items || [];
      
      // Don't open drawer if cart is empty
      if (items.length === 0) {
        window.Toast?.info('Your cart is empty. Start shopping!', { duration: 3000 });
        return;
      }
      
      cartDrawer.classList.remove('hidden');
      requestAnimationFrame(() => cartDrawer.classList.add('open'));
      render(cartDrawer, cart);
      
      // Also update saved section
      updateSavedSection(cartDrawer);
    } catch (error) {
      window.Toast?.error(error.message || 'Unable to open cart right now.');
    }
  }

  function close(cartDrawer) {
    if (!cartDrawer) return;
    cartDrawer.classList.remove('open');
    setTimeout(() => {
      cartDrawer.classList.add('hidden');
    }, 250);
  }

  function init() {
    const cartDrawer = document.getElementById('cart-drawer');
    if (!cartDrawer) return;

    // Open cart buttons
    document.querySelectorAll('[data-cart-open]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        open(cartDrawer);
      });
    });

    // Close handlers
    cartDrawer.querySelectorAll('[data-cart-close]').forEach(btn => 
      btn.addEventListener('click', () => close(cartDrawer))
    );
    cartDrawer.addEventListener('click', (e) => { 
      if (e.target === cartDrawer) close(cartDrawer); 
    });
    document.addEventListener('keydown', (e) => { 
      if (e.key === 'Escape' && cartDrawer.classList.contains('open')) close(cartDrawer); 
    });

    // Setup additional handlers
    setupSavedToggle(cartDrawer);

    // Listen for cart updates
    document.addEventListener('cart:updated', async () => {
      if (cartDrawer.classList.contains('open')) {
        const response = await window.CartApi.getCart();
        if (response.success) render(cartDrawer, response.data);
      }
    });

    // Listen for add to cart events (refresh cart)
    document.addEventListener('cart:item-added', async () => {
      const response = await window.CartApi.getCart();
      if (response.success) {
        // Update cart count in header if needed
        const countBadges = document.querySelectorAll('[data-cart-count]');
        const itemCount = (response.data?.items || []).reduce((sum, it) => sum + Number(it.quantity || 1), 0);
        countBadges.forEach(badge => badge.textContent = itemCount);
      }
    });
  }

  document.addEventListener('DOMContentLoaded', init);
  
  // Export for external use
  window.CartDrawer = { 
    open: () => open(document.getElementById('cart-drawer')), 
    close: () => close(document.getElementById('cart-drawer')), 
    render, 
    init,
    CONFIG
  };
})();
