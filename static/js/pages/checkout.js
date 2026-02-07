/**
 * Checkout Page
 * @module pages/checkout
 */

const CheckoutPage = (async function() {
    'use strict';

    let cart = null;
    let checkoutData = {
        shipping_address: null,
        billing_address: null,
        same_as_shipping: true,
        shipping_method: null,
        payment_method: null,
        notes: ''
    };
    let currentStep = 1;
    let cartLoadFailed = false;
    let cartLoadError = null;
    const toNumber = (value, fallback = 0) => {
        if (value === null || value === undefined || value === '') return fallback;
        const num = Number(value);
        return Number.isFinite(num) ? num : fallback;
    };
    const SHIPPING_STORAGE_KEY = 'checkout:shipping';
    const SHIPPING_CACHE_TTL = 1000 * 60 * 60 * 24 * 7;

    function readStoredShipping() {
        try {
            const raw = localStorage.getItem(SHIPPING_STORAGE_KEY);
            if (!raw) return null;
            const data = JSON.parse(raw);
            if (!data || typeof data !== 'object') return null;
            if (data.ts && Date.now() - data.ts > SHIPPING_CACHE_TTL) return null;
            if (data.cost !== undefined && data.cost !== null) {
                const cost = Number(data.cost);
                if (!Number.isFinite(cost)) return null;
                data.cost = cost;
            }
            return data;
        } catch (err) {
            return null;
        }
    }

    function getCheckedShippingInput() {
        return document.querySelector('input[name="shipping_rate_id"]:checked') ||
            document.querySelector('input[name="shipping_method"]:checked');
    }

    function resolveShippingCost(container, cartPayload) {
        const checked = getCheckedShippingInput();
        if (checked) {
            const cost = toNumber(checked.dataset.price, NaN);
            if (Number.isFinite(cost)) {
                return {
                    cost,
                    display: checked.dataset.display || null,
                    currency: checked.dataset.currency || null,
                    source: 'checked'
                };
            }
        }

        const stored = readStoredShipping();
        if (stored && Number.isFinite(Number(stored.cost))) {
            return {
                cost: Number(stored.cost),
                display: stored.display || null,
                currency: stored.currency || null,
                source: 'stored'
            };
        }

        const shippingEl = container?.querySelector('#shipping-cost') || document.getElementById('shipping-cost');
        const existingRaw = shippingEl?.dataset?.price;
        if (existingRaw !== undefined && existingRaw !== '') {
            const cost = toNumber(existingRaw, NaN);
            if (Number.isFinite(cost)) {
                const text = shippingEl?.textContent?.trim() || '';
                const display = text && text.toLowerCase() !== 'calculated next' ? text : null;
                return {
                    cost,
                    display,
                    currency: shippingEl?.dataset?.currency || null,
                    source: 'dom'
                };
            }
        }

        if (cartPayload && cartPayload.shipping_cost !== undefined && cartPayload.shipping_cost !== null) {
            const cost = toNumber(cartPayload.shipping_cost, NaN);
            if (Number.isFinite(cost) && cost > 0) {
                return {
                    cost,
                    display: null,
                    currency: null,
                    source: 'cart'
                };
            }
        }

        return null;
    }

    function formatShippingDisplay(info) {
        if (!info || info.cost === null || Number.isNaN(info.cost)) {
            return 'Calculated next';
        }
        if (info.display && String(info.display).trim() && String(info.display).toLowerCase() !== 'calculated next') {
            return String(info.display);
        }
        if (info.cost <= 0) return 'Free';
        return Templates.formatPrice(info.cost, info.currency || null);
    }

    function getPaymentFeeMeta() {
        const selected = document.querySelector('input[name="payment_method"]:checked');
        if (selected) {
            return {
                type: selected.dataset.feeType || 'none',
                amount: toNumber(selected.dataset.feeAmount, 0),
                percent: toNumber(selected.dataset.feePercent, 0),
                name: selected.dataset.feeName || ''
            };
        }

        const summary = document.getElementById('order-summary');
        if (summary) {
            const feeAmount = toNumber(summary.dataset.paymentFee, 0);
            return {
                type: feeAmount > 0 ? 'flat' : 'none',
                amount: feeAmount,
                percent: 0,
                name: summary.dataset.paymentFeeLabel || ''
            };
        }

        return { type: 'none', amount: 0, percent: 0, name: '' };
    }

    function computePaymentFee(total, meta) {
        if (!meta) return 0;
        if (meta.type === 'flat') return toNumber(meta.amount, 0);
        if (meta.type === 'percent') return Math.max(0, total * (toNumber(meta.percent, 0) / 100));
        return 0;
    }

    function updatePaymentFeeRow(totalValue = null) {
        const row = document.getElementById('payment-fee-row');
        const amountEl = document.getElementById('payment-fee-amount');
        const labelEl = document.getElementById('payment-fee-label');
        if (!row || !amountEl) return;

        const totalEl = document.getElementById('order-total');
        const total = totalValue !== null ? totalValue : toNumber(totalEl?.dataset?.price ?? totalEl?.textContent, 0);
        const meta = getPaymentFeeMeta();
        const feeAmount = computePaymentFee(total, meta);

        if (!feeAmount || feeAmount <= 0) {
            row.classList.add('hidden');
            return;
        }

        row.classList.remove('hidden');
        amountEl.textContent = Templates.formatPrice(feeAmount);
        if (labelEl) {
            labelEl.textContent = meta?.name ? `Extra payment fee (${meta.name})` : 'Extra payment fee';
        }

        const summary = document.getElementById('order-summary');
        if (summary) {
            summary.dataset.paymentFee = feeAmount;
            summary.dataset.paymentFeeLabel = meta?.name || '';
        }
    }

    async function init() {
        if (!AuthApi.isAuthenticated()) {
            const continueAsGuest = document.getElementById('guest-checkout');
            if (!continueAsGuest) {
                Toast.info('Please login to continue checkout.');
                window.location.href = '/account/login/?next=/checkout/';
                return;
            }
        }

        await loadCart();
        if (cartLoadFailed) return;
        
        if (!cart || !cart.items || cart.items.length === 0) {
            Toast.warning('Your cart is empty.');
            return;
        }

        await loadUserAddresses();
        initStepNavigation();
        initFormValidation();
        initOrderSummaryToggle();
        initStepForms();
    }

    async function loadCart() {
        try {
            const response = await CartApi.getCart();
            if (!response || response.success === false) {
                throw { message: response?.message || 'Failed to load cart', data: response?.data };
            }
            cart = response.data;
            renderOrderSummary();
            cartLoadFailed = false;
            cartLoadError = null;
        } catch (error) {
            console.error('Failed to load cart:', error);
            cartLoadFailed = true;
            cartLoadError = error;
            Toast.error(resolveErrorMessage(error, 'Failed to load cart.'));
        }
    }

    function resolveErrorMessage(error, fallback) {
        const generic = [
            'request failed.',
            'request failed',
            'invalid response format',
            'invalid request format'
        ];

        const isGeneric = (msg) => {
            if (!msg) return true;
            const normalized = String(msg).trim().toLowerCase();
            return generic.includes(normalized);
        };

        const pickFromErrors = (errObj) => {
            if (!errObj) return null;
            if (typeof errObj === 'string') return errObj;
            if (Array.isArray(errObj)) return errObj[0];
            if (typeof errObj === 'object') {
                const values = Object.values(errObj);
                const flattened = values.flat ? values.flat() : values.reduce((acc, val) => acc.concat(val), []);
                const first = flattened[0] ?? values[0];
                if (typeof first === 'string') return first;
                if (first && typeof first === 'object') {
                    return pickFromErrors(first);
                }
            }
            return null;
        };

        const candidates = [];
        if (error?.message) candidates.push(error.message);
        if (error?.data?.message) candidates.push(error.data.message);
        if (error?.data?.detail) candidates.push(error.data.detail);
        if (error?.data && typeof error.data === 'string') candidates.push(error.data);
        if (error?.errors) candidates.push(pickFromErrors(error.errors));
        if (error?.data && typeof error.data === 'object') candidates.push(pickFromErrors(error.data));

        const best = candidates.find(msg => msg && !isGeneric(msg));
        return best || fallback;
    }

    function getSubmitUi(form) {
        if (!form) return {};
        const button = form.querySelector('button[type="submit"]');
        const textEl = form.querySelector('#btn-text') || form.querySelector('#button-text');
        const spinnerEl = form.querySelector('#btn-spinner') || form.querySelector('#spinner');
        const arrowEl = form.querySelector('#arrow-icon');
        const defaultText = textEl ? textEl.textContent : button ? button.textContent : '';
        return { button, textEl, spinnerEl, arrowEl, defaultText };
    }

    function setSubmitLoading(ui, loading, loadingText = 'Processing...') {
        if (!ui) return;
        if (ui.button) ui.button.disabled = loading;
        if (ui.textEl) ui.textEl.textContent = loading ? loadingText : ui.defaultText;
        if (ui.spinnerEl) ui.spinnerEl.classList.toggle('hidden', !loading);
        if (ui.arrowEl) ui.arrowEl.classList.toggle('hidden', loading);
    }

    async function submitCheckoutForm(form, options = {}) {
        if (!form || form.dataset.submitting === 'true') return;
        if (cartLoadFailed) {
            Toast.error(resolveErrorMessage(cartLoadError, 'Failed to load cart.'));
            return;
        }

        const validate = options.validate;
        if (typeof validate === 'function') {
            const isValid = await validate();
            if (!isValid) return;
        }

        const ui = getSubmitUi(form);
        setSubmitLoading(ui, true, options.loadingText || 'Processing...');
        form.dataset.submitting = 'true';

        try {
            const response = await fetch(form.action || window.location.href, {
                method: (form.method || 'POST').toUpperCase(),
                body: new FormData(form),
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                },
                credentials: 'same-origin'
            });

            let data = null;
            try {
                data = await response.json();
            } catch (e) {
                data = null;
            }

            if (!response.ok || (data && data.success === false)) {
                throw { message: data?.message || response.statusText || 'Request failed.', data };
            }

            const redirectUrl = data?.redirect_url || options.redirectUrl;
            if (redirectUrl) {
                window.location.href = redirectUrl;
                return;
            }

            if (typeof options.onSuccess === 'function') {
                options.onSuccess(data);
            }
        } catch (error) {
            Toast.error(resolveErrorMessage(error, options.errorMessage || 'Request failed.'));
            if (typeof options.onError === 'function') {
                options.onError(error);
            }
        } finally {
            form.dataset.submitting = 'false';
            setSubmitLoading(ui, false, options.loadingText || 'Processing...');
        }
    }

    function renderOrderSummary() {
        const container = document.getElementById('order-summary');
        if (!container || !cart) return;

        const items = Array.isArray(cart.items) ? cart.items : [];
        const itemCount = Number(cart.item_count ?? items.length ?? 0);
        const countText = `${itemCount} item${itemCount === 1 ? '' : 's'}`;

        document.querySelectorAll('[data-order-items-count]').forEach(el => {
            el.textContent = countText;
        });

        const safeNumber = (value) => {
            const num = Number(value);
            return Number.isFinite(num) ? num : 0;
        };

        const getTaxRate = () => {
            const el = document.getElementById('tax-rate-data') || document.querySelector('[data-tax-rate]');
            if (!el) return 0;
            const raw = el.dataset?.taxRate ?? el.textContent ?? '';
            const val = parseFloat(raw);
            return Number.isFinite(val) ? val : 0;
        };

        const escape = (val) => Templates.escapeHtml(String(val ?? ''));

        const renderItems = () => {
            if (!items.length) {
                return `<p class="text-gray-500 dark:text-gray-400 text-center py-4">Your cart is empty</p>`;
            }

            return items.map((item, idx) => {
                const productName = item.product?.name || item.product_name || item.name || 'Item';
                const productImage = item.product?.image || item.product_image || item.image || null;
                const variantName = item.variant?.name || item.variant?.value || item.variant_name || '';
                const quantity = safeNumber(item.quantity || 0);
                const unitPrice = safeNumber(item.price ?? item.unit_price ?? item.unitPrice ?? item.price_at_add ?? 0);
                const lineTotal = safeNumber(item.total ?? (unitPrice * quantity));
                const borderClass = idx !== items.length - 1 ? 'border-b border-gray-100 dark:border-gray-700' : '';

                return `
                    <div class="flex items-start space-x-4 py-3 ${borderClass}">
                        <div class="relative flex-shrink-0">
                            ${productImage ? `
                                <img src="${productImage}" alt="${escape(productName)}" class="w-16 h-16 object-cover rounded-lg" loading="lazy" decoding="async" onerror="this.style.display='none'">
                            ` : `
                                <div class="w-16 h-16 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center text-gray-400">
                                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                                    </svg>
                                </div>
                            `}
                            <span class="absolute -top-2 -right-2 w-5 h-5 bg-gray-600 text-white text-xs rounded-full flex items-center justify-center font-medium">
                                ${quantity}
                            </span>
                        </div>
                        <div class="flex-1 min-w-0">
                            <p class="text-sm font-medium text-gray-900 dark:text-white truncate">${escape(productName)}</p>
                            ${variantName ? `<p class="text-xs text-gray-500 dark:text-gray-400">${escape(variantName)}</p>` : ''}
                        </div>
                        <p class="text-sm font-medium text-gray-900 dark:text-white">${Templates.formatPrice(lineTotal)}</p>
                    </div>
                `;
            }).join('');
        };

        const subtotal = safeNumber(cart.subtotal);
        const discount = safeNumber(cart.discount_amount);
        const baseTotal = cart.total !== undefined && cart.total !== null
            ? safeNumber(cart.total)
            : Math.max(0, subtotal - discount);

        const shippingInfo = resolveShippingCost(container, cart);

        const taxRate = getTaxRate();
        const taxEl = container.querySelector('#tax-amount') || document.getElementById('tax-amount');
        const existingTaxRaw = taxEl?.dataset?.price;
        const existingTax = existingTaxRaw !== undefined && existingTaxRaw !== '' ? safeNumber(existingTaxRaw) : null;
        const taxAmount = cart.tax_amount !== undefined && cart.tax_amount !== null
            ? safeNumber(cart.tax_amount)
            : (existingTax !== null ? existingTax : (taxRate > 0 ? (baseTotal * taxRate / 100) : 0));

        const giftWrapRow = container.querySelector('#gift-wrap-row') || document.getElementById('gift-wrap-row');
        const giftWrapCostEl = container.querySelector('#gift-wrap-cost') || document.getElementById('gift-wrap-cost');
        const giftWrapLabel = giftWrapRow?.querySelector('span')?.textContent?.trim() || 'Gift Wrapping';
        const giftWrapCostRaw = giftWrapCostEl?.dataset?.price ?? giftWrapCostEl?.textContent ?? container.dataset?.giftWrapCost ?? 0;
        const giftWrapAmount = safeNumber(container.dataset?.giftWrapAmount ?? 0);
        const giftWrapToggle = document.getElementById('gift_wrap');
        const giftWrapSelected = !!giftWrapToggle?.checked || (giftWrapRow && giftWrapRow.style.display !== 'none' && !giftWrapRow.classList.contains('hidden'));
        let giftWrapCost = safeNumber(giftWrapCostRaw);
        if (giftWrapSelected && giftWrapCost <= 0 && giftWrapAmount > 0) {
            giftWrapCost = giftWrapAmount;
        }
        const giftWrapVisible = giftWrapSelected || giftWrapCost > 0;

        const hasShipping = shippingInfo && shippingInfo.cost !== null && !Number.isNaN(shippingInfo.cost);
        const total = baseTotal + (hasShipping ? shippingInfo.cost : 0) + (taxAmount || 0) + (giftWrapVisible ? giftWrapCost : 0);
        const shippingDisplay = hasShipping ? formatShippingDisplay(shippingInfo) : 'Calculated next';
        const shippingCurrencyAttr = shippingInfo?.currency ? ` data-currency="${escape(shippingInfo.currency)}"` : '';
        const feeMeta = getPaymentFeeMeta();
        const paymentFee = computePaymentFee(total, feeMeta);
        if (container) {
            container.dataset.paymentFee = paymentFee;
            if (feeMeta?.name) container.dataset.paymentFeeLabel = feeMeta.name;
            container.dataset.giftWrapCost = giftWrapCost;
            container.dataset.giftWrapAmount = giftWrapAmount;
        }

        const itemsHtml = `
            <div class="space-y-4 max-h-80 overflow-y-auto scrollbar-thin pr-2">
                ${renderItems()}
            </div>
        `;

        const feeRowHtml = `
            <div id="payment-fee-row" class="flex justify-between text-sm text-gray-600 dark:text-gray-400 ${paymentFee > 0 ? '' : 'hidden'}">
                <span id="payment-fee-label">Extra payment fee${feeMeta?.name ? ` (${escape(feeMeta.name)})` : ''}</span>
                <span id="payment-fee-amount">${Templates.formatPrice(paymentFee)}</span>
            </div>
        `;
        const giftWrapRowHtml = `
            <div id="gift-wrap-row" class="flex justify-between text-sm text-gray-600 dark:text-gray-400" style="display: ${giftWrapVisible ? 'flex' : 'none'};">
                <span>${escape(giftWrapLabel)}</span>
                <span id="gift-wrap-cost" data-price="${giftWrapCost}">+${Templates.formatPrice(giftWrapCost)}</span>
            </div>
        `;

        const totalsHtml = `
            <div class="space-y-3 border-t border-gray-200 dark:border-gray-700 mt-4 pt-4">
                <div class="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                    <span>Subtotal</span>
                    <span id="subtotal" data-price="${subtotal}">${Templates.formatPrice(subtotal)}</span>
                </div>
                <div id="discount-row" class="flex justify-between text-sm text-green-600 ${discount > 0 ? '' : 'hidden'}">
                    <span>Discount</span>
                    <span id="discount-amount" data-price="${discount}">-${Templates.formatPrice(discount)}</span>
                    <span id="discount" class="hidden" data-price="${discount}">-${Templates.formatPrice(discount)}</span>
                </div>
                <div class="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                    <span>Shipping</span>
                    <span id="shipping-cost" data-price="${hasShipping ? shippingInfo.cost : ''}"${shippingCurrencyAttr} class="font-medium text-gray-900 dark:text-white">
                        ${hasShipping ? escape(shippingDisplay) : 'Calculated next'}
                    </span>
                </div>
                ${giftWrapRowHtml}
                ${feeRowHtml}
                <div class="flex justify-between text-sm text-gray-600 dark:text-gray-400 ${taxRate > 0 || taxAmount > 0 ? '' : 'hidden'}">
                    <span>Tax${taxRate > 0 ? ` (${taxRate}%)` : ''}</span>
                    <span id="tax-amount" data-price="${taxAmount}">${Templates.formatPrice(taxAmount)}</span>
                </div>
                <div class="flex justify-between text-lg font-bold text-gray-900 dark:text-white border-t border-gray-200 dark:border-gray-700 pt-3">
                    <span>Total</span>
                    <span id="order-total" data-price="${total}">${Templates.formatPrice(total)}</span>
                </div>
            </div>
        `;

        const itemsTarget = container.querySelector('[data-order-items]');
        const totalsTarget = container.querySelector('[data-order-totals]');

        if (itemsTarget || totalsTarget) {
            if (itemsTarget) itemsTarget.innerHTML = itemsHtml;
            if (totalsTarget) totalsTarget.innerHTML = totalsHtml;
            updatePaymentFeeRow(total);
            return;
        }

        container.innerHTML = itemsHtml + totalsHtml;
        updatePaymentFeeRow(total);
    }

    async function loadUserAddresses() {
        if (!AuthApi.isAuthenticated()) return;

        try {
            const response = await AuthApi.getAddresses();
            const addresses = response.data || [];

            const shippingContainer = document.getElementById('saved-addresses');
            const shouldRender = shippingContainer && (shippingContainer.dataset.jsRender === 'true' || shippingContainer.children.length === 0);
            if (shippingContainer && addresses.length > 0) {
                if (shouldRender) {
                    shippingContainer.innerHTML = `
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Saved Addresses</label>
                        <div class="space-y-2">
                            ${addresses.map(addr => `
                                <label class="flex items-start p-3 border border-gray-200 rounded-lg cursor-pointer hover:border-primary-500 transition-colors">
                                    <input type="radio" name="saved_address" value="${addr.id}" class="mt-1 text-primary-600 focus:ring-primary-500">
                                    <div class="ml-3">
                                        <p class="font-medium text-gray-900">${Templates.escapeHtml(addr.full_name || `${addr.first_name} ${addr.last_name}`)}</p>
                                        <p class="text-sm text-gray-600">${Templates.escapeHtml(addr.address_line_1)}</p>
                                        ${addr.address_line_2 ? `<p class="text-sm text-gray-600">${Templates.escapeHtml(addr.address_line_2)}</p>` : ''}
                                        <p class="text-sm text-gray-600">${Templates.escapeHtml(addr.city)}, ${Templates.escapeHtml(addr.state || '')} ${Templates.escapeHtml(addr.postal_code)}</p>
                                        <p class="text-sm text-gray-600">${Templates.escapeHtml(addr.country)}</p>
                                        ${addr.is_default ? '<span class="inline-block mt-1 px-2 py-0.5 bg-primary-100 text-primary-700 text-xs rounded">Default</span>' : ''}
                                    </div>
                                </label>
                            `).join('')}
                            <label class="flex items-center p-3 border border-gray-200 rounded-lg cursor-pointer hover:border-primary-500 transition-colors">
                                <input type="radio" name="saved_address" value="new" class="text-primary-600 focus:ring-primary-500" checked>
                                <span class="ml-3 text-gray-700">Enter a new address</span>
                            </label>
                        </div>
                    </div>
                `;
                }

                bindAddressSelection();
            }
        } catch (error) {
            console.error('Failed to load addresses:', error);
        }
    }

    function bindAddressSelection() {
        const addressRadios = document.querySelectorAll('input[name="saved_address"]');
        const newAddressForm = document.getElementById('new-address-form');

        addressRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.value === 'new') {
                    newAddressForm?.classList.remove('hidden');
                } else {
                    newAddressForm?.classList.add('hidden');
                    checkoutData.shipping_address = e.target.value;
                }
            });
        });
    }

    function initStepNavigation() {
        const steps = document.querySelectorAll('[data-step]');
        const stepIndicators = document.querySelectorAll('[data-step-indicator]');
        const nextBtns = document.querySelectorAll('[data-next-step]');
        const prevBtns = document.querySelectorAll('[data-prev-step]');

        function showStep(stepNumber) {
            steps.forEach(step => {
                step.classList.toggle('hidden', parseInt(step.dataset.step) !== stepNumber);
            });

            stepIndicators.forEach(indicator => {
                const indicatorStep = parseInt(indicator.dataset.stepIndicator);
                indicator.classList.toggle('bg-primary-600', indicatorStep <= stepNumber);
                indicator.classList.toggle('text-white', indicatorStep <= stepNumber);
                indicator.classList.toggle('bg-gray-200', indicatorStep > stepNumber);
                indicator.classList.toggle('text-gray-600', indicatorStep > stepNumber);
            });

            currentStep = stepNumber;
        }

        nextBtns.forEach(btn => {
            btn.addEventListener('click', async () => {
                const valid = await validateCurrentStep();
                if (valid) {
                    if (currentStep === 1) {
                        await loadShippingMethods();
                    }
                    showStep(currentStep + 1);
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                }
            });
        });

        prevBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                showStep(currentStep - 1);
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
        });

        showStep(1);
    }

    async function validateCurrentStep() {
        switch (currentStep) {
            case 1:
                return validateShippingAddress();
            case 2:
                return validateShippingMethod();
            case 3:
                return validatePaymentMethod();
            default:
                return true;
        }
    }

    // Inline error helpers
    function clearStepErrors(container) {
        if (!container) return;
        container.querySelectorAll('[data-error-for]').forEach(el => el.remove());
        // Use CSS attribute selector to find elements with the !border-red-500 class
        container.querySelectorAll('[class*="!border-red-500"]').forEach(el => el.classList.remove('!border-red-500'));
    }

    function showInlineError(fieldEl, message) {
        if (!fieldEl) return;
        // Remove existing error for this field
        const name = fieldEl.getAttribute('name') || fieldEl.id || Math.random().toString(36).slice(2, 8);
        const existing = fieldEl.closest('form')?.querySelector(`[data-error-for="${name}"]`);
        if (existing) existing.remove();

        const msg = document.createElement('p');
        msg.className = 'text-sm text-red-600 mt-1';
        msg.setAttribute('data-error-for', name);
        msg.textContent = message;

        // Add visual state to the field
        fieldEl.classList.add('!border-red-500');

        // Insert after field or its parent wrapper
        if (fieldEl.nextSibling) {
            fieldEl.parentNode.insertBefore(msg, fieldEl.nextSibling);
        } else {
            fieldEl.parentNode.appendChild(msg);
        }
    }

    function focusFirstInvalid(container) {
        if (!container) return;
        const first = container.querySelector('[data-error-for]');
        if (!first) return;
        // Try to focus associated input without scrolling the page
        const name = first.getAttribute('data-error-for');
        const input = container.querySelector(`[name="${name}"]`) || container.querySelector(`#${name}`) || first.previousElementSibling;
        if (input && typeof input.focus === 'function') {
            try {
                input.focus({ preventScroll: true });
            } catch (e) {
                // Old browsers may not support options — fallback to normal focus
                input.focus();
            }
        }
    }

    function validateShippingAddress() {
        // Support both 'information-form' (wizard first step) and dedicated 'shipping-address-form'
        const savedAddressRadio = document.querySelector('input[name="saved_address"]:checked');
        
        if (savedAddressRadio && savedAddressRadio.value !== 'new') {
            clearStepErrors(document.getElementById('new-address-form') || document.getElementById('information-form'));
            checkoutData.shipping_address = savedAddressRadio.value;
            return true;
        }

        const form = document.getElementById('shipping-address-form') || document.getElementById('information-form') || document.getElementById('new-address-form');
        if (!form) return false;

        clearStepErrors(form);

        const formData = new FormData(form);
        const address = {
            first_name: formData.get('first_name') || formData.get('full_name')?.split(' ')?.[0],
            last_name: formData.get('last_name') || (formData.get('full_name') ? formData.get('full_name').split(' ').slice(1).join(' ') : ''),
            email: formData.get('email'),
            phone: formData.get('phone'),
            address_line_1: formData.get('address_line1') || formData.get('address_line_1'),
            address_line_2: formData.get('address_line2') || formData.get('address_line_2'),
            city: formData.get('city'),
            state: formData.get('state'),
            postal_code: formData.get('postal_code'),
            country: formData.get('country')
        };

        // Required fields depending on context
        const required = ['email', 'first_name', 'address_line_1', 'city', 'postal_code'];
        const missing = required.filter(field => !address[field]);

        if (missing.length > 0) {
            // Show inline errors and focus first
            missing.forEach(field => {
                // Map back to input names in form
                let selector = `[name="${field}"]`;
                if (field === 'address_line_1') selector = `[name="address_line1"],[name="address_line_1"]`;
                const fieldEl = form.querySelector(selector);
                showInlineError(fieldEl || form, field.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) + ' is required.');
            });

            focusFirstInvalid(form);
            return false;
        }

        // Basic email format check
        if (address.email && !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(address.email)) {
            const emailEl = form.querySelector('[name="email"]');
            showInlineError(emailEl || form, 'Please enter a valid email address.');
            focusFirstInvalid(form);
            return false;
        }

        checkoutData.shipping_address = address;
        return true;
    }

    async function loadShippingMethods() {
        const container = document.getElementById('shipping-methods');
        if (!container) return;

        Loader.show(container, 'spinner');

        try {
            const address = checkoutData.shipping_address;

            if (!address) {
                // No address available yet (user hasn't filled or selected one)
                container.innerHTML = '<p class="text-gray-500">Please provide a shipping address to view shipping methods.</p>';
                return;
            }

            const params = typeof address === 'object' ? {
                country: address.country,
                postal_code: address.postal_code,
                city: address.city
            } : { address_id: address };

            const response = await ShippingApi.getRates(params);
            const methods = response.data || [];

            if (methods.length === 0) {
                container.innerHTML = '<p class="text-gray-500">No shipping methods available for your location.</p>';
                return;
            }

            container.innerHTML = `
                <div class="space-y-3">
                    ${methods.map((method, index) => `
                        <label class="flex items-center justify-between p-4 border border-gray-200 rounded-lg cursor-pointer hover:border-primary-500 transition-colors">
                            <div class="flex items-center">
                                <input 
                                    type="radio" 
                                    name="shipping_method" 
                                    value="${method.id}" 
                                    ${index === 0 ? 'checked' : ''}
                                    class="text-primary-600 focus:ring-primary-500"
                                    data-price="${method.price}"
                                >
                                <div class="ml-3">
                                    <p class="font-medium text-gray-900">${Templates.escapeHtml(method.name)}</p>
                                    ${method.description ? `<p class="text-sm text-gray-500">${Templates.escapeHtml(method.description)}</p>` : ''}
                                    ${method.estimated_days ? `<p class="text-sm text-gray-500">Delivery in ${method.estimated_days} days</p>` : ''}
                                </div>
                            </div>
                            <span class="font-semibold text-gray-900">${method.price > 0 ? Templates.formatPrice(method.price) : 'Free'}</span>
                        </label>
                    `).join('')}
                </div>
            `;

            const shippingRadios = container.querySelectorAll('input[name="shipping_method"]');
            shippingRadios.forEach((radio, idx) => {
                const method = methods[idx] || {};
                const price = Number(method.price ?? method.rate ?? 0) || 0;
                const display = method.price_display || method.rate_display || (price > 0 ? Templates.formatPrice(price) : 'Free');
                radio.__price = price;
                radio.dataset.display = display;
                if (method.currency && method.currency.code) {
                    radio.dataset.currency = method.currency.code;
                }
                radio.addEventListener('change', () => {
                    updateShippingCost(parseFloat(radio.__price) || 0, {
                        rateId: radio.value,
                        display: radio.dataset.display,
                        currency: radio.dataset.currency || null,
                        persist: true,
                        type: 'delivery'
                    });
                });
            });

            if (methods.length > 0) {
                checkoutData.shipping_method = methods[0].id;
                const first = methods[0] || {};
                const price = Number(first.price ?? first.rate ?? 0) || 0;
                const display = first.price_display || first.rate_display || (price > 0 ? Templates.formatPrice(price) : 'Free');
                updateShippingCost(price, {
                    rateId: shippingRadios[0]?.value || first.id,
                    display,
                    currency: shippingRadios[0]?.dataset?.currency || null,
                    persist: true,
                    type: 'delivery'
                });
            }
        } 
        catch (error) {
            console.error('Failed to load shipping methods:', error);
            container.innerHTML = '<p class="text-red-500">Failed to load shipping methods. Please try again.</p>';
        }
    }

    // -----------------------------
    // Payment gateways (dynamic)
    // -----------------------------
    function updatePaymentContinueState(hasGateways) {
        const submitButton = document.getElementById('submit-button');
        const buttonText = document.getElementById('button-text');
        if (!submitButton || !buttonText) return;

        submitButton.disabled = !hasGateways;
        buttonText.textContent = hasGateways ? 'Continue to Review' : 'No payment methods available';
    }

    async function fetchAndRenderPaymentGateways() {
        const container = document.getElementById('payment-methods-container');
        if (!container) return;

        try {
            const params = new URLSearchParams();
            // Client-side currency parameter removed; server handles currency formatting.
            if (window.CONFIG && CONFIG.shippingData && CONFIG.shippingData.countryCode) params.set('country', CONFIG.shippingData.countryCode);
            if (cart && (cart.total || cart.total === 0)) params.set('amount', cart.total);

            const resp = await fetch(`/api/v1/payments/gateways/available/?${params.toString()}`, {
                credentials: 'same-origin'
            });
            const data = await resp.json();

            const gateways = (data && data.data) || [];

            // If server already rendered gateways (non-empty) try to avoid re-render to prevent UI flicker.
            const existing = container.querySelectorAll('.payment-option');
            if (existing && existing.length > 0) {
                try {
                    const existingCodes = Array.from(existing).map(el => el.dataset.gateway).filter(Boolean);
                    const remoteCodes = (gateways || []).map(g => g.code);

                    // If codes match exactly in order and length, keep server markup and just re-bind handlers.
                const same = existingCodes.length === remoteCodes.length && existingCodes.every((c, i) => c === remoteCodes[i]);
                if (same) {
                    initFormValidation(); // ensure event handlers and initial visibility are in place
                    updatePaymentContinueState(existingCodes.length > 0);
                    return;
                }
            } catch (err) {
                // Fall back to previous behavior if anything goes wrong
                console.warn('Failed to compare existing payment gateways:', err);
            }

            // If remote returned nothing but server has content, keep server content (avoid replacing with empty state)
            if (gateways.length === 0) {
                updatePaymentContinueState(existing.length > 0);
                return;
            }
        }

        // If no gateways, show informative block (template also handles this case)
        if (!gateways || gateways.length === 0) {
            container.innerHTML = `
                    <div class="text-center py-8 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl">
                        <svg class="w-16 h-16 mx-auto mb-4 text-gray-300 dark:text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"></path></svg>
                        <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">No payment methods are configured</h3>
                        <p class="text-gray-500 dark:text-gray-400 mb-2">We don't have any payment providers configured for your currency or location. Please contact support to enable online payments.</p>
                        <p class="text-sm text-gray-400">You can still place an order if Cash on Delivery or Bank Transfer is available from admin.</p>
                  </div>
              `;
            updatePaymentContinueState(false);
            return;
        }

            // Build radio options into a fragment, add animations and staggered delays to improve perceived performance
            const frag = document.createDocumentFragment();
            gateways.forEach((g, idx) => {
                const wrapper = document.createElement('div');
                wrapper.className = 'relative payment-option transform transition-all duration-300 hover:scale-[1.01]';
                wrapper.dataset.gateway = g.code;

                // Apply slideIn animation with staggered delay so new DOM shows with animation
                wrapper.style.animation = 'slideIn 0.3s ease-out both';
                wrapper.style.animationDelay = `${idx * 80}ms`;

                const input = document.createElement('input');
                input.type = 'radio';
                input.name = 'payment_method';
                input.value = g.code;
                input.id = `payment-${g.code}`;
                input.className = 'peer sr-only';
                input.dataset.feeType = g.fee_type || 'none';
                input.dataset.feeAmount = (g.fee_amount_converted ?? g.fee_amount ?? 0);
                input.dataset.feePercent = g.fee_amount ?? 0;
                input.dataset.feeName = g.name || '';
                if (idx === 0) input.checked = true;

                const label = document.createElement('label');
                label.setAttribute('for', input.id);
                label.className = 'flex items-center justify-between p-4 border-2 rounded-xl cursor-pointer transition-all duration-300 hover:border-gray-400 border-gray-200';

                label.innerHTML = `
                    <div class="flex items-center">
                        <div class="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mr-4">
                            ${g.icon_url ? `<img src="${g.icon_url}" class="h-6" alt="${g.name}">` : `<span class="font-bold">${g.code.toUpperCase()}</span>`}
                        </div>
                        <div>
                            <p class="font-medium text-gray-900 dark:text-white">${Templates.escapeHtml(g.name)}</p>
                            <p class="text-sm text-gray-500 dark:text-gray-400">${Templates.escapeHtml(g.description || '')}</p>
                            ${g.fee_text ? `<p class="text-xs text-amber-600 dark:text-amber-400 mt-1">${Templates.escapeHtml(g.fee_text)}</p>` : ''}
                            ${g.instructions ? `<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">${g.instructions}</p>` : ''}
                        </div>
                    </div>
                `;

                wrapper.appendChild(input);
                wrapper.appendChild(label);

                // Append selection indicator
                const indicator = document.createElement('div');
                indicator.className = 'absolute top-4 right-4 opacity-0 peer-checked:opacity-100 transition-opacity duration-300';
                indicator.innerHTML = `<svg class="w-6 h-6 text-primary-600" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"></path></svg>`;
                wrapper.appendChild(indicator);

                frag.appendChild(wrapper);

                // If gateway requires client side (Stripe), load JS and init (do this early so elements can initialize)
                if (g.public_key && g.requires_client) {
                    loadStripeJsIfNeeded(g.public_key).catch(err => console.error('Failed to load Stripe:', err));
                }
            });

        // Atomically replace existing nodes to reduce flicker
        container.replaceChildren(frag);

        // Re-bind form handlers
        initFormValidation();
        updatePaymentContinueState(gateways.length > 0);
        updatePaymentFeeRow();

    } catch (error) {
        console.error('Failed to load payment gateways:', error);
    }
    }

    function loadStripeJsIfNeeded(publishableKey) {
        return new Promise((resolve, reject) => {
            if (window.Stripe && window.STRIPE_PUBLISHABLE_KEY === publishableKey) {
                // Already configured
                resolve();
                return;
            }

            // Set global publishable key
            window.STRIPE_PUBLISHABLE_KEY = publishableKey;

            // If Stripe script is already loaded, just initialize elements
            if (window.Stripe) {
                initStripeElements(publishableKey);
                resolve();
                return;
            }

            const script = document.createElement('script');
            script.src = 'https://js.stripe.com/v3/';
            script.async = true;
            script.onload = () => {
                try {
                    initStripeElements(publishableKey);
                    resolve();
                } catch (err) {
                    reject(err);
                }
            };
            script.onerror = (e) => reject(e);
            document.head.appendChild(script);
        });
    }

    function initStripeElements(publishableKey) {
        if (typeof Stripe === 'undefined') throw new Error('Stripe script not loaded');

        try {
            const stripe = Stripe(publishableKey);
            const elements = stripe.elements();
            const cardEl = elements.create('card');
            const mountPoint = document.getElementById('card-element');
            if (mountPoint) {
                // Clean previous mount if any
                mountPoint.innerHTML = '';
                cardEl.mount('#card-element');

                // Show realtime validation errors
                cardEl.on('change', (e) => {
                    const errContainer = document.getElementById('card-errors');
                    if (errContainer) errContainer.textContent = e.error ? e.error.message : '';
                });

                // Expose card element for confirmCardPayment
                window.stripeInstance = stripe;
                window.stripeCard = cardEl;
            }
        } catch (err) {
            console.error('Error initializing Stripe elements:', err);
            throw err;
        }
    }

    // Call dynamic loading on init
    (function autoInitGateways() {
        // Delay slightly to let server-rendered DOM settle
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(() => {
                fetchAndRenderPaymentGateways();
            }, 50);
        });
    })();



    function updateShippingCost(cost, meta = {}) {
        const shippingCostEl = document.getElementById('shipping-cost');
        const orderTotalEl = document.getElementById('order-total');
        const numericCost = toNumber(cost, 0);
        const display = meta.display || (numericCost > 0 ? Templates.formatPrice(numericCost, meta.currency || null) : 'Free');

        if (shippingCostEl) {
            shippingCostEl.textContent = display;
            shippingCostEl.dataset.price = String(numericCost);
            if (meta.currency) shippingCostEl.dataset.currency = meta.currency;
        }

        if (meta.persist) {
            try {
                localStorage.setItem(SHIPPING_STORAGE_KEY, JSON.stringify({
                    type: meta.type || 'delivery',
                    rateId: meta.rateId || null,
                    cost: numericCost,
                    display: display || '',
                    currency: meta.currency || null,
                    ts: Date.now()
                }));
            } catch (err) {
                // ignore storage failures
            }
        }

        if (orderTotalEl && cart) {
            const total = (cart.total || 0) + numericCost;
            orderTotalEl.textContent = Templates.formatPrice(total);
            orderTotalEl.dataset.price = total;
        }
    }

    function validateShippingMethod() {
        const selected = document.querySelector('input[name="shipping_method"]:checked');
        if (!selected) {
            Toast.error('Please select a shipping method.');
            return false;
        }
        checkoutData.shipping_method = selected.value;
        return true;
    }

    function initOrderSummaryToggle() {
        const toggle = document.getElementById('order-summary-toggle');
        const block = document.getElementById('order-summary-block');
        if (!toggle || !block) return;

        toggle.addEventListener('click', () => {
            const hidden = block.classList.toggle('hidden');
            toggle.setAttribute('aria-expanded', (!hidden).toString());
            const chevron = toggle.querySelector('svg');
            if (chevron) chevron.classList.toggle('rotate-180', !hidden);
        });

        // If the block is hidden on mobile by responsive classes, ensure aria is accurate
        const isHidden = window.getComputedStyle(block).display === 'none' || block.classList.contains('hidden');
        toggle.setAttribute('aria-expanded', (!isHidden).toString());
    }

    function validatePaymentMethod() {
        const selected = document.querySelector('input[name="payment_method"]:checked');
        const form = document.getElementById('payment-form');
        clearStepErrors(form);

        if (!selected) {
            // Show inline message in payment block
            const container = document.getElementById('payment-methods-container') || form;
            showInlineError(container, 'Please select a payment method.');
            focusFirstInvalid(container);
            return false;
        }

        const code = selected.value;

        // Payment-specific checks
        if (code === 'stripe') {
            const nameEl = document.getElementById('cardholder-name');
            if (!nameEl || !nameEl.value.trim()) {
                showInlineError(nameEl || form, 'Cardholder name is required.');
                focusFirstInvalid(form);
                return false;
            }
            // Ensure Stripe element exists (mounting handled elsewhere)
            if (!window.stripeCard) {
                showInlineError(document.getElementById('card-element') || form, 'Card input not ready. Please wait and try again.');
                return false;
            }
        }

        if (code === 'bkash') {
            const el = document.getElementById('bkash-number');
            if (!el || !el.value.trim()) {
                showInlineError(el || form, 'bKash mobile number is required.');
                focusFirstInvalid(form);
                return false;
            }
        }

        if (code === 'nagad') {
            const el = document.getElementById('nagad-number');
            if (!el || !el.value.trim()) {
                showInlineError(el || form, 'Nagad mobile number is required.');
                focusFirstInvalid(form);
                return false;
            }
        }

        // Clear any errors and proceed
        checkoutData.payment_method = code;
        return true;
    }

    function initFormValidation() {
        const sameAsBillingCheckbox = document.getElementById('same-as-shipping');
        const billingAddressForm = document.getElementById('billing-address-form');

        sameAsBillingCheckbox?.addEventListener('change', (e) => {
            checkoutData.same_as_shipping = e.target.checked;
            billingAddressForm?.classList.toggle('hidden', e.target.checked);
        });

        const paymentMethods = document.querySelectorAll('input[name="payment_method"]');
        paymentMethods.forEach(method => {
            const handler = (e) => {
                // Hide all payment form blocks marked with data-payment-form
                document.querySelectorAll('[data-payment-form]').forEach(form => {
                    form.classList.add('hidden');
                });

                // Show the matching form using data attribute; fallback to id `${code}-form`
                const code = e.target ? e.target.value : (method.value || null);
                if (!code) return;

                let targetForm = document.querySelector(`[data-payment-form="${code}"]`);
                if (!targetForm) {
                    targetForm = document.getElementById(`${code}-form`);
                }

                  targetForm?.classList.remove('hidden');
                  updatePaymentFeeRow();
              };

            method.addEventListener('change', handler);

            // If this method is pre-selected on page load, trigger handler to set initial visibility
            if (method.checked) {
                handler({ target: method });
            }
        });

        // Only attach the SPA place order handler if NOT on the traditional form-based review page
        // The traditional form has action attribute and we should let it submit normally
        const placeOrderBtn = document.getElementById('place-order-btn');
        const placeOrderForm = document.getElementById('place-order-form');
        
        // Only use JavaScript-based checkout if the form doesn't have a traditional action
        if (placeOrderBtn && (!placeOrderForm || !placeOrderForm.action || placeOrderForm.action.includes('javascript'))) {
            placeOrderBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                await placeOrder();
            });
        }
    }

    function initStepForms() {
        const infoForm = document.getElementById('information-form');
        if (infoForm && infoForm.dataset.ajaxBound !== 'true') {
            infoForm.dataset.ajaxBound = 'true';
            infoForm.addEventListener('submit', (e) => {
                e.preventDefault();
                e.stopImmediatePropagation();
                submitCheckoutForm(infoForm, {
                    validate: validateShippingAddress,
                    redirectUrl: '/checkout/shipping/',
                    loadingText: 'Processing...'
                });
            }, true);
        }

        const shippingForm = document.getElementById('shipping-form');
        if (shippingForm && shippingForm.dataset.ajaxBound !== 'true') {
            shippingForm.dataset.ajaxBound = 'true';
            shippingForm.addEventListener('submit', (e) => {
                e.preventDefault();
                e.stopImmediatePropagation();
                const shippingType = document.getElementById('shipping-type')?.value || 'delivery';
                if (shippingType === 'pickup') {
                    const selectedPickup = document.querySelector('input[name="pickup_location"]:checked');
                    if (!selectedPickup) {
                        Toast.error('Please select a pickup location.');
                        return;
                    }
                } else {
                    const selectedRate = document.querySelector('input[name="shipping_rate_id"]:checked') ||
                        document.querySelector('input[name="shipping_method"]:checked');
                    if (!selectedRate) {
                        Toast.error('Please select a shipping method.');
                        return;
                    }
                }

                submitCheckoutForm(shippingForm, {
                    redirectUrl: '/checkout/payment/',
                    loadingText: 'Processing...'
                });
            }, true);
        }
    }

    async function placeOrder() {
        if (!validatePaymentMethod()) return;

        const placeOrderBtn = document.getElementById('place-order-btn');
        placeOrderBtn.disabled = true;
        placeOrderBtn.innerHTML = '<svg class="animate-spin h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';

        try {
            const orderNotes = document.getElementById('order-notes')?.value;
            checkoutData.notes = orderNotes || '';

            if (!checkoutData.same_as_shipping) {
                const billingForm = document.getElementById('billing-address-form');
                if (billingForm) {
                    const formData = new FormData(billingForm);
                    checkoutData.billing_address = {
                        first_name: formData.get('billing_first_name'),
                        last_name: formData.get('billing_last_name'),
                        address_line_1: formData.get('billing_address_line_1'),
                        address_line_2: formData.get('billing_address_line_2'),
                        city: formData.get('billing_city'),
                        state: formData.get('billing_state'),
                        postal_code: formData.get('billing_postal_code'),
                        country: formData.get('billing_country')
                    };
                }
            }

            const response = await CheckoutApi.createOrder(checkoutData);
            const order = response.data;

            if (checkoutData.payment_method === 'stripe' || checkoutData.payment_method === 'card') {
                await processStripePayment(order);
            } else if (checkoutData.payment_method === 'paypal') {
                await processPayPalPayment(order);
            } else {
                window.location.href = `/orders/${order.id}/confirmation/`;
            }
        } catch (error) {
            console.error('Failed to place order:', error);
            Toast.error(error.message || 'Failed to place order. Please try again.');
            
            placeOrderBtn.disabled = false;
            placeOrderBtn.textContent = 'Place Order';
        }
    }

    async function processStripePayment(order) {
        try {
            const response = await CheckoutApi.createPaymentIntent(order.id);
            const { client_secret } = response.data;
            const publishable_key = response.data.publishable_key || window.STRIPE_PUBLISHABLE_KEY || (window.stripeInstance ? window.STRIPE_PUBLISHABLE_KEY : null);

            if (typeof Stripe === 'undefined' && !window.stripeInstance) {
                throw new Error('Stripe is not loaded.');
            }

            const stripe = window.stripeInstance || Stripe(publishable_key);
            const result = await stripe.confirmCardPayment(client_secret, {
                payment_method: {
                    card: window.stripeCard,
                    billing_details: {
                        name: `${checkoutData.shipping_address.first_name} ${checkoutData.shipping_address.last_name}`
                    }
                }
            });

            if (result.error) {
                throw new Error(result.error.message);
            }

            window.location.href = `/orders/${order.id}/confirmation/`;
        } catch (error) {
            console.error('Stripe payment failed:', error);
            Toast.error(error.message || 'Payment failed. Please try again.');
            
            const placeOrderBtn = document.getElementById('place-order-btn');
            placeOrderBtn.disabled = false;
            placeOrderBtn.textContent = 'Place Order';
        }
    }

    async function processPayPalPayment(order) {
        try {
            const response = await CheckoutApi.createPayPalOrder(order.id);
            const { approval_url } = response.data;
            window.location.href = approval_url;
        } catch (error) {
            console.error('PayPal payment failed:', error);
            Toast.error(error.message || 'Payment failed. Please try again.');
            
            const placeOrderBtn = document.getElementById('place-order-btn');
            placeOrderBtn.disabled = false;
            placeOrderBtn.textContent = 'Place Order';
        }
    }

    function destroy() {
        cart = null;
        checkoutData = {
            shipping_address: null,
            billing_address: null,
            same_as_shipping: true,
            shipping_method: null,
            payment_method: null,
            notes: ''
        };
        currentStep = 1;
    }

    return {
        init,
        destroy
    };
}());

window.CheckoutPage = CheckoutPage;
export default CheckoutPage;
