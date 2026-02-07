import { formatCurrency } from '../utils/currency.js';

function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return '';
}

function getCsrfToken() {
    // 1. Try cookie
    let token = getCookie('csrftoken') || getCookie('csrf_token') || getCookie('csrf');
    if (token) token = String(token).trim().replace(/^"|"$/g, '');

    // 2. Try meta tags used by some apps
    if (!token) {
        const meta = document.querySelector('meta[name="csrf-token"]') || document.querySelector('meta[name="csrfmiddlewaretoken"]');
        if (meta && meta.content) token = String(meta.content).trim().replace(/^"|"$/g, '');
    }

    // 3. Try hidden input (forms that render csrf tokens)
    if (!token) {
        const input = document.querySelector('input[name="csrfmiddlewaretoken"]');
        if (input && input.value) token = String(input.value).trim().replace(/^"|"$/g, '');
    }

    // 4. Basic sanity check
    if (token) {
        if (token.length < 8 || token.length > 1024) {
            // warning removed
            return null;
        }
        return token;
    }

    return null;
}

export async function initCurrencySelector(selector) {

    const resolveRoots = () => {
        if (typeof selector === 'string') return Array.from(document.querySelectorAll(selector));
        if (NodeList.prototype.isPrototypeOf(selector) || Array.isArray(selector)) return Array.from(selector);
        return [selector];
    };

    let roots = resolveRoots().filter(Boolean);

    // If no roots found and DOM not ready, wait and retry once
    if (!roots.length) {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                roots = resolveRoots().filter(Boolean);
                if (!roots.length) {
                    return;
                }
                roots.forEach(r => maybeInitRoot(r));
            }, { once: true });
            return;
        }
        return;
    }

    roots.forEach(r => maybeInitRoot(r));

    function maybeInitRoot(root) {
        if (!root) return;
        if (root.dataset && root.dataset.currencySelectorInit === 'true') {
            return;
        }
        initForRoot(root);
        if (root.dataset) root.dataset.currencySelectorInit = 'true';
    }

    function initForRoot(root) {

        const toggle = root.querySelector('#currency-selector-toggle');
        const dropdown = root.querySelector('#currency-dropdown');
        const list = root.querySelector('#currency-list');
        const currentEl = root.querySelector('#currency-current');

        // Guard to prevent concurrent setCurrency calls and double-click issues
        let isSetting = false;
        function setLoadingState(state) {
            isSetting = !!state;
            if (state) {
                toggle.setAttribute('disabled', 'true');
                toggle.classList.add('opacity-60', 'cursor-not-allowed');
                toggle.setAttribute('aria-busy', 'true');
                // add spinner if not present
                if (!toggle.querySelector('.currency-spinner')) {
                    const spinner = document.createElement('span');
                    spinner.className = 'currency-spinner inline-block ml-2 h-4 w-4 animate-spin border-2 border-current rounded-full border-t-transparent';
                    spinner.setAttribute('aria-hidden', 'true');
                    toggle.appendChild(spinner);
                }
            } else {
                toggle.removeAttribute('disabled');
                toggle.classList.remove('opacity-60', 'cursor-not-allowed');
                toggle.removeAttribute('aria-busy');
                const spinner = toggle.querySelector('.currency-spinner');
                if (spinner) spinner.remove();
            }
        }
        let currentCode = window.BUNORAA_CURRENCY && window.BUNORAA_CURRENCY.code ? window.BUNORAA_CURRENCY.code : (currentEl ? currentEl.textContent.trim() : null);
        // currencies cache for this root
        let currencies = [];

        function shouldAutoReload() {
            return !document.getElementById('djDebug');
        }

        async function fetchCurrencies(retries = 2, attempt = 0) {
            // show inline loading indicator
            list.innerHTML = '';
            const spinnerDiv = document.createElement('div');
            spinnerDiv.className = 'px-3 py-2 text-sm text-stone-500 flex items-center';
            spinnerDiv.innerHTML = '<span class="inline-block h-4 w-4 animate-spin border-2 border-current rounded-full border-t-transparent mr-2" aria-hidden="true"></span><span>Loading currencies…</span>';
            list.appendChild(spinnerDiv);

            try {
                const resp = await fetch('/api/v1/i18n/currencies/', { credentials: 'same-origin' });
                if (!resp.ok) {
                    const txt = await resp.text().catch(() => String(resp.status));
                    throw new Error('HTTP ' + resp.status + ': ' + txt);
                }
                const body = await resp.json();

                // Support different API shapes: array, { results: [] }, { data: [...] }, or { success: true, data: [...] }
                if (Array.isArray(body)) {
                    currencies = body;
                } else if (body && Array.isArray(body.results)) {
                    currencies = body.results;
                } else if (body && Array.isArray(body.data)) {
                    currencies = body.data;
                } else if (body && body.success && Array.isArray(body.data)) {
                    currencies = body.data;
                } else {
                    // Last-resort: try nested shapes
                    try {
                        const maybe = (body && body.data && body.data.results) ? body.data.results : [];
                        currencies = Array.isArray(maybe) ? maybe : [];
                    } catch (err) {
                        currencies = [];
                    }
                }

                if (!currencies || currencies.length === 0) throw new Error('No currencies in response');

                // cache the result for offline/failure fallback
                try { localStorage.setItem('currencies_cache', JSON.stringify({ data: currencies, ts: Date.now() })); } catch (e) {}

                renderList();
                return;
            } catch (err) {
                // retry with backoff
                if (attempt < retries) {
                    const delay = (attempt + 1) * 1000;
                    setTimeout(() => fetchCurrencies(retries, attempt + 1), delay);
                    return;
                }

                // final failure: try cached list
                let cachedLoaded = false;
                try {
                    const raw = localStorage.getItem('currencies_cache');
                    if (raw) {
                        const parsed = JSON.parse(raw);
                        if (parsed && Array.isArray(parsed.data) && parsed.data.length) {
                            currencies = parsed.data;
                            cachedLoaded = true;
                        }
                    }
                } catch (e) {}

                list.innerHTML = '';
                const el = document.createElement('div');
                el.className = 'px-3 py-2 text-sm text-stone-500';

                if (cachedLoaded) {
                    el.textContent = 'Couldn\'t fetch latest currencies — loaded cached list.';
                    list.appendChild(el);
                    const note = document.createElement('div');
                    note.className = 'px-3 py-1 text-xs text-stone-400';
                    note.textContent = 'Prices may be outdated; refresh the page to retry.';
                    list.appendChild(note);
                    renderList();
                } else {
                    el.textContent = 'Failed to load currencies';
                    const retryBtn = document.createElement('button');
                    retryBtn.type = 'button';
                    retryBtn.className = 'ml-3 text-sm text-blue-600 hover:underline';
                    retryBtn.textContent = 'Retry';
                    retryBtn.addEventListener('click', () => { fetchCurrencies(retries, 0); });
                    el.appendChild(retryBtn);
                    list.appendChild(el);
                }
            }
        }

        function renderList() {
            list.innerHTML = '';

            if (!currencies || currencies.length === 0) {
                const el = document.createElement('div');
                el.className = 'px-3 py-2 text-sm text-stone-500';
                el.textContent = 'No currencies available';
                list.appendChild(el);
                return;
            }

            currencies.forEach(c => {
                const el = document.createElement('button');
                el.type = 'button';
                el.className = 'w-full text-left px-3 py-2 hover:bg-stone-50 dark:hover:bg-stone-900';
                el.setAttribute('data-code', c.code);
                el.setAttribute('role', 'option');
                el.setAttribute('tabindex', '0');
                el.style.cursor = 'pointer';
                el.innerHTML = `<div class="flex items-center justify-between"><span>${c.code} ${c.name ? '- ' + c.name : ''}</span><span class="text-sm text-stone-400">${c.symbol || ''}</span></div>`;
                if (c.code === currentCode) {
                    el.classList.add('font-semibold');
                }
                el.addEventListener('click', async (e) => {
                    // Prevent double clicks while a change is in progress
                    if (isSetting) {
                        window.Toast?.info('Currency change in progress, please wait...');
                        return;
                    }

                    // Enter loading state (disables toggle and shows spinner)
                    setLoadingState(true);
                    el.disabled = true;
                    try {
                        const result = await setCurrency(c.code);
                        // Close the dropdown (doToggle is defined below in scope)
                        try { doToggle(false); } catch (err) { /* ignore if not available */ }

                        // Show a toast to the user with the outcome
                        if (result && result.success) {
                            window.Toast?.success(result.message || 'Currency updated');
                        } else {
                            window.Toast?.error((result && result.message) || 'Failed to update currency');
                        }
                    } catch (err) {
                        try { doToggle(false); } catch (e) {}
                        window.Toast?.error('Failed to update currency');
                    } finally {
                        el.disabled = false;
                        setLoadingState(false);
                    }
                });
                el.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        el.click();
                    }
                });
                list.appendChild(el);
            });
        }

        async function showListMessage(msg, type = 'info') {
            list.innerHTML = '';
            const el = document.createElement('div');
            el.className = 'px-3 py-2 text-sm';
            if (type === 'error') el.classList.add('text-red-500'); else el.classList.add('text-stone-500');
            el.textContent = msg;
            list.appendChild(el);
        }

        async function setCurrency(code) {
            try {
                let csrftoken = getCsrfToken();
                if (!csrftoken) {
                    await showListMessage('Unable to find a valid CSRF token. Please reload the page or sign out and sign back in to refresh your session.', 'error');
                    // warning removed
                    return { success: false, message: 'Missing CSRF token' };
                }

                async function doPost(token) {
                    return fetch('/api/v1/i18n/preferences/', {
                        method: 'POST',
                        credentials: 'same-origin',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': token
                        },
                        body: JSON.stringify({ currency_code: code, auto_detect: false })
                    });
                }

                // First attempt
                let resp = await doPost(csrftoken);

                // If server denies the request with a 403, attempt to recover gracefully (CSRF refresh or server-provided token)
                if (resp.status === 403) {
                    // warning removed
                    try {
                        // Prefer JSON body (our custom csrf_failure handler returns JSON with meta.new_csrf_token)
                        const jsonBody = await resp.json().catch(() => null);
                        const newTokenFromBody = jsonBody && jsonBody.meta && jsonBody.meta.new_csrf_token ? jsonBody.meta.new_csrf_token : null;

                        if (newTokenFromBody) {
                            // Write cookie and retry once
                            const secure = location.protocol === 'https:' ? '; Secure' : '';
                            document.cookie = `csrftoken=${newTokenFromBody}; path=/; samesite=Lax${secure}`;
                            const retryResp = await doPost(newTokenFromBody);
                            if (retryResp.ok) {
                                const data = await retryResp.json().catch(() => null);
                                if (data && data.success) {
                                    currentCode = code;
                                    if (currentEl) currentEl.textContent = code;
                                    window.BUNORAA_CURRENCY = window.BUNORAA_CURRENCY || {};
                                    window.BUNORAA_CURRENCY.code = code;
                                    document.dispatchEvent(new CustomEvent('currency:changed', { detail: { code } }));
                                    if (shouldAutoReload()) {
                                        setTimeout(() => location.reload(), 250);
                                    }
                                    return { success: true, message: 'Currency updated', source: 'server' };
                                }
                            }
                        }

                        // Otherwise, try refreshing cookies by fetching the currencies endpoint, then retry once
                        try { await fetch('/api/v1/i18n/currencies/', { credentials: 'same-origin' }); } catch (e) { /* ignore */ }
                        const refreshedToken = getCsrfToken();
                        if (refreshedToken && refreshedToken !== csrftoken) {
                            const retryResp2 = await doPost(refreshedToken);
                            if (retryResp2.ok) {
                                const data = await retryResp2.json().catch(() => null);
                                if (data && data.success) {
                                    currentCode = code;
                                    if (currentEl) currentEl.textContent = code;
                                    window.BUNORAA_CURRENCY = window.BUNORAA_CURRENCY || {};
                                    window.BUNORAA_CURRENCY.code = code;
                                    document.dispatchEvent(new CustomEvent('currency:changed', { detail: { code } }));
                                    setTimeout(() => location.reload(), 250);
                                    return { success: true, message: 'Currency updated', source: 'server' };
                                }
                            }
                        }

                        // Recovery failed, fall back to local preference
                        localStorage.setItem('currency_preference', JSON.stringify({ code, savedAt: Date.now() }));
                        currentCode = code;
                        if (currentEl) currentEl.textContent = code;
                        window.BUNORAA_CURRENCY = window.BUNORAA_CURRENCY || {};
                        window.BUNORAA_CURRENCY.code = code;
                        document.dispatchEvent(new CustomEvent('currency:changed', { detail: { code } }));
                        await showListMessage('Currency set locally for this session. If you are logged in and this persists, please try logging out and in again.', 'error');
                        if (shouldAutoReload()) {
                            setTimeout(() => location.reload(), 350);
                        }
                        return { success: true, message: 'Currency saved locally', source: 'local' };
                    } catch (e) {
                        // error logging removed
                        localStorage.setItem('currency_preference', JSON.stringify({ code, savedAt: Date.now() }));
                        currentCode = code;
                        if (currentEl) currentEl.textContent = code;
                        window.BUNORAA_CURRENCY = window.BUNORAA_CURRENCY || {};
                        window.BUNORAA_CURRENCY.code = code;
                        document.dispatchEvent(new CustomEvent('currency:changed', { detail: { code } }));
                        await showListMessage('Currency set locally for this session. If you are logged in and this persists, please try logging out and in again.', 'error');
                        if (shouldAutoReload()) {
                            setTimeout(() => location.reload(), 350);
                        }
                        return { success: true, message: 'Currency saved locally', source: 'local' };
                    }
                }

                if (!resp.ok) {
                    let bodyText = '';
                    try { bodyText = await resp.text(); } catch (_) { bodyText = String(resp.status); }
                    await showListMessage('Failed to set currency. Please reload the page and try again.', 'error');
                    console.debug('Currency preference response:', resp.status, bodyText);
                    return { success: false, message: 'Server error while setting currency' };
                }

                const data = await resp.json();
                // Update UI and reload to get server-rendered prices
                if (data && data.success) {
                    // Save current code locally for immediate UI update
                    currentCode = code;
                    if (currentEl) currentEl.textContent = code;
                    // Update global client-side currency and notify listeners so SPA pages can refresh without full reload
                    window.BUNORAA_CURRENCY = window.BUNORAA_CURRENCY || {};
                    window.BUNORAA_CURRENCY.code = code;
                    document.dispatchEvent(new CustomEvent('currency:changed', { detail: { code } }));
                    // Give short delay then reload so server-rendered templates re-render with new currency (keeps existing behavior)
                    if (shouldAutoReload()) {
                        setTimeout(() => location.reload(), 250);
                    }
                    return { success: true, message: 'Currency updated', source: 'server' };
                } else {
                    await showListMessage('Could not set currency. See console for details.', 'error');
                    return { success: false, message: 'Could not set currency' };
                }
            } catch (e) {
                console.error('Error setting currency preference', e);
                await showListMessage('Network or server error while setting currency.', 'error');
                return { success: false, message: 'Network or server error while setting currency' };
            }
        }

        // Toggle behavior - use delegated toggling to be resilient
        const doToggle = (show) => {
            const expanded = toggle.getAttribute('aria-expanded') === 'true';
            const next = (typeof show === 'boolean') ? show : !expanded;
            toggle.setAttribute('aria-expanded', String(next));
            dropdown.classList.toggle('hidden', !next);
            if (next) {
                const first = list.querySelector('button');
                if (first) first.focus();
            }
        };

        toggle.addEventListener('click', (e) => {
            e.stopPropagation();
            if (isSetting) {
                window.Toast?.info('Currency change in progress, please wait...');
                return;
            }
            doToggle();
        });

        // Keyboard support: Enter/Space to open, Escape to close
        toggle.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                if (isSetting) {
                    window.Toast?.info('Currency change in progress, please wait...');
                    return;
                }
                toggle.click();
            }
            if (e.key === 'Escape') {
                doToggle(false);
                toggle.focus();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                doToggle(false);
                toggle.focus();
            }
        });

        // Click outside closes
        document.addEventListener('click', (e) => {
            if (!root.contains(e.target)) {
                doToggle(false);
            }
        });

        // Accessibility: keyboard navigation inside list
        list.addEventListener('keydown', (e) => {
            const items = Array.from(list.querySelectorAll('button'));
            const idx = items.indexOf(document.activeElement);
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                const next = items[(idx + 1) % items.length];
                if (next) next.focus();
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                const prev = items[(idx - 1 + items.length) % items.length];
                if (prev) prev.focus();
            } else if (e.key === 'Escape') {
                doToggle(false);
                toggle.focus();
            }
        });

        // Initial load
        fetchCurrencies();
    }
}

export default initCurrencySelector;
