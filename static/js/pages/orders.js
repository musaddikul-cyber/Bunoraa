/**
 * Orders Page - Enhanced with Advanced Features
 * @module pages/orders
 */

const OrdersPage = (function() {
    'use strict';

    let currentPage = 1;
    let currentFilter = 'all';

    async function init() {
        if (!AuthGuard.protectPage()) return;

        const orderId = getOrderIdFromUrl();
        if (orderId) {
            await loadOrderDetail(orderId);
        } else {
            await loadOrders();
            initFilters();
        }
    }

    function getOrderIdFromUrl() {
        const path = window.location.pathname;
        const match = path.match(/\/orders\/([^\/]+)/);
        return match ? match[1] : null;
    }

    async function loadOrders() {
        const container = document.getElementById('orders-list');
        if (!container) return;

        Loader.show(container, 'skeleton');

        try {
            const params = {
                page: currentPage,
                limit: 10
            };

            if (currentFilter !== 'all') {
                params.status = currentFilter;
            }

            const response = await OrdersApi.getAll(params);
            const orders = response.data || [];
            const meta = response.meta || {};

            renderOrders(orders, meta);
        } catch (error) {
            console.error('Failed to load orders:', error);
            container.innerHTML = '<p class="text-red-500 text-center py-8">Failed to load orders.</p>';
        }
    }

    function renderOrders(orders, meta) {
        const container = document.getElementById('orders-list');
        if (!container) return;

        if (orders.length === 0) {
            container.innerHTML = `
                <div class="text-center py-16">
                    <svg class="w-24 h-24 text-gray-300 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
                    </svg>
                    <h2 class="text-2xl font-bold text-gray-900 mb-2">No orders yet</h2>
                    <p class="text-gray-600 mb-8">When you place an order, it will appear here.</p>
                    <a href="/products/" class="inline-flex items-center px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 transition-colors">
                        Start Shopping
                    </a>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="space-y-4">
                ${orders.map(order => renderOrderCard(order)).join('')}
            </div>
            ${meta.total_pages > 1 ? `
                <div id="orders-pagination" class="mt-8">${Pagination.render({
                    currentPage: meta.current_page || currentPage,
                    totalPages: meta.total_pages,
                    totalItems: meta.total
                })}</div>
            ` : ''}
        `;

        const paginationContainer = document.getElementById('orders-pagination');
        paginationContainer?.addEventListener('click', (e) => {
            const pageBtn = e.target.closest('[data-page]');
            if (pageBtn) {
                currentPage = parseInt(pageBtn.dataset.page);
                loadOrders();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });
    }

    function renderOrderCard(order) {
        const statusClasses = {
            pending: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
            processing: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
            shipped: 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400',
            delivered: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
            cancelled: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
            refunded: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400'
        };

        const statusClass = statusClasses[order.status] || 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400';
        const items = order.items || [];
        const displayItems = items.slice(0, 3);
        const remainingCount = items.length - 3;

        // Visual progress steps
        const steps = ['pending', 'processing', 'shipped', 'delivered'];
        const currentStepIndex = steps.indexOf(order.status);
        const isCancelled = order.status === 'cancelled' || order.status === 'refunded';

        return `
            <div class="bg-white dark:bg-stone-800 rounded-xl shadow-sm border border-gray-100 dark:border-stone-700 overflow-hidden">
                <div class="px-6 py-4 border-b border-gray-100 dark:border-stone-700 flex flex-wrap items-center justify-between gap-4">
                    <div>
                        <p class="text-sm text-gray-500 dark:text-stone-400">Order #${Templates.escapeHtml(order.order_number || order.id)}</p>
                        <p class="text-sm text-gray-500 dark:text-stone-400">Placed on ${Templates.formatDate(order.created_at)}</p>
                    </div>
                    <div class="flex items-center gap-4">
                        <span class="px-3 py-1 rounded-full text-sm font-medium ${statusClass}">
                            ${Templates.escapeHtml(order.status_display || order.status)}
                        </span>
                        <a href="/orders/${order.id}/" class="text-primary-600 dark:text-amber-400 hover:text-primary-700 dark:hover:text-amber-300 font-medium text-sm">
                            View Details
                        </a>
                    </div>
                </div>
                
                <!-- Visual Progress Bar -->
                ${!isCancelled ? `
                    <div class="px-6 py-3 bg-stone-50 dark:bg-stone-900/50 border-b border-gray-100 dark:border-stone-700">
                        <div class="flex items-center justify-between relative">
                            <div class="absolute left-0 right-0 top-1/2 h-1 bg-stone-200 dark:bg-stone-700 -translate-y-1/2 rounded-full"></div>
                            <div class="absolute left-0 top-1/2 h-1 bg-primary-500 dark:bg-amber-500 -translate-y-1/2 rounded-full transition-all duration-500" style="width: ${Math.max(0, (currentStepIndex / (steps.length - 1)) * 100)}%"></div>
                            ${steps.map((step, i) => `
                                <div class="relative z-10 flex flex-col items-center">
                                    <div class="w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${i <= currentStepIndex ? 'bg-primary-500 dark:bg-amber-500 text-white' : 'bg-stone-200 dark:bg-stone-700 text-stone-500 dark:text-stone-400'}">
                                        ${i < currentStepIndex ? '<svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>' : (i + 1)}
                                    </div>
                                    <span class="text-xs text-stone-500 dark:text-stone-400 mt-1 capitalize hidden sm:block">${step}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                
                <div class="p-6">
                    <div class="flex flex-wrap gap-4">
                        ${displayItems.map(item => `
                            <div class="flex items-center gap-3">
                                <div class="w-16 h-16 bg-gray-100 dark:bg-stone-700 rounded-lg overflow-hidden flex-shrink-0">
                                    ${item.product?.image ? `<img src="${item.product.image}" alt="" class="w-full h-full object-cover" onerror="this.parentElement.innerHTML='<div class=\\'w-full h-full flex items-center justify-center text-gray-400 dark:text-stone-500\\'><svg class=\\'w-6 h-6\\' fill=\\'none\\' stroke=\\'currentColor\\' viewBox=\\'0 0 24 24\\'><path stroke-linecap=\\'round\\' stroke-linejoin=\\'round\\' stroke-width=\\'1.5\\' d=\\'M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z\\'/></svg></div>'">` : `<div class="w-full h-full flex items-center justify-center text-gray-400 dark:text-stone-500"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg></div>`}
                                </div>
                                <div>
                                    <p class="text-sm font-medium text-gray-900 dark:text-white">${Templates.escapeHtml(item.product?.name || item.product_name)}</p>
                                    <p class="text-sm text-gray-500 dark:text-stone-400">Qty: ${item.quantity}</p>
                                </div>
                            </div>
                        `).join('')}
                        ${remainingCount > 0 ? `
                            <div class="flex items-center justify-center w-16 h-16 bg-gray-100 dark:bg-stone-700 rounded-lg">
                                <span class="text-sm text-gray-500 dark:text-stone-400">+${remainingCount}</span>
                            </div>
                        ` : ''}
                    </div>
                    <div class="mt-4 pt-4 border-t border-gray-100 flex justify-between items-center">
                        <p class="text-sm text-gray-600">
                            ${items.length} ${items.length === 1 ? 'item' : 'items'}
                        </p>
                        <p class="font-semibold text-gray-900">Total: ${Templates.formatPrice(order.total)}</p>
                    </div>
                </div>
            </div>
        `;
    }

    async function loadOrderDetail(orderId) {
        const container = document.getElementById('order-detail');
        if (!container) return;

        Loader.show(container, 'skeleton');

        try {
            const response = await OrdersApi.getById(orderId);
            const order = response.data;

            if (!order) {
                window.location.href = '/orders/';
                return;
            }

            renderOrderDetail(order);
        } catch (error) {
            console.error('Failed to load order:', error);
            container.innerHTML = '<p class="text-red-500 text-center py-8">Failed to load order details.</p>';
        }
    }

    function renderOrderDetail(order) {
        const container = document.getElementById('order-detail');
        if (!container) return;

        const statusClasses = {
            pending: 'bg-yellow-100 text-yellow-700',
            processing: 'bg-blue-100 text-blue-700',
            shipped: 'bg-indigo-100 text-indigo-700',
            delivered: 'bg-green-100 text-green-700',
            cancelled: 'bg-red-100 text-red-700',
            refunded: 'bg-gray-100 text-gray-700'
        };

        const statusClass = statusClasses[order.status] || 'bg-gray-100 text-gray-700';
        const items = order.items || [];

        container.innerHTML = `
            <div class="mb-6">
                <a href="/orders/" class="inline-flex items-center text-primary-600 hover:text-primary-700">
                    <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16l-4-4m0 0l4-4m-4 4h18"/>
                    </svg>
                    Back to Orders
                </a>
            </div>

            <div class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                <div class="px-6 py-4 border-b border-gray-100">
                    <div class="flex flex-wrap items-center justify-between gap-4">
                        <div>
                            <h1 class="text-xl font-bold text-gray-900">Order #${Templates.escapeHtml(order.order_number || order.id)}</h1>
                            <p class="text-sm text-gray-500">Placed on ${Templates.formatDate(order.created_at)}</p>
                        </div>
                        <span class="px-4 py-1.5 rounded-full text-sm font-medium ${statusClass}">
                            ${Templates.escapeHtml(order.status_display || order.status)}
                        </span>
                    </div>
                </div>

                <!-- Order Timeline -->
                ${order.timeline && order.timeline.length > 0 ? `
                    <div class="px-6 py-4 border-b border-gray-100">
                        <h2 class="text-sm font-semibold text-gray-900 mb-4">Order Timeline</h2>
                        <div class="relative">
                            <div class="absolute left-2 top-2 bottom-2 w-0.5 bg-gray-200"></div>
                            <div class="space-y-4">
                                ${order.timeline.map((event, index) => `
                                    <div class="relative pl-8">
                                        <div class="absolute left-0 w-4 h-4 rounded-full ${index === 0 ? 'bg-primary-600' : 'bg-gray-300'}"></div>
                                        <p class="text-sm font-medium text-gray-900">${Templates.escapeHtml(event.status)}</p>
                                        <p class="text-xs text-gray-500">${Templates.formatDate(event.timestamp, { includeTime: true })}</p>
                                        ${event.note ? `<p class="text-sm text-gray-600 mt-1">${Templates.escapeHtml(event.note)}</p>` : ''}
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                ` : ''}

                <!-- Tracking Info -->
                ${order.tracking_number ? `
                    <div class="px-6 py-4 border-b border-gray-100 bg-blue-50">
                        <div class="flex items-center justify-between">
                            <div>
                                <p class="text-sm font-medium text-blue-900">Tracking Number</p>
                                <p class="text-lg font-mono text-blue-700">${Templates.escapeHtml(order.tracking_number)}</p>
                            </div>
                            ${order.tracking_url ? `
                                <a href="${order.tracking_url}" target="_blank" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium">
                                    Track Package
                                </a>
                            ` : ''}
                        </div>
                    </div>
                ` : ''}

                <!-- Order Items -->
                <div class="px-6 py-4 border-b border-gray-100">
                    <h2 class="text-sm font-semibold text-gray-900 mb-4">Items Ordered</h2>
                    <div class="space-y-4">
                        ${items.map(item => `
                            <div class="flex gap-4">
                                <div class="w-20 h-20 bg-gray-100 rounded-lg overflow-hidden flex-shrink-0">
                                    ${item.product?.image ? `<img src="${item.product.image}" alt="" class="w-full h-full object-cover" onerror="this.style.display='none'">` : ''}
                                </div>
                                <div class="flex-1">
                                    <div class="flex justify-between">
                                        <div>
                                            <h3 class="font-medium text-gray-900">${Templates.escapeHtml(item.product?.name || item.product_name)}</h3>
                                            ${item.variant ? `<p class="text-sm text-gray-500">${Templates.escapeHtml(item.variant.name || item.variant_name)}</p>` : ''}
                                            <p class="text-sm text-gray-500">Qty: ${item.quantity}</p>
                                        </div>
                                        <p class="font-medium text-gray-900">${Templates.formatPrice(item.price * item.quantity)}</p>
                                    </div>
                                    ${item.product?.slug ? `
                                        <a href="/products/${item.product.slug}/" class="text-sm text-primary-600 hover:text-primary-700 mt-2 inline-block">
                                            View Product
                                        </a>
                                    ` : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <!-- Addresses -->
                <div class="px-6 py-4 border-b border-gray-100 grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h2 class="text-sm font-semibold text-gray-900 mb-2">Shipping Address</h2>
                        ${order.shipping_address ? `
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(order.shipping_address.full_name || `${order.shipping_address.first_name} ${order.shipping_address.last_name}`)}</p>
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(order.shipping_address.address_line_1)}</p>
                            ${order.shipping_address.address_line_2 ? `<p class="text-sm text-gray-600">${Templates.escapeHtml(order.shipping_address.address_line_2)}</p>` : ''}
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(order.shipping_address.city)}, ${Templates.escapeHtml(order.shipping_address.state || '')} ${Templates.escapeHtml(order.shipping_address.postal_code)}</p>
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(order.shipping_address.country)}</p>
                        ` : '<p class="text-sm text-gray-500">Not available</p>'}
                    </div>
                    <div>
                        <h2 class="text-sm font-semibold text-gray-900 mb-2">Billing Address</h2>
                        ${order.billing_address ? `
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(order.billing_address.full_name || `${order.billing_address.first_name} ${order.billing_address.last_name}`)}</p>
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(order.billing_address.address_line_1)}</p>
                            ${order.billing_address.address_line_2 ? `<p class="text-sm text-gray-600">${Templates.escapeHtml(order.billing_address.address_line_2)}</p>` : ''}
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(order.billing_address.city)}, ${Templates.escapeHtml(order.billing_address.state || '')} ${Templates.escapeHtml(order.billing_address.postal_code)}</p>
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(order.billing_address.country)}</p>
                        ` : '<p class="text-sm text-gray-500">Same as shipping</p>'}
                    </div>
                </div>

                <!-- Order Summary -->
                <div class="px-6 py-4">
                    <h2 class="text-sm font-semibold text-gray-900 mb-4">Order Summary</h2>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between">
                            <span class="text-gray-600">Subtotal</span>
                            <span class="font-medium">${Templates.formatPrice(order.subtotal || 0)}</span>
                        </div>
                        ${order.discount_amount ? `
                            <div class="flex justify-between text-green-600">
                                <span>Discount</span>
                                <span>-${Templates.formatPrice(order.discount_amount)}</span>
                            </div>
                        ` : ''}
                        <div class="flex justify-between">
                            <span class="text-gray-600">Shipping</span>
                            <span class="font-medium">${order.shipping_cost > 0 ? Templates.formatPrice(order.shipping_cost) : 'Free'}</span>
                        </div>
                        ${order.tax_amount ? `
                            <div class="flex justify-between">
                                <span class="text-gray-600">Tax</span>
                                <span class="font-medium">${Templates.formatPrice(order.tax_amount)}</span>
                            </div>
                        ` : ''}
                        <div class="flex justify-between pt-2 border-t border-gray-200">
                            <span class="font-semibold text-gray-900">Total</span>
                            <span class="font-bold text-gray-900">${Templates.formatPrice(order.total)}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Actions -->
            <div class="mt-6 flex flex-wrap gap-4">
                ${order.status === 'delivered' ? `
                    <button id="reorder-btn" class="px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 transition-colors" data-order-id="${order.id}">
                        Order Again
                    </button>
                ` : ''}
                ${['pending', 'processing'].includes(order.status) ? `
                    <button id="cancel-order-btn" class="px-6 py-3 border border-red-300 text-red-600 font-semibold rounded-lg hover:bg-red-50 transition-colors" data-order-id="${order.id}">
                        Cancel Order
                    </button>
                ` : ''}
                <button id="print-invoice-btn" class="px-6 py-3 border border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-colors">
                    Print Invoice
                </button>
            </div>
        `;

        bindOrderDetailEvents(order);
    }

    function bindOrderDetailEvents(order) {
        const reorderBtn = document.getElementById('reorder-btn');
        const cancelBtn = document.getElementById('cancel-order-btn');
        const printBtn = document.getElementById('print-invoice-btn');

        reorderBtn?.addEventListener('click', async () => {
            try {
                await OrdersApi.reorder(order.id);
                Toast.success('Items added to cart!');
                document.dispatchEvent(new CustomEvent('cart:updated'));
                window.location.href = '/cart/';
            } catch (error) {
                Toast.error(error.message || 'Failed to reorder.');
            }
        });

        cancelBtn?.addEventListener('click', async () => {
            const confirmed = await Modal.confirm({
                title: 'Cancel Order',
                message: 'Are you sure you want to cancel this order? This action cannot be undone.',
                confirmText: 'Cancel Order',
                cancelText: 'Keep Order'
            });

            if (confirmed) {
                try {
                    await OrdersApi.cancel(order.id);
                    Toast.success('Order cancelled.');
                    window.location.reload();
                } catch (error) {
                    Toast.error(error.message || 'Failed to cancel order.');
                }
            }
        });

        printBtn?.addEventListener('click', () => {
            window.print();
        });
    }

    function initFilters() {
        const filterBtns = document.querySelectorAll('[data-filter-status]');
        
        filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                filterBtns.forEach(b => {
                    b.classList.remove('bg-primary-100', 'text-primary-700');
                    b.classList.add('text-gray-600');
                });
                btn.classList.add('bg-primary-100', 'text-primary-700');
                btn.classList.remove('text-gray-600');

                currentFilter = btn.dataset.filterStatus;
                currentPage = 1;
                loadOrders();
            });
        });
    }

    function destroy() {
        currentPage = 1;
        currentFilter = 'all';
    }

    return {
        init,
        destroy
    };
})();

window.OrdersPage = OrdersPage;
export default OrdersPage;
