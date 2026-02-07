/**
 * Pre-orders Pages Module
 * @module pages/preorders
 */

const PreordersPage = (function() {
    'use strict';

    const routeMap = window.BUNORAA_ROUTES || {};
    const wizardUrl = routeMap.preordersWizard || '/preorders/create/';
    const landingUrl = routeMap.preordersLanding || '/preorders/';

    /**
     * Initialize the landing page
     */
    async function initLanding() {
        await Promise.all([
            loadFeaturedCategories(),
            loadPopularTemplates(),
            loadStats()
        ]);
    }

    /**
     * Load featured pre-order categories
     */
    async function loadFeaturedCategories() {
        const container = document.getElementById('preorder-categories');
        if (!container) return;

        try {
            const response = await PreordersApi.getCategories({ featured: true, pageSize: 8 });
            const categories = response?.data?.results || response?.data || response?.results || [];

            if (categories.length === 0) {
                container.innerHTML = `
                    <div class="col-span-full text-center py-12">
                        <svg class="w-16 h-16 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                                  d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/>
                        </svg>
                        <p class="text-gray-600 dark:text-gray-400 mb-4">No categories available at the moment</p>
                        <p class="text-sm text-gray-500 dark:text-gray-500">Check back soon or contact us for custom requests</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = categories.map(category => renderCategoryCard(category)).join('');
        } catch (error) {
            console.error('Failed to load pre-order categories:', error);
            container.innerHTML = `
                <div class="col-span-full text-center py-12">
                    <svg class="w-16 h-16 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                              d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/>
                    </svg>
                    <p class="text-gray-600 dark:text-gray-400">No categories available at the moment</p>
                </div>
            `;
        }
    }

    /**
     * Render a category card
     */
    function renderCategoryCard(category) {
        const imageUrl = category.image?.url || category.image || category.thumbnail || '';
        const hasImage = imageUrl && imageUrl.length > 0;
        const escapeHtml = Templates?.escapeHtml || ((s) => s);
        const formatPrice = Templates?.formatPrice || ((p) => `${window.BUNORAA_CURRENCY?.symbol || '৳'}${p}`);

        return `
            <a href="${landingUrl}category/${category.slug}/" 
               class="group bg-white dark:bg-gray-800 rounded-2xl shadow-sm hover:shadow-xl transition-all duration-300 overflow-hidden border border-gray-200 dark:border-gray-700">
                ${hasImage ? `
                    <div class="aspect-video relative overflow-hidden">
                        <img src="${imageUrl}" alt="${escapeHtml(category.name)}" 
                             class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                             loading="lazy">
                        <div class="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                    </div>
                ` : `
                    <div class="aspect-video bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
                        ${category.icon ? `<span class="text-5xl">${category.icon}</span>` : `
                            <svg class="w-16 h-16 text-white/80" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                                      d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/>
                            </svg>
                        `}
                    </div>
                `}
                
                <div class="p-6">
                    <div class="flex items-start justify-between mb-3">
                        <h3 class="text-xl font-bold text-gray-900 dark:text-white group-hover:text-purple-600 transition-colors">
                            ${escapeHtml(category.name)}
                        </h3>
                        ${category.icon ? `<span class="text-2xl">${category.icon}</span>` : ''}
                    </div>
                    
                    ${category.description ? `
                        <p class="text-gray-600 dark:text-gray-400 mb-4 line-clamp-2">
                            ${escapeHtml(category.description)}
                        </p>
                    ` : ''}
                    
                    <div class="flex items-center justify-between text-sm">
                        ${category.base_price ? `
                            <span class="text-gray-500 dark:text-gray-500">
                                Starting from <span class="font-semibold text-purple-600">${formatPrice(category.base_price)}</span>
                            </span>
                        ` : '<span></span>'}
                        ${category.min_production_days && category.max_production_days ? `
                            <span class="text-gray-500 dark:text-gray-500">
                                ${category.min_production_days}-${category.max_production_days} days
                            </span>
                        ` : ''}
                    </div>
                    
                    <div class="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700 flex items-center justify-between">
                        <div class="flex gap-2 flex-wrap">
                            ${category.requires_design ? `
                                <span class="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full">
                                    Design Required
                                </span>
                            ` : ''}
                            ${category.allow_rush_order ? `
                                <span class="text-xs px-2 py-1 bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 rounded-full">
                                    Rush Available
                                </span>
                            ` : ''}
                        </div>
                        <svg class="w-5 h-5 text-gray-400 group-hover:text-purple-600 group-hover:translate-x-1 transition-all flex-shrink-0" 
                             fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                        </svg>
                    </div>
                </div>
            </a>
        `;
    }

    /**
     * Load popular templates
     */
    async function loadPopularTemplates() {
        const container = document.getElementById('preorder-templates');
        if (!container) return;

        // Templates would be loaded from API when available
        // For now, show a placeholder or hide the section
        container.closest('section')?.classList.add('hidden');
    }

    /**
     * Load statistics
     */
    async function loadStats() {
        const statsContainer = document.getElementById('preorder-stats');
        if (!statsContainer) return;

        // Stats can be loaded from API or use defaults
        const stats = {
            totalOrders: '500+',
            happyCustomers: '450+',
            avgRating: '4.9'
        };

        statsContainer.innerHTML = `
            <div class="flex items-center gap-8 justify-center flex-wrap">
                <div class="text-center">
                    <p class="text-3xl font-bold text-purple-600 dark:text-purple-400">${stats.totalOrders}</p>
                    <p class="text-sm text-gray-600 dark:text-gray-400">Orders Completed</p>
                </div>
                <div class="h-12 w-px bg-gray-200 dark:bg-gray-700 hidden sm:block"></div>
                <div class="text-center">
                    <p class="text-3xl font-bold text-purple-600 dark:text-purple-400">${stats.happyCustomers}</p>
                    <p class="text-sm text-gray-600 dark:text-gray-400">Happy Customers</p>
                </div>
                <div class="h-12 w-px bg-gray-200 dark:bg-gray-700 hidden sm:block"></div>
                <div class="text-center">
                    <p class="text-3xl font-bold text-purple-600 dark:text-purple-400">${stats.avgRating}</p>
                    <p class="text-sm text-gray-600 dark:text-gray-400">Average Rating</p>
                </div>
            </div>
        `;
    }

    /**
     * Initialize category detail page
     */
    async function initCategoryDetail(categorySlug) {
        const optionsContainer = document.getElementById('category-options');
        if (!optionsContainer || !categorySlug) return;

        try {
            const category = await PreordersApi.getCategory(categorySlug);
            const options = await PreordersApi.getCategoryOptions(category.id);

            renderCategoryOptions(optionsContainer, options);
        } catch (error) {
            console.error('Failed to load category options:', error);
        }
    }

    /**
     * Render category customization options
     */
    function renderCategoryOptions(container, options) {
        if (!options || options.length === 0) {
            container.innerHTML = '<p class="text-gray-500">No customization options available.</p>';
            return;
        }

        container.innerHTML = options.map(option => `
            <div class="border border-gray-200 dark:border-gray-700 rounded-xl p-4">
                <h4 class="font-semibold text-gray-900 dark:text-white mb-2">${Templates.escapeHtml(option.name)}</h4>
                ${option.description ? `<p class="text-sm text-gray-600 dark:text-gray-400 mb-3">${Templates.escapeHtml(option.description)}</p>` : ''}
                <div class="space-y-2">
                    ${option.choices?.map(choice => `
                        <label class="flex items-center gap-3 p-3 border border-gray-200 dark:border-gray-600 rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
                            <input type="${option.allow_multiple ? 'checkbox' : 'radio'}" name="option_${option.id}" value="${choice.id}" class="text-purple-600 focus:ring-purple-500">
                            <span class="flex-1">
                                <span class="font-medium text-gray-900 dark:text-white">${Templates.escapeHtml(choice.name)}</span>
                                ${choice.price_modifier && choice.price_modifier !== '0.00' ? `
                                    <span class="text-sm text-purple-600 dark:text-purple-400 ml-2">+${Templates.formatPrice(choice.price_modifier)}</span>
                                ` : ''}
                            </span>
                        </label>
                    `).join('') || ''}
                </div>
            </div>
        `).join('');
    }

    /**
     * Initialize my pre-orders page
     */
    async function initMyPreorders() {
        const container = document.getElementById('my-preorders-list');
        if (!container) return;

        Loader.show(container, 'skeleton');

        try {
            const response = await PreordersApi.getMyPreorders();
            const preorders = response?.data?.results || response?.data || response?.results || [];

            if (preorders.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-12">
                        <svg class="w-20 h-20 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                        </svg>
                        <h3 class="text-xl font-semibold text-gray-900 dark:text-white mb-2">No custom orders yet</h3>
                        <p class="text-gray-600 dark:text-gray-400 mb-6">Start creating your first custom order today!</p>
                        <a href="${wizardUrl}" class="inline-flex items-center gap-2 px-6 py-3 bg-purple-600 text-white font-semibold rounded-xl hover:bg-purple-700 transition-colors">
                            Create Custom Order
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>
                        </a>
                    </div>
                `;
                return;
            }

            container.innerHTML = preorders.map(preorder => renderPreorderCard(preorder)).join('');
        } catch (error) {
            console.error('Failed to load pre-orders:', error);
            container.innerHTML = `
                <div class="text-center py-12">
                    <p class="text-red-500">Failed to load your orders. Please try again.</p>
                </div>
            `;
        }
    }

    /**
     * Render a pre-order card
     */
    function renderPreorderCard(preorder) {
        const statusColors = {
            'pending': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
            'quoted': 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
            'accepted': 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-400',
            'in_progress': 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400',
            'review': 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400',
            'approved': 'bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-400',
            'completed': 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
            'cancelled': 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
            'refunded': 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400'
        };

        const statusLabels = {
            'pending': 'Pending Review',
            'quoted': 'Quote Sent',
            'accepted': 'Quote Accepted',
            'in_progress': 'In Progress',
            'review': 'Under Review',
            'approved': 'Approved',
            'completed': 'Completed',
            'cancelled': 'Cancelled',
            'refunded': 'Refunded'
        };

        const statusClass = statusColors[preorder.status] || 'bg-gray-100 text-gray-800';
        const statusLabel = statusLabels[preorder.status] || preorder.status;

        return `
            <a href="${landingUrl}order/${preorder.preorder_number}/" class="block bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6 hover:shadow-lg transition-shadow">
                <div class="flex items-start justify-between gap-4 mb-4">
                    <div>
                        <p class="text-sm text-gray-500 dark:text-gray-400">${preorder.preorder_number}</p>
                        <h3 class="text-lg font-semibold text-gray-900 dark:text-white">${Templates.escapeHtml(preorder.title || preorder.category?.name || 'Custom Order')}</h3>
                    </div>
                    <span class="px-3 py-1 text-xs font-medium rounded-full ${statusClass}">${statusLabel}</span>
                </div>
                ${preorder.description ? `
                    <p class="text-gray-600 dark:text-gray-400 text-sm mb-4 line-clamp-2">${Templates.escapeHtml(preorder.description)}</p>
                ` : ''}
                <div class="flex items-center justify-between text-sm">
                    <span class="text-gray-500 dark:text-gray-400">
                        ${new Date(preorder.created_at).toLocaleDateString()}
                    </span>
                    ${preorder.final_price || preorder.estimated_price ? `
                        <span class="font-semibold text-purple-600 dark:text-purple-400">
                            ${Templates.formatPrice(preorder.final_price || preorder.estimated_price)}
                        </span>
                    ` : ''}
                </div>
            </a>
        `;
    }

    /**
     * Initialize pre-order detail page
     */
    async function initDetail(preorderNumber) {
        if (!preorderNumber) return;

        // Load messages, status updates, etc.
        await Promise.all([
            loadPreorderStatus(preorderNumber),
            initMessageForm(preorderNumber)
        ]);
    }

    /**
     * Load pre-order status
     */
    async function loadPreorderStatus(preorderNumber) {
        const statusContainer = document.getElementById('preorder-status');
        if (!statusContainer) return;

        try {
            const status = await PreordersApi.getPreorderStatus(preorderNumber);
            // Update UI with status
        } catch (error) {
        }
    }

    /**
     * Initialize message form
     */
    function initMessageForm(preorderNumber) {
        const form = document.getElementById('message-form');
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const messageInput = form.querySelector('textarea[name="message"]');
            const submitBtn = form.querySelector('button[type="submit"]');
            const message = messageInput?.value?.trim();

            if (!message) {
                Toast.error('Please enter a message');
                return;
            }

            const originalText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<svg class="animate-spin h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';

            try {
                await PreordersApi.sendMessage(preorderNumber, message);
                Toast.success('Message sent successfully');
                messageInput.value = '';
                // Reload messages
                location.reload();
            } catch (error) {
                Toast.error(error.message || 'Failed to send message');
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
            }
        });
    }

    return {
        initLanding,
        initCategoryDetail,
        initMyPreorders,
        initDetail,
        loadFeaturedCategories,
        renderCategoryCard,
        renderPreorderCard
    };
})();

window.PreordersPage = PreordersPage;
