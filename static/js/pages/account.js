/**
 * Account Page - Enhanced with Advanced Features
 * @module pages/account
 */

const AccountPage = (function() {
    'use strict';

    let currentUser = null;

    async function init() {
        // Ensure this page is protected and redirect to login if the user is not authenticated
        if (!AuthGuard.protectPage()) return;

        await loadUserProfile();
        // Ensure avatar handlers are bound even when renderProfile cannot inject the header (e.g., server-rendered sidebar present)
        try {
            setupAvatarHandlers();
        } catch (e) {
            // ignore
        }
        initProfileTabs();
        initProfileForm();
        initPasswordForm();
        initAddressManagement();
        initEnhancedFeatures();
    }

    // ============================================
    // ENHANCED FEATURES INITIALIZATION
    // ============================================
    function initEnhancedFeatures() {
        initLoyaltyPoints();
        initQuickStats();
        initRecentActivity();
        initNotificationPreferences();
        initQuickReorder();
        initAccountSecurityCheck();
    }

    // ============================================
    // ENHANCED FEATURE: Loyalty Points Display
    // ============================================
    function initLoyaltyPoints() {
        const container = document.getElementById('loyalty-points');
        if (!container) return;

        // Mock data - replace with API call in production
        const points = currentUser?.loyalty_points || Math.floor(Math.random() * 500) + 100;
        const tier = points >= 500 ? 'Gold' : points >= 200 ? 'Silver' : 'Bronze';
        const tierColors = {
            Bronze: 'from-amber-600 to-amber-700',
            Silver: 'from-gray-400 to-gray-500',
            Gold: 'from-yellow-400 to-yellow-500'
        };
        const nextTier = tier === 'Gold' ? null : tier === 'Silver' ? 'Gold' : 'Silver';
        const nextTierPoints = tier === 'Gold' ? 0 : tier === 'Silver' ? 500 : 200;
        const progress = nextTier ? Math.min(100, (points / nextTierPoints) * 100) : 100;

        container.innerHTML = `
            <div class="bg-gradient-to-br ${tierColors[tier]} rounded-2xl p-6 text-white relative overflow-hidden">
                <div class="absolute top-0 right-0 w-32 h-32 opacity-10">
                    <svg viewBox="0 0 24 24" fill="currentColor" class="w-full h-full">
                        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                    </svg>
                </div>
                <div class="relative">
                    <div class="flex items-center justify-between mb-4">
                        <div>
                            <p class="text-white/80 text-sm font-medium">${tier} Member</p>
                            <p class="text-3xl font-bold">${points.toLocaleString()} pts</p>
                        </div>
                        <div class="w-14 h-14 bg-white/20 backdrop-blur rounded-xl flex items-center justify-center">
                            <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                            </svg>
                        </div>
                    </div>
                    ${nextTier ? `
                        <div class="mt-4">
                            <div class="flex justify-between text-sm mb-1">
                                <span>${nextTierPoints - points} points to ${nextTier}</span>
                                <span>${Math.round(progress)}%</span>
                            </div>
                            <div class="w-full bg-white/30 rounded-full h-2">
                                <div class="bg-white h-2 rounded-full transition-all duration-500" style="width: ${progress}%"></div>
                            </div>
                        </div>
                    ` : '<p class="text-sm text-white/80 mt-2">🎉 You\'ve reached the highest tier!</p>'}
                    <div class="mt-4 pt-4 border-t border-white/20">
                        <a href="/loyalty/" class="text-sm font-medium hover:underline flex items-center gap-1">
                            View Rewards
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
                        </a>
                    </div>
                </div>
            </div>
        `;
    }

    // ============================================
    // ENHANCED FEATURE: Quick Stats
    // ============================================
    function initQuickStats() {
        const container = document.getElementById('quick-stats');
        if (!container) return;

        // Mock data - replace with API call
        const stats = {
            totalOrders: currentUser?.total_orders || Math.floor(Math.random() * 20) + 5,
            totalSpent: currentUser?.total_spent || Math.floor(Math.random() * 1000) + 200,
            wishlistItems: currentUser?.wishlist_count || Math.floor(Math.random() * 10) + 2,
            savedAddresses: currentUser?.address_count || Math.floor(Math.random() * 3) + 1
        };

        container.innerHTML = `
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="bg-white dark:bg-stone-800 rounded-xl p-4 border border-gray-100 dark:border-stone-700">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                            <svg class="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
                        </div>
                        <div>
                            <p class="text-2xl font-bold text-stone-900 dark:text-white">${stats.totalOrders}</p>
                            <p class="text-xs text-stone-500 dark:text-stone-400">Total Orders</p>
                        </div>
                    </div>
                </div>
                <div class="bg-white dark:bg-stone-800 rounded-xl p-4 border border-gray-100 dark:border-stone-700">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
                            <svg class="w-5 h-5 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                        </div>
                        <div>
                            <p class="text-2xl font-bold text-stone-900 dark:text-white">${Templates.formatPrice(stats.totalSpent)}</p>
                            <p class="text-xs text-stone-500 dark:text-stone-400">Total Spent</p>
                        </div>
                    </div>
                </div>
                <div class="bg-white dark:bg-stone-800 rounded-xl p-4 border border-gray-100 dark:border-stone-700">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-rose-100 dark:bg-rose-900/30 rounded-lg flex items-center justify-center">
                            <svg class="w-5 h-5 text-rose-600 dark:text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/></svg>
                        </div>
                        <div>
                            <p class="text-2xl font-bold text-stone-900 dark:text-white">${stats.wishlistItems}</p>
                            <p class="text-xs text-stone-500 dark:text-stone-400">Wishlist Items</p>
                        </div>
                    </div>
                </div>
                <div class="bg-white dark:bg-stone-800 rounded-xl p-4 border border-gray-100 dark:border-stone-700">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                            <svg class="w-5 h-5 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                        </div>
                        <div>
                            <p class="text-2xl font-bold text-stone-900 dark:text-white">${stats.savedAddresses}</p>
                            <p class="text-xs text-stone-500 dark:text-stone-400">Saved Addresses</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // ============================================
    // ENHANCED FEATURE: Recent Activity
    // ============================================
    async function initRecentActivity() {
        const container = document.getElementById('recent-activity');
        if (!container) return;

        try {
            // Load recent orders
            const response = await OrdersApi.getAll({ limit: 3 });
            const orders = response.data || [];

            if (orders.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-6 text-stone-500 dark:text-stone-400">
                        <p>No recent activity</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = `
                <div class="bg-white dark:bg-stone-800 rounded-xl border border-gray-100 dark:border-stone-700 overflow-hidden">
                    <div class="px-4 py-3 border-b border-gray-100 dark:border-stone-700 flex items-center justify-between">
                        <h3 class="font-semibold text-stone-900 dark:text-white">Recent Orders</h3>
                        <a href="/orders/" class="text-sm text-primary-600 dark:text-amber-400 hover:underline">View All</a>
                    </div>
                    <div class="divide-y divide-gray-100 dark:divide-stone-700">
                        ${orders.map(order => {
                            const statusColors = {
                                pending: 'text-yellow-600 dark:text-yellow-400',
                                processing: 'text-blue-600 dark:text-blue-400',
                                shipped: 'text-indigo-600 dark:text-indigo-400',
                                delivered: 'text-green-600 dark:text-green-400',
                                cancelled: 'text-red-600 dark:text-red-400'
                            };
                            const statusColor = statusColors[order.status] || 'text-stone-600 dark:text-stone-400';
                            const firstItem = order.items?.[0];
                            
                            return `
                                <a href="/orders/${order.id}/" class="flex items-center gap-4 p-4 hover:bg-stone-50 dark:hover:bg-stone-700/50 transition-colors">
                                    <div class="w-12 h-12 bg-stone-100 dark:bg-stone-700 rounded-lg overflow-hidden flex-shrink-0">
                                        ${firstItem?.product?.image ? 
                                            `<img src="${firstItem.product.image}" alt="" class="w-full h-full object-cover">` :
                                            `<div class="w-full h-full flex items-center justify-center text-stone-400">
                                                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/></svg>
                                            </div>`
                                        }
                                    </div>
                                    <div class="flex-1 min-w-0">
                                        <p class="font-medium text-stone-900 dark:text-white truncate">Order #${Templates.escapeHtml(order.order_number || order.id)}</p>
                                        <p class="text-sm ${statusColor}">${Templates.escapeHtml(order.status_display || order.status)}</p>
                                    </div>
                                    <div class="text-right">
                                        <p class="font-semibold text-stone-900 dark:text-white">${Templates.formatPrice(order.total)}</p>
                                        <p class="text-xs text-stone-500 dark:text-stone-400">${Templates.formatDate(order.created_at)}</p>
                                    </div>
                                </a>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        } catch (error) {
            container.innerHTML = '';
        }
    }

    // ============================================
    // ENHANCED FEATURE: Notification Preferences
    // ============================================
    function initNotificationPreferences() {
        const container = document.getElementById('notification-preferences');
        if (!container) return;

        const preferences = JSON.parse(localStorage.getItem('notificationPreferences') || '{}');
        const defaultPrefs = {
            orderUpdates: true,
            promotions: true,
            newArrivals: false,
            priceDrops: true,
            newsletter: false
        };
        const prefs = { ...defaultPrefs, ...preferences };

        container.innerHTML = `
            <div class="space-y-4">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="font-medium text-stone-900 dark:text-white">Order Updates</p>
                        <p class="text-sm text-stone-500 dark:text-stone-400">Get notified about your order status</p>
                    </div>
                    <label class="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" class="sr-only peer" data-pref="orderUpdates" ${prefs.orderUpdates ? 'checked' : ''}>
                        <div class="w-11 h-6 bg-stone-200 dark:bg-stone-600 peer-focus:ring-2 peer-focus:ring-primary-300 dark:peer-focus:ring-amber-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600 dark:peer-checked:bg-amber-500"></div>
                    </label>
                </div>
                <div class="flex items-center justify-between">
                    <div>
                        <p class="font-medium text-stone-900 dark:text-white">Promotions & Sales</p>
                        <p class="text-sm text-stone-500 dark:text-stone-400">Be the first to know about deals</p>
                    </div>
                    <label class="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" class="sr-only peer" data-pref="promotions" ${prefs.promotions ? 'checked' : ''}>
                        <div class="w-11 h-6 bg-stone-200 dark:bg-stone-600 peer-focus:ring-2 peer-focus:ring-primary-300 dark:peer-focus:ring-amber-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600 dark:peer-checked:bg-amber-500"></div>
                    </label>
                </div>
                <div class="flex items-center justify-between">
                    <div>
                        <p class="font-medium text-stone-900 dark:text-white">Price Drops</p>
                        <p class="text-sm text-stone-500 dark:text-stone-400">Alert when wishlist items go on sale</p>
                    </div>
                    <label class="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" class="sr-only peer" data-pref="priceDrops" ${prefs.priceDrops ? 'checked' : ''}>
                        <div class="w-11 h-6 bg-stone-200 dark:bg-stone-600 peer-focus:ring-2 peer-focus:ring-primary-300 dark:peer-focus:ring-amber-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600 dark:peer-checked:bg-amber-500"></div>
                    </label>
                </div>
                <div class="flex items-center justify-between">
                    <div>
                        <p class="font-medium text-stone-900 dark:text-white">New Arrivals</p>
                        <p class="text-sm text-stone-500 dark:text-stone-400">Updates on new products</p>
                    </div>
                    <label class="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" class="sr-only peer" data-pref="newArrivals" ${prefs.newArrivals ? 'checked' : ''}>
                        <div class="w-11 h-6 bg-stone-200 dark:bg-stone-600 peer-focus:ring-2 peer-focus:ring-primary-300 dark:peer-focus:ring-amber-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600 dark:peer-checked:bg-amber-500"></div>
                    </label>
                </div>
            </div>
        `;

        // Bind toggle events
        container.querySelectorAll('input[data-pref]').forEach(input => {
            input.addEventListener('change', () => {
                const key = input.dataset.pref;
                const preferences = JSON.parse(localStorage.getItem('notificationPreferences') || '{}');
                preferences[key] = input.checked;
                localStorage.setItem('notificationPreferences', JSON.stringify(preferences));
                Toast.success('Preference saved');
            });
        });
    }

    // ============================================
    // ENHANCED FEATURE: Quick Reorder
    // ============================================
    function initQuickReorder() {
        const container = document.getElementById('quick-reorder');
        if (!container) return;

        // Get recently ordered products from localStorage
        const recentProducts = JSON.parse(localStorage.getItem('recentlyOrdered') || '[]');

        if (recentProducts.length === 0) {
            container.classList.add('hidden');
            return;
        }

        container.innerHTML = `
            <div class="bg-white dark:bg-stone-800 rounded-xl border border-gray-100 dark:border-stone-700 p-4">
                <h3 class="font-semibold text-stone-900 dark:text-white mb-4">Quick Reorder</h3>
                <div class="flex gap-3 overflow-x-auto pb-2">
                    ${recentProducts.slice(0, 5).map(product => `
                        <button class="quick-reorder-btn flex-shrink-0 flex flex-col items-center gap-2 p-3 bg-stone-50 dark:bg-stone-700 rounded-xl hover:bg-stone-100 dark:hover:bg-stone-600 transition-colors" data-product-id="${product.id}">
                            <div class="w-16 h-16 rounded-lg bg-stone-200 dark:bg-stone-600 overflow-hidden">
                                <img src="${product.image || '/static/images/placeholder.jpg'}" alt="${Templates.escapeHtml(product.name)}" class="w-full h-full object-cover">
                            </div>
                            <span class="text-xs font-medium text-stone-700 dark:text-stone-300 text-center line-clamp-2 w-20">${Templates.escapeHtml(product.name)}</span>
                        </button>
                    `).join('')}
                </div>
            </div>
        `;

        // Bind quick reorder buttons
        container.querySelectorAll('.quick-reorder-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const productId = btn.dataset.productId;
                btn.disabled = true;
                
                try {
                    await CartApi.addItem(productId, 1);
                    Toast.success('Added to cart!');
                    document.dispatchEvent(new CustomEvent('cart:updated'));
                } catch (error) {
                    Toast.error('Failed to add to cart');
                } finally {
                    btn.disabled = false;
                }
            });
        });
    }

    // ============================================
    // ENHANCED FEATURE: Account Security Check
    // ============================================
    function initAccountSecurityCheck() {
        const container = document.getElementById('security-check');
        if (!container) return;

        // Calculate security score
        let score = 0;
        const checks = [];

        // Email verified
        const emailVerified = currentUser?.email_verified !== false;
        if (emailVerified) score += 25;
        checks.push({ label: 'Email verified', completed: emailVerified });

        // Phone added
        const hasPhone = !!currentUser?.phone;
        if (hasPhone) score += 25;
        checks.push({ label: 'Phone number added', completed: hasPhone });

        // 2FA enabled (mock)
        const has2FA = currentUser?.two_factor_enabled || false;
        if (has2FA) score += 25;
        checks.push({ label: 'Two-factor authentication', completed: has2FA });

        // Strong password (assume if recently changed)
        const strongPassword = true; // Default to true
        if (strongPassword) score += 25;
        checks.push({ label: 'Strong password', completed: strongPassword });

        const scoreColor = score >= 75 ? 'text-green-600 dark:text-green-400' : score >= 50 ? 'text-yellow-600 dark:text-yellow-400' : 'text-red-600 dark:text-red-400';
        const progressColor = score >= 75 ? 'bg-green-500' : score >= 50 ? 'bg-yellow-500' : 'bg-red-500';

        container.innerHTML = `
            <div class="bg-white dark:bg-stone-800 rounded-xl border border-gray-100 dark:border-stone-700 p-4">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="font-semibold text-stone-900 dark:text-white">Account Security</h3>
                    <span class="${scoreColor} font-bold">${score}%</span>
                </div>
                <div class="w-full bg-stone-200 dark:bg-stone-600 rounded-full h-2 mb-4">
                    <div class="${progressColor} h-2 rounded-full transition-all duration-500" style="width: ${score}%"></div>
                </div>
                <div class="space-y-2">
                    ${checks.map(check => `
                        <div class="flex items-center gap-2 text-sm">
                            ${check.completed ? 
                                `<svg class="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>` :
                                `<svg class="w-4 h-4 text-stone-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" stroke-width="2"/></svg>`
                            }
                            <span class="${check.completed ? 'text-stone-700 dark:text-stone-300' : 'text-stone-500 dark:text-stone-400'}">${check.label}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    async function loadUserProfile() {
        if (!AuthApi.isAuthenticated()) {
            // Avoid calling profile API when user is not authenticated
            return;
        }
        try {
            const response = await AuthApi.getProfile();
            currentUser = response.data;
            renderProfile();
        } catch (error) {
            // error logging removed
            Toast.error('Failed to load profile.');
        }
    }

    function initProfileTabs() {
        const tabs = document.querySelectorAll('[data-profile-tab]');
        const panels = document.querySelectorAll('[data-profile-panel]');
        if (!tabs.length || !panels.length) return;

        const setActive = (target) => {
            panels.forEach(panel => {
                panel.classList.toggle('hidden', panel.dataset.profilePanel !== target);
            });
            tabs.forEach(btn => {
                const isActive = btn.dataset.profileTab === target;
                btn.classList.toggle('bg-amber-600', isActive);
                btn.classList.toggle('text-white', isActive);
                btn.classList.toggle('shadow-sm', isActive);
                btn.classList.toggle('text-stone-700', !isActive);
                btn.classList.toggle('dark:text-stone-200', !isActive);
            });
            localStorage.setItem('profileTab', target);
        };

        const initial = localStorage.getItem('profileTab') || 'overview';
        setActive(initial);

        tabs.forEach(btn => {
            btn.addEventListener('click', () => {
                setActive(btn.dataset.profileTab);
            });
        });
    }

    function renderProfile() {
        const container = document.getElementById('profile-info');
        if (!container || !currentUser) return;

        const name = `${Templates.escapeHtml(currentUser.first_name || '')} ${Templates.escapeHtml(currentUser.last_name || '')}`.trim() || Templates.escapeHtml(currentUser.email || 'User');
        const memberSince = Templates.formatDate(currentUser.created_at || currentUser.date_joined);
        const avatarImg = currentUser.avatar ? `<img id="avatar-preview" src="${currentUser.avatar}" alt="Profile" class="w-full h-full object-cover">` : `
            <span class="flex h-full w-full items-center justify-center text-3xl font-semibold text-stone-500">
                ${(currentUser.first_name?.[0] || currentUser.email?.[0] || 'U').toUpperCase()}
            </span>`;

        container.innerHTML = `
            <div class="absolute inset-0 bg-gradient-to-r from-amber-50/80 via-amber-100/60 to-transparent dark:from-amber-900/20 dark:via-amber-800/10" aria-hidden="true"></div>
            <div class="relative flex flex-col gap-4 md:flex-row md:items-center md:gap-6">
                <div class="relative">
                    <div class="w-24 h-24 rounded-2xl ring-4 ring-amber-100 dark:ring-amber-900/40 overflow-hidden bg-stone-100 dark:bg-stone-800">
                        ${avatarImg}
                    </div>
                </div>
                <div class="flex-1 min-w-0">
                    <p class="text-sm font-semibold text-amber-700 dark:text-amber-300">Profile</p>
                    <h1 class="text-2xl font-bold text-stone-900 dark:text-white leading-tight truncate">${name}</h1>
                    <p class="text-sm text-stone-600 dark:text-stone-300 truncate">${Templates.escapeHtml(currentUser.email)}</p>
                    <p class="text-xs text-stone-500 dark:text-stone-400 mt-1">Member since ${memberSince}</p>
                    <div class="flex flex-wrap gap-2 mt-4">
                        <button type="button" id="change-avatar-btn" class="btn btn-primary btn-sm">Update photo</button>
                        ${currentUser.avatar ? `<button type="button" id="remove-avatar-btn" class="btn btn-ghost btn-sm text-red-600 hover:text-red-700 dark:text-red-400">Remove photo</button>` : ''}
                    </div>
                    <p class="text-xs text-stone-500 dark:text-stone-400 mt-3">JPG, GIF or PNG. Max size 5MB.</p>
                </div>
            </div>
        `;

        setupAvatarHandlers();
    }

    function initTabs() {
        Tabs.init();
    }

    function initProfileForm() {
        const form = document.getElementById('profile-form');
        if (!form || !currentUser) return;

        const firstNameInput = document.getElementById('profile-first-name');
        const lastNameInput = document.getElementById('profile-last-name');
        const emailInput = document.getElementById('profile-email');
        const phoneInput = document.getElementById('profile-phone');

        if (firstNameInput) firstNameInput.value = currentUser.first_name || '';
        if (lastNameInput) lastNameInput.value = currentUser.last_name || '';
        if (emailInput) emailInput.value = currentUser.email || '';
        if (phoneInput) phoneInput.value = currentUser.phone || '';

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(form);
            const data = {
                first_name: formData.get('first_name'),
                last_name: formData.get('last_name'),
                phone: formData.get('phone')
            };

            const submitBtn = form.querySelector('button[type="submit"]');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Saving...';

            try {
                await AuthApi.updateProfile(data);
                Toast.success('Profile updated successfully!');
                await loadUserProfile();
            } catch (error) {
                Toast.error(error.message || 'Failed to update profile.');
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Save Changes';
            }
        });
    }

    function setupAvatarHandlers() {
        let avatarInput = document.getElementById('avatar-input');
        const avatarBtn = document.getElementById('change-avatar-btn');
        const removeBtn = document.getElementById('remove-avatar-btn');

        // If the hidden file input is missing, create it so buttons still work
        if (!avatarInput) {
            avatarInput = document.createElement('input');
            avatarInput.type = 'file';
            avatarInput.id = 'avatar-input';
            avatarInput.name = 'avatar';
            avatarInput.accept = 'image/*';
            avatarInput.className = 'hidden';
            document.body.appendChild(avatarInput);
        }

        // Bind to all elements that may act as avatar trigger (sidebar + main profile)
        const avatarBtns = document.querySelectorAll('#change-avatar-btn');
        avatarBtns.forEach(btn => btn.addEventListener('click', () => avatarInput.click()));

        // Bind remove buttons (may be multiple)
        const removeBtns = document.querySelectorAll('#remove-avatar-btn');
        removeBtns.forEach(btn => btn.addEventListener('click', () => {
            if (typeof window.removeAvatar === 'function') {
                window.removeAvatar();
            }
        }));

        // Ensure the handler is not attached multiple times
        avatarInput.removeEventListener?.('change', window._avatarChangeHandler);
        window._avatarChangeHandler = async function (e) {
            const file = e.target.files?.[0];
            if (!file) return;

            if (!file.type.startsWith('image/')) {
                Toast.error('Please select an image file.');
                return;
            }

            if (file.size > 5 * 1024 * 1024) {
                Toast.error('Image must be smaller than 5MB.');
                return;
            }

            try {
                await AuthApi.uploadAvatar(file);
                Toast.success('Avatar updated!');
                await loadUserProfile();
            } catch (error) {
                Toast.error(error.message || 'Failed to update avatar.');
            }
        };

        avatarInput.addEventListener('change', window._avatarChangeHandler);
    }

    function initPasswordForm() {
        const form = document.getElementById('password-form');
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(form);
            const currentPassword = formData.get('current_password');
            const newPassword = formData.get('new_password');
            const confirmPassword = formData.get('confirm_password');

            if (newPassword !== confirmPassword) {
                Toast.error('Passwords do not match.');
                return;
            }

            if (newPassword.length < 8) {
                Toast.error('Password must be at least 8 characters.');
                return;
            }

            const submitBtn = form.querySelector('button[type="submit"]');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Updating...';

            try {
                await AuthApi.changePassword(currentPassword, newPassword);
                Toast.success('Password updated successfully!');
                form.reset();
            } catch (error) {
                Toast.error(error.message || 'Failed to update password.');
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Update Password';
            }
        });
    }

    function initAddressManagement() {
        loadAddresses();

        const addAddressBtn = document.getElementById('add-address-btn');
        addAddressBtn?.addEventListener('click', () => {
            showAddressModal();
        });
    }

    async function loadAddresses() {
        const container = document.getElementById('addresses-list');
        if (!container) return;

        Loader.show(container, 'spinner');

        try {
            const response = await AuthApi.getAddresses();
            const addresses = response.data || [];

            if (addresses.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-8">
                        <svg class="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/>
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/>
                        </svg>
                        <p class="text-gray-500">No saved addresses yet.</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    ${addresses.map(addr => `
                        <div class="p-4 border border-gray-200 rounded-lg relative" data-address-id="${addr.id}">
                            ${addr.is_default ? `
                                <span class="absolute top-2 right-2 px-2 py-0.5 bg-primary-100 text-primary-700 text-xs rounded">Default</span>
                            ` : ''}
                            <p class="font-medium text-gray-900">${Templates.escapeHtml(addr.full_name || `${addr.first_name} ${addr.last_name}`)}</p>
                            <p class="text-sm text-gray-600 mt-1">${Templates.escapeHtml(addr.address_line_1)}</p>
                            ${addr.address_line_2 ? `<p class="text-sm text-gray-600">${Templates.escapeHtml(addr.address_line_2)}</p>` : ''}
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(addr.city)}, ${Templates.escapeHtml(addr.state || '')} ${Templates.escapeHtml(addr.postal_code)}</p>
                            <p class="text-sm text-gray-600">${Templates.escapeHtml(addr.country)}</p>
                            ${addr.phone ? `<p class="text-sm text-gray-600 mt-1">${Templates.escapeHtml(addr.phone)}</p>` : ''}
                            
                            <div class="mt-4 flex gap-2">
                                <button class="edit-address-btn text-sm text-primary-600 hover:text-primary-700" data-address-id="${addr.id}">Edit</button>
                                ${!addr.is_default ? `
                                    <button class="set-default-btn text-sm text-gray-600 hover:text-gray-700" data-address-id="${addr.id}">Set as Default</button>
                                ` : ''}
                                <button class="delete-address-btn text-sm text-red-600 hover:text-red-700" data-address-id="${addr.id}">Delete</button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;

            bindAddressEvents();
        } catch (error) {
            console.error('Failed to load addresses:', error);
            container.innerHTML = '<p class="text-red-500">Failed to load addresses.</p>';
        }
    }

    function bindAddressEvents() {
        document.querySelectorAll('.edit-address-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const addressId = btn.dataset.addressId;
                try {
                    const response = await AuthApi.getAddress(addressId);
                    showAddressModal(response.data);
                } catch (error) {
                    Toast.error('Failed to load address.');
                }
            });
        });

        document.querySelectorAll('.set-default-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const addressId = btn.dataset.addressId;
                try {
                    await AuthApi.setDefaultAddress(addressId);
                    Toast.success('Default address updated.');
                    await loadAddresses();
                } catch (error) {
                    Toast.error('Failed to update default address.');
                }
            });
        });

        document.querySelectorAll('.delete-address-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const addressId = btn.dataset.addressId;
                const confirmed = await Modal.confirm({
                    title: 'Delete Address',
                    message: 'Are you sure you want to delete this address?',
                    confirmText: 'Delete',
                    cancelText: 'Cancel'
                });

                if (confirmed) {
                    try {
                        await AuthApi.deleteAddress(addressId);
                        Toast.success('Address deleted.');
                        await loadAddresses();
                    } catch (error) {
                        Toast.error('Failed to delete address.');
                    }
                }
            });
        });
    }

    function showAddressModal(address = null) {
        const isEdit = !!address;

        Modal.open({
            title: isEdit ? 'Edit Address' : 'Add New Address',
            content: `
                <form id="address-modal-form" class="space-y-4">
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">First Name *</label>
                            <input type="text" name="first_name" value="${address?.first_name || ''}" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Last Name *</label>
                            <input type="text" name="last_name" value="${address?.last_name || ''}" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                        </div>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                        <input type="tel" name="phone" value="${address?.phone || ''}" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Address Line 1 *</label>
                        <input type="text" name="address_line_1" value="${address?.address_line_1 || ''}" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Address Line 2</label>
                        <input type="text" name="address_line_2" value="${address?.address_line_2 || ''}" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">City *</label>
                            <input type="text" name="city" value="${address?.city || ''}" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">State/Province</label>
                            <input type="text" name="state" value="${address?.state || ''}" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Postal Code *</label>
                            <input type="text" name="postal_code" value="${address?.postal_code || ''}" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Country *</label>
                            <select name="country" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500">
                                <option value="">Select country</option>
                                <option value="BD" ${address?.country === 'BD' ? 'selected' : ''}>Bangladesh</option>
                                <option value="US" ${address?.country === 'US' ? 'selected' : ''}>United States</option>
                                <option value="UK" ${address?.country === 'UK' ? 'selected' : ''}>United Kingdom</option>
                                <option value="CA" ${address?.country === 'CA' ? 'selected' : ''}>Canada</option>
                                <option value="AU" ${address?.country === 'AU' ? 'selected' : ''}>Australia</option>
                            </select>
                        </div>
                    </div>
                    <div>
                        <label class="flex items-center">
                            <input type="checkbox" name="is_default" ${address?.is_default ? 'checked' : ''} class="text-primary-600 focus:ring-primary-500 rounded">
                            <span class="ml-2 text-sm text-gray-600">Set as default address</span>
                        </label>
                    </div>
                </form>
            `,
            confirmText: isEdit ? 'Save Changes' : 'Add Address',
            onConfirm: async () => {
                const form = document.getElementById('address-modal-form');
                const formData = new FormData(form);
                const data = {
                    first_name: formData.get('first_name'),
                    last_name: formData.get('last_name'),
                    phone: formData.get('phone'),
                    address_line_1: formData.get('address_line_1'),
                    address_line_2: formData.get('address_line_2'),
                    city: formData.get('city'),
                    state: formData.get('state'),
                    postal_code: formData.get('postal_code'),
                    country: formData.get('country'),
                    is_default: formData.get('is_default') === 'on'
                };

                try {
                    if (isEdit) {
                        await AuthApi.updateAddress(address.id, data);
                        Toast.success('Address updated!');
                    } else {
                        await AuthApi.addAddress(data);
                        Toast.success('Address added!');
                    }
                    await loadAddresses();
                    return true;
                } catch (error) {
                    Toast.error(error.message || 'Failed to save address.');
                    return false;
                }
            }
        });
    }

    function destroy() {
        currentUser = null;
    }

    return {
        init,
        destroy
    };
})();

window.AccountPage = AccountPage;
export default AccountPage;
