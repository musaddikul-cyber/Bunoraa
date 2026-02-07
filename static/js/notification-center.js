/**
 * Bunoraa Notification Center
 * Handles in-app notifications with real-time updates
 */

(function() {
    'use strict';

    // Notification state
    const notificationState = {
        isOpen: false,
        notifications: [],
        unreadCount: 0,
        currentFilter: 'all',
        loading: false,
        socket: null,
        lastFetch: null,
        reconnectAttempts: 0
    };

    // DOM Elements
    let elements = {};

    // Configuration
    const config = {
        apiBaseUrl: '/api/v1/notifications/',
        wsBaseUrl: window.location.protocol === 'https:' ? 'wss://' : 'ws://',
        pollInterval: 60000, // 1 minute
        maxNotifications: 50
    };

    // Notification type icons and colors
    const notificationTypes = {
        order_status: {
            icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"></path></svg>',
            bgColor: 'bg-blue-100 dark:bg-blue-900/30',
            textColor: 'text-blue-600 dark:text-blue-400'
        },
        order_shipped: {
            icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>',
            bgColor: 'bg-green-100 dark:bg-green-900/30',
            textColor: 'text-green-600 dark:text-green-400'
        },
        order_delivered: {
            icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>',
            bgColor: 'bg-green-100 dark:bg-green-900/30',
            textColor: 'text-green-600 dark:text-green-400'
        },
        promotion: {
            icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v13m0-13V6a2 2 0 112 2h-2zm0 0V5.5A2.5 2.5 0 109.5 8H12zm-7 4h14M5 12a2 2 0 110-4h14a2 2 0 110 4M5 12v7a2 2 0 002 2h10a2 2 0 002-2v-7"></path></svg>',
            bgColor: 'bg-amber-100 dark:bg-amber-900/30',
            textColor: 'text-amber-600 dark:text-amber-400'
        },
        price_drop: {
            icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"></path></svg>',
            bgColor: 'bg-red-100 dark:bg-red-900/30',
            textColor: 'text-red-600 dark:text-red-400'
        },
        back_in_stock: {
            icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"></path></svg>',
            bgColor: 'bg-purple-100 dark:bg-purple-900/30',
            textColor: 'text-purple-600 dark:text-purple-400'
        },
        wishlist: {
            icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path></svg>',
            bgColor: 'bg-pink-100 dark:bg-pink-900/30',
            textColor: 'text-pink-600 dark:text-pink-400'
        },
        chat: {
            icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path></svg>',
            bgColor: 'bg-stone-100 dark:bg-stone-800',
            textColor: 'text-stone-600 dark:text-stone-400'
        },
        system: {
            icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>',
            bgColor: 'bg-stone-100 dark:bg-stone-800',
            textColor: 'text-stone-600 dark:text-stone-400'
        }
    };

    /**
     * Initialize notification center
     */
    function init() {
        // Cache DOM elements
        elements = {
            center: document.getElementById('notification-center'),
            bell: document.getElementById('notification-bell'),
            badge: document.getElementById('notification-badge'),
            dot: document.getElementById('notification-dot'),
            dropdown: document.getElementById('notification-dropdown'),
            list: document.getElementById('notification-list'),
            items: document.getElementById('notification-items'),
            loading: document.getElementById('notification-loading'),
            empty: document.getElementById('notification-empty'),
            markAllRead: document.getElementById('notification-mark-all-read'),
            template: document.getElementById('notification-item-template')
        };

        if (!elements.center) {
            console.debug('[Notifications] Center not found in DOM');
            return;
        }

        // Bind events
        bindEvents();

        // Load notifications if authenticated
        if (window.__DJANGO_SESSION_AUTH__) {
            loadNotifications();
            connectWebSocket();
            startPolling();
        }

        console.debug('[Notifications] Initialized');
    }

    /**
     * Bind event listeners
     */
    function bindEvents() {
        // Toggle dropdown
        elements.bell?.addEventListener('click', toggleDropdown);

        // Close on click outside
        document.addEventListener('click', (e) => {
            if (notificationState.isOpen && 
                !elements.center?.contains(e.target)) {
                closeDropdown();
            }
        });

        // Close on escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && notificationState.isOpen) {
                closeDropdown();
            }
        });

        // Mark all as read
        elements.markAllRead?.addEventListener('click', markAllAsRead);

        // Filter buttons
        document.querySelectorAll('.notification-filter-btn').forEach(btn => {
            btn.addEventListener('click', () => filterNotifications(btn.dataset.filter));
        });
    }

    /**
     * Toggle dropdown visibility
     */
    function toggleDropdown(e) {
        e.stopPropagation();
        if (notificationState.isOpen) {
            closeDropdown();
        } else {
            openDropdown();
        }
    }

    /**
     * Open dropdown
     */
    function openDropdown() {
        notificationState.isOpen = true;
        
        elements.dropdown?.classList.remove('hidden');
        requestAnimationFrame(() => {
            elements.dropdown?.classList.remove('opacity-0', 'scale-95');
            elements.dropdown?.classList.add('opacity-100', 'scale-100');
        });
        
        elements.bell?.setAttribute('aria-expanded', 'true');
        
        // Refresh notifications
        loadNotifications();
    }

    /**
     * Close dropdown
     */
    function closeDropdown() {
        notificationState.isOpen = false;
        
        elements.dropdown?.classList.remove('opacity-100', 'scale-100');
        elements.dropdown?.classList.add('opacity-0', 'scale-95');
        
        setTimeout(() => {
            elements.dropdown?.classList.add('hidden');
        }, 200);
        
        elements.bell?.setAttribute('aria-expanded', 'false');
    }

    /**
     * Load notifications from API
     */
    async function loadNotifications() {
        if (notificationState.loading) return;
        
        notificationState.loading = true;
        showLoading(true);

        try {
            const response = await fetch(`${config.apiBaseUrl}?limit=${config.maxNotifications}`);
            if (!response.ok) throw new Error('Failed to load notifications');

            const data = await response.json();
            
            // Handle different API response formats:
            // 1. { success: true, data: [...], meta: { unread_count: N } } - wrapped response
            // 2. { results: [...], count: N } - paginated response
            // 3. [...] - direct array
            let notifications = [];
            let unreadCount = 0;
            
            if (data && typeof data === 'object') {
                if (Array.isArray(data)) {
                    // Direct array response
                    notifications = data;
                    unreadCount = notifications.filter(n => !n.is_read).length;
                } else if (data.success && Array.isArray(data.data)) {
                    // Wrapped response format: { success: true, data: [...], meta: {...} }
                    notifications = data.data;
                    unreadCount = data.meta?.unread_count ?? notifications.filter(n => !n.is_read).length;
                } else if (Array.isArray(data.results)) {
                    // Paginated response format: { results: [...], count: N }
                    notifications = data.results;
                    unreadCount = notifications.filter(n => !n.is_read).length;
                } else if (Array.isArray(data.notifications)) {
                    // Alternative format: { notifications: [...] }
                    notifications = data.notifications;
                    unreadCount = notifications.filter(n => !n.is_read).length;
                }
            }
            
            notificationState.notifications = notifications;
            notificationState.unreadCount = unreadCount;
            notificationState.lastFetch = Date.now();

            renderNotifications();
            updateBadge();
        } catch (error) {
            console.error('[Notifications] Failed to load:', error);
            showEmpty('Failed to load notifications');
        } finally {
            notificationState.loading = false;
            showLoading(false);
        }
    }

    /**
     * Render notifications list
     */
    function renderNotifications() {
        if (!elements.items) return;
        
        const filtered = filterByCurrentFilter(notificationState.notifications);
        
        if (filtered.length === 0) {
            showEmpty();
            return;
        }

        elements.empty?.classList.add('hidden');
        elements.items.innerHTML = '';

        filtered.forEach(notification => {
            const el = createNotificationElement(notification);
            elements.items.appendChild(el);
        });
    }

    /**
     * Create notification element from template
     */
    function createNotificationElement(notification) {
        const type = notificationTypes[notification.notification_type] || notificationTypes.system;
        
        const el = document.createElement('div');
        el.className = `notification-item p-4 border-b border-stone-100 dark:border-stone-800 hover:bg-stone-50 dark:hover:bg-stone-800/50 transition-colors cursor-pointer ${!notification.is_read ? 'bg-amber-50/50 dark:bg-amber-900/10' : ''}`;
        el.dataset.notificationId = notification.id;
        
        el.innerHTML = `
            <div class="flex gap-3">
                <div class="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${type.bgColor} ${type.textColor}">
                    ${type.icon}
                </div>
                <div class="flex-1 min-w-0">
                    <div class="flex items-start justify-between gap-2">
                        <h4 class="text-sm font-medium text-stone-900 dark:text-stone-100 line-clamp-1">${escapeHtml(notification.title)}</h4>
                        <span class="flex-shrink-0 text-xs text-stone-400 dark:text-stone-500">${formatTime(notification.created_at)}</span>
                    </div>
                    <p class="text-sm text-stone-600 dark:text-stone-400 line-clamp-2 mt-0.5">${escapeHtml(notification.message)}</p>
                    ${notification.action_url ? `
                        <a href="${notification.action_url}" class="inline-flex items-center gap-1 mt-2 text-xs text-amber-600 dark:text-amber-400 hover:underline">
                            <span>${notification.action_text || 'View'}</span>
                            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
                        </a>
                    ` : ''}
                </div>
                ${!notification.is_read ? `
                    <div class="flex-shrink-0">
                        <span class="w-2 h-2 bg-amber-500 rounded-full block"></span>
                    </div>
                ` : ''}
            </div>
        `;
        
        // Click handler
        el.addEventListener('click', () => {
            markAsRead(notification.id);
            if (notification.action_url) {
                window.location.href = notification.action_url;
            }
        });
        
        return el;
    }

    /**
     * Filter notifications by type
     */
    function filterNotifications(filter) {
        notificationState.currentFilter = filter;
        
        // Update filter button states
        document.querySelectorAll('.notification-filter-btn').forEach(btn => {
            if (btn.dataset.filter === filter) {
                btn.classList.remove('bg-stone-100', 'text-stone-600', 'dark:bg-stone-800', 'dark:text-stone-400');
                btn.classList.add('bg-amber-100', 'text-amber-700', 'dark:bg-amber-900/30', 'dark:text-amber-400');
            } else {
                btn.classList.add('bg-stone-100', 'text-stone-600', 'dark:bg-stone-800', 'dark:text-stone-400');
                btn.classList.remove('bg-amber-100', 'text-amber-700', 'dark:bg-amber-900/30', 'dark:text-amber-400');
            }
        });
        
        renderNotifications();
    }

    /**
     * Apply current filter to notifications
     */
    function filterByCurrentFilter(notifications) {
        switch (notificationState.currentFilter) {
            case 'unread':
                return notifications.filter(n => !n.is_read);
            case 'orders':
                return notifications.filter(n => 
                    n.notification_type?.startsWith('order') || 
                    n.category === 'order'
                );
            case 'promotions':
                return notifications.filter(n => 
                    ['promotion', 'price_drop', 'back_in_stock'].includes(n.notification_type) ||
                    n.category === 'promotion'
                );
            default:
                return notifications;
        }
    }

    /**
     * Mark notification as read
     */
    async function markAsRead(notificationId) {
        try {
            const response = await fetch(`${config.apiBaseUrl}${notificationId}/read/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCsrfToken()
                }
            });
            
            if (response.ok) {
                const notification = notificationState.notifications.find(n => n.id === notificationId);
                if (notification && !notification.is_read) {
                    notification.is_read = true;
                    notificationState.unreadCount = Math.max(0, notificationState.unreadCount - 1);
                    updateBadge();
                    renderNotifications();
                }
            }
        } catch (error) {
            console.error('[Notifications] Failed to mark as read:', error);
        }
    }

    /**
     * Mark all notifications as read
     */
    async function markAllAsRead() {
        try {
            const response = await fetch(`${config.apiBaseUrl}mark-all-read/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCsrfToken()
                }
            });
            
            if (response.ok) {
                notificationState.notifications.forEach(n => n.is_read = true);
                notificationState.unreadCount = 0;
                updateBadge();
                renderNotifications();
            }
        } catch (error) {
            console.error('[Notifications] Failed to mark all as read:', error);
        }
    }

    /**
     * Update badge count
     */
    function updateBadge() {
        const count = notificationState.unreadCount;
        
        if (count > 0) {
            elements.badge?.classList.remove('hidden');
            if (elements.badge) elements.badge.textContent = count > 99 ? '99+' : count;
            elements.dot?.classList.add('hidden');
        } else {
            elements.badge?.classList.add('hidden');
            elements.dot?.classList.add('hidden');
        }
    }

    /**
     * Show/hide loading state
     */
    function showLoading(show) {
        if (show) {
            elements.loading?.classList.remove('hidden');
            elements.items?.classList.add('hidden');
            elements.empty?.classList.add('hidden');
        } else {
            elements.loading?.classList.add('hidden');
            elements.items?.classList.remove('hidden');
        }
    }

    /**
     * Show empty state
     */
    function showEmpty(message) {
        elements.empty?.classList.remove('hidden');
        elements.items?.classList.add('hidden');
        
        if (message) {
            const text = elements.empty?.querySelector('p');
            if (text) text.textContent = message;
        }
    }

    /**
     * Connect to WebSocket for real-time updates
     */
    function connectWebSocket() {
        if (!window.__DJANGO_SESSION_AUTH__) {
            console.debug('[Notifications] Skipping WebSocket - not authenticated');
            return;
        }
        
        // Check if WebSocket is already connected
        if (notificationState.socket && notificationState.socket.readyState === WebSocket.OPEN) {
            console.debug('[Notifications] WebSocket already connected');
            return;
        }
        
        // Close existing socket if any
        if (notificationState.socket) {
            try {
                notificationState.socket.close();
            } catch (e) {}
            notificationState.socket = null;
        }
        
        const wsUrl = `${config.wsBaseUrl}${window.location.host}/ws/notifications/`;
        
        try {
            notificationState.socket = new WebSocket(wsUrl);
            
            notificationState.socket.onopen = () => {
                console.debug('[Notifications] WebSocket connected');
                notificationState.reconnectAttempts = 0;
            };
            
            notificationState.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                } catch (e) {
                    console.error('[Notifications] Failed to parse WebSocket message:', e);
                }
            };
            
            notificationState.socket.onclose = (event) => {
                console.debug('[Notifications] WebSocket disconnected', event.code);
                notificationState.socket = null;
                
                // Only reconnect if not a clean close and not too many attempts
                if (!event.wasClean && notificationState.reconnectAttempts < 5) {
                    notificationState.reconnectAttempts++;
                    const delay = Math.min(5000 * notificationState.reconnectAttempts, 30000);
                    console.debug(`[Notifications] Reconnecting in ${delay}ms (attempt ${notificationState.reconnectAttempts})`);
                    setTimeout(connectWebSocket, delay);
                } else {
                    console.debug('[Notifications] Max reconnect attempts reached or clean close, falling back to polling');
                }
            };
            
            notificationState.socket.onerror = (error) => {
                console.debug('[Notifications] WebSocket error - will fallback to polling');
            };
        } catch (error) {
            console.debug('[Notifications] WebSocket connection failed - using polling instead');
            notificationState.socket = null;
        }
    }

    /**
     * Handle WebSocket messages
     */
    function handleWebSocketMessage(data) {
        if (data.type === 'notification' && data.notification) {
            // Add new notification
            notificationState.notifications.unshift(data.notification);
            notificationState.unreadCount++;
            
            updateBadge();
            
            if (notificationState.isOpen) {
                renderNotifications();
            }
            
            // Show browser notification if permitted
            showBrowserNotification(data.notification);
            
            // Play sound
            playNotificationSound();
        } else if (data.type === 'connection_established') {
            // Update unread count from server
            if (typeof data.unread_count === 'number') {
                notificationState.unreadCount = data.unread_count;
                updateBadge();
            }
        } else if (data.type === 'pong') {
            // Heartbeat response - connection is alive
        }
    }

    /**
     * Show browser notification
     */
    function showBrowserNotification(notification) {
        if (!notification || !('Notification' in window)) return;
        
        if (Notification.permission === 'granted') {
            try {
                new Notification(notification.title || 'New Notification', {
                    body: notification.message || '',
                    icon: '/static/images/favicon.svg',
                    tag: notification.id || Date.now().toString(),
                    requireInteraction: false
                });
            } catch (e) {
                console.debug('[Notifications] Browser notification failed:', e);
            }
        }
    }

    /**
     * Play notification sound
     */
    function playNotificationSound() {
        try {
            const audio = new Audio('/static/audio/notification.mp3');
            audio.volume = 0.3;
            audio.play().catch(() => {});
        } catch (e) {}
    }

    /**
     * Start polling for updates
     */
    function startPolling() {
        setInterval(() => {
            // Poll if WebSocket is not connected
            if (!notificationState.socket || notificationState.socket.readyState !== WebSocket.OPEN) {
                loadNotifications();
            }
        }, config.pollInterval);
    }

    /**
     * Request notification permission
     */
    function requestPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
    }

    /**
     * Get CSRF token
     */
    function getCsrfToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]')?.value ||
               document.cookie.split('; ').find(row => row.startsWith('csrftoken='))?.split('=')[1] ||
               '';
    }

    /**
     * Format timestamp
     */
    function formatTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h`;
        if (diff < 604800000) return `${Math.floor(diff / 86400000)}d`;
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }

    /**
     * Escape HTML
     */
    function escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Request permission after user interaction
    document.addEventListener('click', requestPermission, { once: true });

    // Expose API
    window.BunoraNotifications = {
        open: openDropdown,
        close: closeDropdown,
        refresh: loadNotifications,
        getUnreadCount: () => notificationState.unreadCount
    };

})();
