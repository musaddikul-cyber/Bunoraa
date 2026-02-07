/**
 * Bunoraa WebSocket Client
 * Handles real-time updates for notifications, cart, and live features
 */

(function(window, document) {
    'use strict';

    class BunoraWebSocket {
        constructor(options = {}) {
            this.options = {
                reconnectAttempts: 5,
                reconnectInterval: 3000,
                heartbeatInterval: 30000,
                ...options
            };
            
            this.connections = new Map();
            this.reconnectAttempts = new Map();
            this.heartbeatTimers = new Map();
            this.messageHandlers = new Map();
        }

        /**
         * Connect to a WebSocket channel
         */
        connect(channel, handlers = {}) {
            if (this.connections.has(channel)) {
                console.warn(`Already connected to ${channel}`);
                return this.connections.get(channel);
            }

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${channel}/`;

            const ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log(`Connected to ${channel}`);
                this.reconnectAttempts.set(channel, 0);
                this.startHeartbeat(channel, ws);
                
                if (handlers.onConnect) {
                    handlers.onConnect(ws);
                }
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(channel, data, handlers);
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            ws.onerror = (error) => {
                console.error(`WebSocket error on ${channel}:`, error);
                if (handlers.onError) {
                    handlers.onError(error);
                }
            };

            ws.onclose = (event) => {
                console.log(`Disconnected from ${channel}`, event.code, event.reason);
                this.stopHeartbeat(channel);
                this.connections.delete(channel);
                
                if (handlers.onDisconnect) {
                    handlers.onDisconnect(event);
                }

                // Attempt reconnection
                if (!event.wasClean) {
                    this.attemptReconnect(channel, handlers);
                }
            };

            this.connections.set(channel, ws);
            this.messageHandlers.set(channel, handlers);
            
            return ws;
        }

        /**
         * Disconnect from a channel
         */
        disconnect(channel) {
            const ws = this.connections.get(channel);
            if (ws) {
                ws.close(1000, 'Client disconnect');
                this.connections.delete(channel);
                this.stopHeartbeat(channel);
            }
        }

        /**
         * Send a message to a channel
         */
        send(channel, type, data = {}) {
            const ws = this.connections.get(channel);
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type, ...data }));
            } else {
                console.warn(`Cannot send to ${channel}: not connected`);
            }
        }

        /**
         * Handle incoming messages
         */
        handleMessage(channel, data, handlers) {
            const { type, ...payload } = data;

            // Call specific handler if exists
            if (handlers[`on${this.capitalize(type)}`]) {
                handlers[`on${this.capitalize(type)}`](payload);
            }

            // Call generic message handler
            if (handlers.onMessage) {
                handlers.onMessage(data);
            }

            // Dispatch custom event
            document.dispatchEvent(new CustomEvent(`bunoraa:ws:${channel}`, {
                detail: data
            }));
        }

        /**
         * Attempt reconnection
         */
        attemptReconnect(channel, handlers) {
            const attempts = this.reconnectAttempts.get(channel) || 0;
            
            if (attempts >= this.options.reconnectAttempts) {
                console.error(`Max reconnection attempts reached for ${channel}`);
                return;
            }

            const delay = this.options.reconnectInterval * Math.pow(2, attempts);
            console.log(`Reconnecting to ${channel} in ${delay}ms (attempt ${attempts + 1})`);

            setTimeout(() => {
                this.reconnectAttempts.set(channel, attempts + 1);
                this.connect(channel, handlers);
            }, delay);
        }

        /**
         * Start heartbeat for connection
         */
        startHeartbeat(channel, ws) {
            const timer = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, this.options.heartbeatInterval);
            
            this.heartbeatTimers.set(channel, timer);
        }

        /**
         * Stop heartbeat
         */
        stopHeartbeat(channel) {
            const timer = this.heartbeatTimers.get(channel);
            if (timer) {
                clearInterval(timer);
                this.heartbeatTimers.delete(channel);
            }
        }

        /**
         * Capitalize first letter
         */
        capitalize(str) {
            return str.charAt(0).toUpperCase() + str.slice(1);
        }

        /**
         * Check if connected to a channel
         */
        isConnected(channel) {
            const ws = this.connections.get(channel);
            return ws && ws.readyState === WebSocket.OPEN;
        }

        /**
         * Get all active connections
         */
        getActiveConnections() {
            return Array.from(this.connections.keys());
        }
    }

    // ============================================
    // Notification Handler
    // ============================================

    const NotificationChannel = {
        ws: null,
        unreadCount: 0,

        init() {
            if (!window.BunoraWS) return;

            this.ws = window.BunoraWS.connect('notifications', {
                onConnect: () => {
                    console.log('Notifications connected');
                    this.requestUnreadCount();
                },
                onNotification: (data) => {
                    this.handleNotification(data);
                },
                onUnreadCount: (data) => {
                    this.updateUnreadCount(data.count);
                }
            });
        },

        requestUnreadCount() {
            window.BunoraWS.send('notifications', 'get_unread_count');
        },

        handleNotification(data) {
            // Show toast notification
            this.showToast(data);

            // Update unread count
            this.unreadCount++;
            this.updateUnreadCount(this.unreadCount);

            // Add to notification list if visible
            this.addToList(data);

            // Request browser notification permission
            if ('Notification' in window && Notification.permission === 'granted') {
                new Notification(data.title, {
                    body: data.message,
                    icon: '/static/images/notification-icon.png',
                    badge: '/static/images/badge-icon.png',
                    tag: data.id || 'bunoraa-notification'
                });
            }
        },

        showToast(data) {
            const toast = document.createElement('div');
            toast.className = 'notification-toast';
            toast.innerHTML = `
                <div class="notification-toast-content">
                    ${data.icon ? `<span class="notification-icon">${data.icon}</span>` : ''}
                    <div class="notification-text">
                        <strong>${data.title}</strong>
                        <p>${data.message}</p>
                    </div>
                    <button class="notification-close" aria-label="Dismiss">×</button>
                </div>
            `;

            // Add click handler
            if (data.url) {
                toast.querySelector('.notification-text').addEventListener('click', () => {
                    window.location.href = data.url;
                });
                toast.style.cursor = 'pointer';
            }

            toast.querySelector('.notification-close').addEventListener('click', () => {
                toast.classList.remove('show');
                setTimeout(() => toast.remove(), 300);
            });

            document.body.appendChild(toast);
            
            // Trigger animation
            requestAnimationFrame(() => {
                toast.classList.add('show');
            });

            // Auto remove
            setTimeout(() => {
                toast.classList.remove('show');
                setTimeout(() => toast.remove(), 300);
            }, 5000);
        },

        updateUnreadCount(count) {
            this.unreadCount = count;
            document.querySelectorAll('[data-notification-count]').forEach(el => {
                el.textContent = count;
                el.hidden = count === 0;
            });
        },

        addToList(notification) {
            const list = document.querySelector('[data-notification-list]');
            if (!list) return;

            const item = document.createElement('div');
            item.className = 'notification-item unread';
            item.innerHTML = `
                <div class="notification-item-content">
                    ${notification.icon ? `<span class="notification-icon">${notification.icon}</span>` : ''}
                    <div class="notification-text">
                        <strong>${notification.title}</strong>
                        <p>${notification.message}</p>
                        <time>${new Date().toLocaleString('bn-BD')}</time>
                    </div>
                </div>
            `;

            list.insertBefore(item, list.firstChild);
        },

        markAsRead(notificationId) {
            window.BunoraWS.send('notifications', 'mark_read', { id: notificationId });
        },

        markAllAsRead() {
            window.BunoraWS.send('notifications', 'mark_all_read');
            this.updateUnreadCount(0);
            document.querySelectorAll('.notification-item.unread').forEach(el => {
                el.classList.remove('unread');
            });
        }
    };

    // ============================================
    // Live Cart Sync Handler
    // ============================================

    const LiveCartChannel = {
        init() {
            if (!window.BunoraWS) return;

            window.BunoraWS.connect('cart', {
                onConnect: () => {
                    console.log('Live cart connected');
                },
                onCartUpdate: (data) => {
                    this.handleCartUpdate(data);
                },
                onItemAdded: (data) => {
                    this.handleItemAdded(data);
                },
                onItemRemoved: (data) => {
                    this.handleItemRemoved(data);
                }
            });
        },

        handleCartUpdate(data) {
            // Sync with Bunoraa Cart module
            if (window.Bunoraa && window.Bunoraa.Cart) {
                window.Bunoraa.Cart.items = data.items || [];
                window.Bunoraa.Cart.count = data.count || 0;
                window.Bunoraa.Cart.updateUI();
            }
        },

        handleItemAdded(data) {
            if (window.Bunoraa && window.Bunoraa.Cart) {
                window.Bunoraa.Cart.showNotification(`${data.product_name} added to cart`);
            }
        },

        handleItemRemoved(data) {
            if (window.Bunoraa && window.Bunoraa.Cart) {
                window.Bunoraa.Cart.showNotification('Item removed from cart');
            }
        }
    };

    // ============================================
    // Live Search Handler
    // ============================================

    const LiveSearchChannel = {
        searchInput: null,
        resultsContainer: null,
        searchTimeout: null,

        init() {
            if (!window.BunoraWS) return;

            this.searchInput = document.querySelector('[data-ws-search]');
            if (!this.searchInput) return;

            this.resultsContainer = document.querySelector('[data-ws-search-results]');

            window.BunoraWS.connect('search', {
                onConnect: () => {
                    console.log('Live search connected');
                },
                onResults: (data) => {
                    this.handleResults(data);
                },
                onSuggestions: (data) => {
                    this.handleSuggestions(data);
                }
            });

            this.bindEvents();
        },

        bindEvents() {
            this.searchInput.addEventListener('input', (e) => {
                clearTimeout(this.searchTimeout);
                const query = e.target.value.trim();

                if (query.length >= 2) {
                    this.searchTimeout = setTimeout(() => {
                        window.BunoraWS.send('search', 'search', { query });
                    }, 150);
                } else {
                    this.hideResults();
                }
            });
        },

        handleResults(data) {
            if (!this.resultsContainer) return;

            if (!data.results || data.results.length === 0) {
                this.resultsContainer.innerHTML = '<p class="no-results">No results found</p>';
            } else {
                this.resultsContainer.innerHTML = data.results.map(item => `
                    <a href="${item.url}" class="search-result">
                        ${item.image ? `<img src="${item.image}" alt="" loading="lazy">` : ''}
                        <span>${item.name}</span>
                    </a>
                `).join('');
            }

            this.showResults();
        },

        handleSuggestions(data) {
            // Handle auto-complete suggestions
            const datalist = document.querySelector('[data-search-suggestions]');
            if (datalist && data.suggestions) {
                datalist.innerHTML = data.suggestions.map(s => 
                    `<option value="${s}">`
                ).join('');
            }
        },

        showResults() {
            if (this.resultsContainer) {
                this.resultsContainer.hidden = false;
            }
        },

        hideResults() {
            if (this.resultsContainer) {
                this.resultsContainer.hidden = true;
            }
        }
    };

    // ============================================
    // Price Update Handler (for dynamic pricing)
    // ============================================

    const PriceChannel = {
        init() {
            if (!window.BunoraWS) return;

            window.BunoraWS.connect('prices', {
                onConnect: () => {
                    console.log('Price updates connected');
                },
                onPriceUpdate: (data) => {
                    this.handlePriceUpdate(data);
                },
                onStockUpdate: (data) => {
                    this.handleStockUpdate(data);
                }
            });
        },

        handlePriceUpdate(data) {
            // Update all price displays for this product
            document.querySelectorAll(`[data-product-price="${data.product_id}"]`).forEach(el => {
                const oldPrice = el.textContent;
                const newPrice = this.formatPrice(data.price);
                
                if (oldPrice !== newPrice) {
                    el.classList.add('price-changed');
                    el.textContent = newPrice;
                    
                    setTimeout(() => {
                        el.classList.remove('price-changed');
                    }, 2000);
                }
            });
        },

        handleStockUpdate(data) {
            document.querySelectorAll(`[data-product-stock="${data.product_id}"]`).forEach(el => {
                el.textContent = data.stock;
                
                // Update add to cart button
                const addBtn = document.querySelector(`[data-add-to-cart="${data.product_id}"]`);
                if (addBtn) {
                    addBtn.disabled = data.stock <= 0;
                    addBtn.textContent = data.stock <= 0 ? 'Out of Stock' : 'Add to Cart';
                }
            });
        },

        formatPrice(price) {
            return window.Bunoraa?.formatCurrency?.(price) || `${window.BUNORAA_CURRENCY?.symbol || '৳'}${price}`;
        }
    };

    // ============================================
    // Initialize
    // ============================================

    function init() {
        // Create global WebSocket instance
        window.BunoraWS = new BunoraWebSocket();

        // Initialize channels
        NotificationChannel.init();
        LiveCartChannel.init();
        LiveSearchChannel.init();
        PriceChannel.init();

        // Expose for external use
        window.BunoraNotifications = NotificationChannel;
        window.BunoraLiveCart = LiveCartChannel;
        window.BunoraLiveSearch = LiveSearchChannel;
        window.BunoraPrices = PriceChannel;

        // Request notification permission
        if ('Notification' in window && Notification.permission === 'default') {
            // Wait for user interaction
            document.addEventListener('click', function requestPermission() {
                Notification.requestPermission();
                document.removeEventListener('click', requestPermission);
            }, { once: true });
        }
    }

    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})(window, document);
