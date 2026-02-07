/**
 * Bunoraa Live Chat Widget
 * Handles customer support chat with WebSocket for real-time messaging
 */

(function() {
    'use strict';

    // Chat state
    const chatState = {
        isOpen: false,
        isMinimized: false,
        conversationId: null,
        socket: null,
        reconnectAttempts: 0,
        maxReconnectAttempts: 5,
        unreadCount: 0,
        isTyping: false,
        typingTimeout: null,
        soundEnabled: true,
        messages: [],
        isConnected: false
    };

    // DOM Elements
    let elements = {};

    // Configuration
    const config = {
        wsBaseUrl: window.location.protocol === 'https:' ? 'wss://' : 'ws://',
        apiBaseUrl: '/api/v1/chat/',
        reconnectDelay: 3000,
        typingIndicatorDelay: 3000,
        maxMessageLength: 5000
    };

    /**
     * Initialize the chat widget
     */
    function init() {
        // Cache DOM elements
        elements = {
            widget: document.getElementById('chat-widget'),
            toggleBtn: document.getElementById('chat-toggle-btn'),
            window: document.getElementById('chat-window'),
            closeBtn: document.getElementById('chat-close-btn'),
            minimizeBtn: document.getElementById('chat-minimize-btn'),
            chatIcon: document.getElementById('chat-icon'),
            closeIcon: document.getElementById('chat-close-icon'),
            badge: document.getElementById('chat-badge'),
            statusIndicator: document.getElementById('chat-status-indicator'),
            statusText: document.getElementById('chat-status-text'),
            quickActions: document.getElementById('chat-quick-actions'),
            messages: document.getElementById('chat-messages'),
            typingIndicator: document.getElementById('chat-typing-indicator'),
            form: document.getElementById('chat-form'),
            input: document.getElementById('chat-input'),
            sendBtn: document.getElementById('chat-send-btn'),
            attachBtn: document.getElementById('chat-attach-btn'),
            fileInput: document.getElementById('chat-file-input'),
            soundSend: document.getElementById('chat-sound-send'),
            soundReceive: document.getElementById('chat-sound-receive')
        };

        if (!elements.widget) {
            console.debug('[Chat] Widget not found in DOM');
            return;
        }

        // Bind event listeners
        bindEvents();

        // Load saved state
        loadState();

        // Check for existing conversation
        checkExistingConversation();

        console.debug('[Chat] Widget initialized');
    }

    /**
     * Bind all event listeners
     */
    function bindEvents() {
        // Toggle chat window
        elements.toggleBtn?.addEventListener('click', toggleChat);
        elements.closeBtn?.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            closeChat();
        });
        elements.minimizeBtn?.addEventListener('click', minimizeChat);

        // Form submission
        elements.form?.addEventListener('submit', handleSubmit);

        // Input handling
        elements.input?.addEventListener('input', handleInputChange);
        elements.input?.addEventListener('keydown', handleKeyDown);

        // File attachment
        elements.attachBtn?.addEventListener('click', () => elements.fileInput?.click());
        elements.fileInput?.addEventListener('change', handleFileSelect);

        // Category quick actions
        document.querySelectorAll('.chat-category-btn').forEach(btn => {
            btn.addEventListener('click', () => selectCategory(btn.dataset.category));
        });

        // Listen for custom 'chat:open' event
        document.addEventListener('chat:open', () => {
            if (!chatState.isOpen) {
                openChat();
            }
        });

        // Close on escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && chatState.isOpen) {
                closeChat();
            }
        });

        // Close on click outside
        document.addEventListener('click', (e) => {
            if (chatState.isOpen && 
                !elements.window?.contains(e.target) && 
                !elements.toggleBtn?.contains(e.target)) {
                // Don't close on outside click for better UX
            }
        });
    }

    /**
     * Toggle chat window open/closed
     */
    function toggleChat() {
        if (chatState.isOpen) {
            closeChat();
        } else {
            openChat();
        }
    }

    /**
     * Open the chat window
     */
    function openChat() {
        chatState.isOpen = true;
        chatState.isMinimized = false;
        
        elements.window?.classList.remove('hidden');
        elements.window?.setAttribute('aria-hidden', 'false');
        elements.toggleBtn?.setAttribute('aria-expanded', 'true');
        
        // Swap icons
        elements.chatIcon?.classList.add('hidden');
        elements.closeIcon?.classList.remove('hidden');
        
        // Clear badge
        updateBadge(0);
        
        // Focus input
        setTimeout(() => elements.input?.focus(), 300);
        
        // Connect WebSocket if authenticated and have conversation
        if (window.__DJANGO_SESSION_AUTH__ && chatState.conversationId) {
            connectWebSocket();
        }
        
        // Scroll to bottom
        scrollToBottom();
        
        saveState();
    }

    /**
     * Close the chat window
     */
    function closeChat() {
        chatState.isOpen = false;
        
        elements.window?.classList.add('hidden');
        elements.window?.setAttribute('aria-hidden', 'true');
        elements.toggleBtn?.setAttribute('aria-expanded', 'false');
        
        // Swap icons
        elements.chatIcon?.classList.remove('hidden');
        elements.closeIcon?.classList.add('hidden');
        
        saveState();
    }

    /**
     * Minimize chat (keep connection, hide window)
     */
    function minimizeChat() {
        chatState.isMinimized = true;
        closeChat();
    }

    /**
     * Handle form submission
     */
    async function handleSubmit(e) {
        e.preventDefault();
        
        const message = elements.input?.value?.trim();
        if (!message) return;
        
        if (message.length > config.maxMessageLength) {
            showError('Message is too long');
            return;
        }
        
        // Disable send button
        elements.sendBtn.disabled = true;
        
        try {
            // Add message to UI immediately (optimistic update)
            addMessage({
                content: message,
                is_from_customer: true,
                created_at: new Date().toISOString(),
                status: 'sending'
            });
            
            // Clear input
            elements.input.value = '';
            autoResizeInput();
            
            // Play send sound
            playSound('send');
            
            // Send via WebSocket or API
            if (chatState.socket && chatState.isConnected) {
                sendViaWebSocket(message);
            } else {
                await sendViaApi(message);
            }
        } catch (error) {
            console.error('[Chat] Failed to send message:', error);
            showError('Failed to send message. Please try again.');
        } finally {
            elements.sendBtn.disabled = false;
            elements.input?.focus();
        }
    }

    /**
     * Send message via WebSocket
     */
    function sendViaWebSocket(content) {
        if (!chatState.socket || chatState.socket.readyState !== WebSocket.OPEN) {
            console.warn('[Chat] WebSocket not connected, falling back to API');
            return sendViaApi(content);
        }
        
        chatState.socket.send(JSON.stringify({
            type: 'message',
            content: content,
            conversation_id: chatState.conversationId
        }));
    }

    /**
     * Send message via REST API
     */
    async function sendViaApi(content) {
        // Create conversation if needed
        if (!chatState.conversationId) {
            await createConversation();
        }
        
        // Ensure conversation exists after potential creation
        if (!chatState.conversationId) {
            throw new Error('Failed to establish conversation');
        }
        
        const response = await fetch(`${config.apiBaseUrl}messages/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken()
            },
            body: JSON.stringify({ 
                content: content,
                conversation: chatState.conversationId
            })
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('[Chat] Send message failed:', response.status, errorText);
            throw new Error(`Failed to send message: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update message status
        updateMessageStatus(data.id, 'sent');
        
        return data;
    }

    /**
     * Create a new conversation
     */
    async function createConversation(category = 'general') {
        try {
            const response = await fetch(`${config.apiBaseUrl}conversations/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken()
                },
                body: JSON.stringify({
                    category: category,
                    initial_message: elements.input?.value || ''
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('[Chat] Create conversation failed:', response.status, errorText);
                throw new Error(`Failed to create conversation: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Handle wrapped response format
            const conversation = data.data || data;
            chatState.conversationId = conversation.id;
            
            // Hide quick actions
            elements.quickActions?.classList.add('hidden');
            
            // Connect WebSocket only if we have a valid conversation
            if (chatState.conversationId && window.__DJANGO_SESSION_AUTH__) {
                connectWebSocket();
            }
            
            saveState();
            return conversation;
        } catch (error) {
            console.error('[Chat] Create conversation error:', error);
            throw error;
        }
    }

    /**
     * Check for existing conversation
     */
    async function checkExistingConversation() {
        if (!window.__DJANGO_SESSION_AUTH__) return;
        
        try {
            const response = await fetch(`${config.apiBaseUrl}conversations/active/`);
            if (response.ok) {
                const data = await response.json();
                // Handle wrapped response format
                const conversation = data.data || data;
                if (conversation && conversation.id) {
                    chatState.conversationId = conversation.id;
                    elements.quickActions?.classList.add('hidden');
                    
                    // Use messages from the response if available (ConversationDetailSerializer includes them)
                    if (conversation.messages && Array.isArray(conversation.messages) && conversation.messages.length > 0) {
                        chatState.messages = conversation.messages;
                        renderMessages();
                    } else {
                        // Fallback to loading messages separately
                        await loadMessages();
                    }
                    
                    // Save state after loading conversation
                    saveState();
                }
            } else if (response.status === 404) {
                // No active conversation - this is normal, show quick actions
                console.debug('[Chat] No active conversation found');
                elements.quickActions?.classList.remove('hidden');
            }
        } catch (error) {
            console.debug('[Chat] No active conversation:', error.message);
            elements.quickActions?.classList.remove('hidden');
        }
    }

    /**
     * Load conversation messages
     */
    async function loadMessages() {
        if (!chatState.conversationId) return;
        
        try {
            const response = await fetch(`${config.apiBaseUrl}conversations/${chatState.conversationId}/messages/`);
            if (response.ok) {
                const data = await response.json();
                // Handle different response formats
                let messages = [];
                if (Array.isArray(data)) {
                    messages = data;
                } else if (data.data && Array.isArray(data.data)) {
                    messages = data.data;
                } else if (data.results && Array.isArray(data.results)) {
                    messages = data.results;
                }
                chatState.messages = messages;
                renderMessages();
            }
        } catch (error) {
            console.error('[Chat] Failed to load messages:', error);
        }
    }

    /**
     * Render all messages
     */
    function renderMessages() {
        // Keep welcome message
        const welcomeMsg = elements.messages?.querySelector('.chat-message-bot');
        elements.messages.innerHTML = '';
        if (welcomeMsg) {
            elements.messages.appendChild(welcomeMsg);
        }
        
        chatState.messages.forEach(msg => addMessage(msg, false));
        scrollToBottom();
    }

    /**
     * Add a message to the chat
     */
    function addMessage(message, scroll = true) {
        const isCustomer = message.is_from_customer;
        const isBot = message.is_from_bot;
        
        const messageEl = document.createElement('div');
        messageEl.className = `flex gap-3 ${isCustomer ? 'flex-row-reverse' : ''} chat-message ${isCustomer ? 'chat-message-customer' : 'chat-message-bot'}`;
        messageEl.dataset.messageId = message.id || '';
        
        const time = formatTime(message.created_at);
        const content = escapeHtml(message.content);
        
        if (isCustomer) {
            messageEl.innerHTML = `
                <div class="flex-shrink-0 w-8 h-8 bg-amber-600 rounded-full flex items-center justify-center text-white text-sm font-medium">
                    ${window.__DJANGO_USER__?.first_name?.[0] || 'Y'}
                </div>
                <div class="flex-1 max-w-[80%] text-right">
                    <div class="bg-amber-600 text-white rounded-2xl rounded-tr-none px-4 py-3 inline-block text-left">
                        <p class="text-sm">${content}</p>
                    </div>
                    <span class="text-xs text-stone-400 dark:text-stone-500 mt-1 block">
                        ${time}
                        ${message.status === 'sending' ? '<span class="ml-1">•••</span>' : ''}
                        ${message.status === 'sent' ? '<svg class="w-3 h-3 inline ml-1" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path></svg>' : ''}
                    </span>
                </div>
            `;
        } else {
            messageEl.innerHTML = `
                <div class="flex-shrink-0 w-8 h-8 ${isBot ? 'bg-stone-200 dark:bg-stone-700' : 'bg-amber-100 dark:bg-amber-900/30'} rounded-full flex items-center justify-center">
                    ${isBot ? 
                        '<svg class="w-4 h-4 text-stone-600 dark:text-stone-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path></svg>' :
                        '<svg class="w-4 h-4 text-amber-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>'
                    }
                </div>
                <div class="flex-1 max-w-[80%]">
                    <div class="bg-stone-100 dark:bg-stone-800 rounded-2xl rounded-tl-none px-4 py-3">
                        <p class="text-sm text-stone-700 dark:text-stone-300">${content}</p>
                    </div>
                    <span class="text-xs text-stone-400 dark:text-stone-500 mt-1 block">${time}</span>
                </div>
            `;
        }
        
        elements.messages?.appendChild(messageEl);
        
        if (scroll) {
            scrollToBottom();
        }
    }

    /**
     * Connect to WebSocket
     */
    function connectWebSocket() {
        if (!chatState.conversationId) return;
        if (chatState.socket && chatState.socket.readyState === WebSocket.OPEN) return;
        
        const wsUrl = `${config.wsBaseUrl}${window.location.host}/ws/chat/${chatState.conversationId}/`;
        
        try {
            chatState.socket = new WebSocket(wsUrl);
            
            chatState.socket.onopen = () => {
                console.debug('[Chat] WebSocket connected');
                chatState.isConnected = true;
                chatState.reconnectAttempts = 0;
                updateStatus(true);
            };
            
            chatState.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            chatState.socket.onclose = (event) => {
                console.debug('[Chat] WebSocket disconnected');
                chatState.isConnected = false;
                chatState.socket = null;
                updateStatus(false);
                
                // Only reconnect if not a clean close and not too many attempts
                if (!event.wasClean && chatState.reconnectAttempts < chatState.maxReconnectAttempts) {
                    chatState.reconnectAttempts++;
                    const delay = config.reconnectDelay * chatState.reconnectAttempts;
                    console.debug(`[Chat] Reconnecting in ${delay}ms (attempt ${chatState.reconnectAttempts})`);
                    setTimeout(connectWebSocket, delay);
                } else if (chatState.reconnectAttempts >= chatState.maxReconnectAttempts) {
                    console.debug('[Chat] Max reconnect attempts reached - chat will use HTTP fallback');
                }
            };
            
            chatState.socket.onerror = (error) => {
                // Silently log - don't flood console
                console.debug('[Chat] WebSocket error - will retry or fallback to HTTP');
            };
        } catch (error) {
            console.debug('[Chat] WebSocket connection failed - using HTTP fallback');
            chatState.socket = null;
        }
    }

    /**
     * Handle incoming WebSocket messages
     */
    function handleWebSocketMessage(data) {
        switch (data.type) {
            case 'message':
                if (!data.is_from_customer) {
                    addMessage(data);
                    playSound('receive');
                    
                    if (!chatState.isOpen) {
                        updateBadge(chatState.unreadCount + 1);
                    }
                }
                break;
                
            case 'typing':
                if (!data.is_from_customer) {
                    showTypingIndicator(data.is_typing);
                }
                break;
                
            case 'read':
                // Update read receipts
                break;
                
            case 'agent_joined':
                addSystemMessage(`${data.agent_name || 'An agent'} has joined the chat`);
                break;
                
            case 'agent_left':
                addSystemMessage('Agent has left the chat');
                break;
        }
    }

    /**
     * Handle input changes for auto-resize and typing indicator
     */
    function handleInputChange() {
        autoResizeInput();
        sendTypingIndicator();
    }

    /**
     * Handle keyboard shortcuts
     */
    function handleKeyDown(e) {
        // Submit on Enter (without Shift)
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            elements.form?.dispatchEvent(new Event('submit'));
        }
    }

    /**
     * Auto-resize textarea based on content
     */
    function autoResizeInput() {
        if (!elements.input) return;
        elements.input.style.height = 'auto';
        elements.input.style.height = Math.min(elements.input.scrollHeight, 128) + 'px';
    }

    /**
     * Send typing indicator
     */
    function sendTypingIndicator() {
        if (!chatState.socket || chatState.socket.readyState !== WebSocket.OPEN) return;
        
        if (!chatState.isTyping) {
            chatState.isTyping = true;
            chatState.socket.send(JSON.stringify({
                type: 'typing',
                is_typing: true
            }));
        }
        
        clearTimeout(chatState.typingTimeout);
        chatState.typingTimeout = setTimeout(() => {
            chatState.isTyping = false;
            chatState.socket?.send(JSON.stringify({
                type: 'typing',
                is_typing: false
            }));
        }, config.typingIndicatorDelay);
    }

    /**
     * Show/hide typing indicator
     */
    function showTypingIndicator(show) {
        if (show) {
            elements.typingIndicator?.classList.remove('hidden');
            scrollToBottom();
        } else {
            elements.typingIndicator?.classList.add('hidden');
        }
    }

    /**
     * Handle file selection
     */
    async function handleFileSelect(e) {
        const file = e.target.files?.[0];
        if (!file) return;
        
        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            showError('File is too large. Maximum size is 10MB.');
            return;
        }
        
        // Create conversation if needed
        if (!chatState.conversationId) {
            await createConversation();
        }
        
        // Upload file
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`${config.apiBaseUrl}conversations/${chatState.conversationId}/attachments/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCsrfToken()
                },
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Upload failed');
            }
            
            const data = await response.json();
            addMessage({
                content: `📎 ${file.name}`,
                is_from_customer: true,
                created_at: new Date().toISOString(),
                attachment: data
            });
        } catch (error) {
            console.error('[Chat] File upload failed:', error);
            showError('Failed to upload file. Please try again.');
        }
        
        // Reset input
        e.target.value = '';
    }

    /**
     * Select a category for new conversation
     */
    async function selectCategory(category) {
        // Check if user is authenticated
        if (!window.__DJANGO_SESSION_AUTH__) {
            showError('Please sign in to start a chat');
            // Optionally redirect to login
            const loginUrl = window.__ROUTES__?.accountsLogin || '/account/login/';
            if (confirm('You need to sign in to start a chat. Would you like to sign in now?')) {
                window.location.href = loginUrl + '?next=' + encodeURIComponent(window.location.pathname);
            }
            return;
        }
        
        try {
            // Show loading state
            elements.quickActions?.classList.add('opacity-50', 'pointer-events-none');
            
            await createConversation(category);
            
            // Send initial message based on category
            const messages = {
                order: "I have a question about my order.",
                product: "I'd like to know more about a product.",
                shipping: "I have a shipping question.",
                returns: "I need help with a return or refund.",
                general: "Hi, I have a general question."
            };
            
            addMessage({
                content: messages[category] || messages.general,
                is_from_customer: true,
                created_at: new Date().toISOString()
            });
            
            await sendViaApi(messages[category] || messages.general);
        } catch (error) {
            console.error('[Chat] Failed to start conversation:', error);
            showError('Unable to start chat. Please try again later.');
            elements.quickActions?.classList.remove('hidden', 'opacity-50', 'pointer-events-none');
        } finally {
            elements.quickActions?.classList.remove('opacity-50', 'pointer-events-none');
        }
    }

    /**
     * Add system message
     */
    function addSystemMessage(text) {
        const messageEl = document.createElement('div');
        messageEl.className = 'text-center py-2';
        messageEl.innerHTML = `<span class="text-xs text-stone-400 dark:text-stone-500 bg-stone-100 dark:bg-stone-800 px-3 py-1 rounded-full">${escapeHtml(text)}</span>`;
        elements.messages?.appendChild(messageEl);
        scrollToBottom();
    }

    /**
     * Update message status
     */
    function updateMessageStatus(messageId, status) {
        const messageEl = elements.messages?.querySelector(`[data-message-id="${messageId}"]`);
        if (messageEl) {
            // Update status indicator
        }
    }

    /**
     * Update connection status
     */
    function updateStatus(online) {
        if (online) {
            elements.statusIndicator?.classList.remove('bg-gray-400');
            elements.statusIndicator?.classList.add('bg-green-400');
            if (elements.statusText) elements.statusText.textContent = "We're online";
        } else {
            elements.statusIndicator?.classList.remove('bg-green-400');
            elements.statusIndicator?.classList.add('bg-gray-400');
            if (elements.statusText) elements.statusText.textContent = 'Reconnecting...';
        }
    }

    /**
     * Update unread badge
     */
    function updateBadge(count) {
        chatState.unreadCount = count;
        if (count > 0) {
            elements.badge?.classList.remove('hidden');
            if (elements.badge) elements.badge.textContent = count > 9 ? '9+' : count;
        } else {
            elements.badge?.classList.add('hidden');
        }
    }

    /**
     * Play sound effect
     */
    function playSound(type) {
        if (!chatState.soundEnabled) return;
        
        const audio = type === 'send' ? elements.soundSend : elements.soundReceive;
        if (audio) {
            audio.currentTime = 0;
            audio.volume = 0.3;
            audio.play().catch(() => {});
        }
    }

    /**
     * Show error message
     */
    function showError(message) {
        window.Toast?.error?.(message) || alert(message);
    }

    /**
     * Scroll messages to bottom
     */
    function scrollToBottom() {
        if (elements.messages) {
            elements.messages.scrollTop = elements.messages.scrollHeight;
        }
    }

    /**
     * Save state to localStorage
     */
    function saveState() {
        try {
            localStorage.setItem('bunoraa_chat', JSON.stringify({
                conversationId: chatState.conversationId,
                isMinimized: chatState.isMinimized
            }));
        } catch (e) {}
    }

    /**
     * Load state from localStorage
     */
    function loadState() {
        try {
            const saved = localStorage.getItem('bunoraa_chat');
            if (saved) {
                const data = JSON.parse(saved);
                chatState.conversationId = data.conversationId;
                chatState.isMinimized = data.isMinimized;
            }
        } catch (e) {}
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
        if (!timestamp) return 'Just now';
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
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

    // Expose API for external use
    window.BunoraChat = {
        open: openChat,
        close: closeChat,
        toggle: toggleChat,
        sendMessage: (msg) => {
            if (elements.input) {
                elements.input.value = msg;
                elements.form?.dispatchEvent(new Event('submit'));
            }
        }
    };

})();
