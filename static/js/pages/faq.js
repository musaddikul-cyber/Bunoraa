/**
 * FAQ Page - Enhanced with Advanced Features
 * @module pages/faq
 */

const FAQPage = (function() {
    'use strict';

    let categories = [];
    let allFAQs = [];

    async function init() {
        // Check if FAQs are already rendered server-side
        const existingFAQList = document.getElementById('faq-list');
        if (existingFAQList && existingFAQList.querySelector('.faq-item')) {
            // Server-rendered content exists - just bind events
            bindServerRenderedContent();
        } else {
            // Load dynamically
            await loadFAQs();
        }
        initSearch();
        initEnhancedFeatures();
    }

    // ============================================
    // ENHANCED FEATURES INITIALIZATION
    // ============================================
    function initEnhancedFeatures() {
        initVoiceSearch();
        initFAQRating();
        initContactPromo();
        initPopularQuestions();
        initShareQuestion();
        initKeyboardNavigation();
        trackFAQAnalytics();
    }

    // ============================================
    // ENHANCED FEATURE: Voice Search
    // ============================================
    function initVoiceSearch() {
        const searchContainer = document.querySelector('.faq-search-container');
        if (!searchContainer || !('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) return;

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = 'en-US';

        // Add voice button
        const voiceBtn = document.createElement('button');
        voiceBtn.id = 'faq-voice-search';
        voiceBtn.type = 'button';
        voiceBtn.className = 'absolute right-12 top-1/2 -translate-y-1/2 p-2 text-stone-400 hover:text-primary-600 dark:hover:text-amber-400 transition-colors';
        voiceBtn.innerHTML = `
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
        `;

        const searchInput = document.getElementById('faq-search');
        if (searchInput && searchInput.parentElement) {
            searchInput.parentElement.style.position = 'relative';
            searchInput.parentElement.appendChild(voiceBtn);
        }

        let isListening = false;

        voiceBtn.addEventListener('click', () => {
            if (isListening) {
                recognition.stop();
            } else {
                recognition.start();
                voiceBtn.classList.add('text-red-500', 'animate-pulse');
            }
            isListening = !isListening;
        });

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (searchInput) {
                searchInput.value = transcript;
                searchInput.dispatchEvent(new Event('input'));
            }
            voiceBtn.classList.remove('text-red-500', 'animate-pulse');
            isListening = false;
        };

        recognition.onerror = () => {
            voiceBtn.classList.remove('text-red-500', 'animate-pulse');
            isListening = false;
        };
    }

    // ============================================
    // ENHANCED FEATURE: FAQ Rating (Was this helpful?)
    // ============================================
    function initFAQRating() {
        // Add rating buttons to each FAQ answer
        document.querySelectorAll('.faq-content, .accordion-content, .faq-answer').forEach(content => {
            if (content.querySelector('.faq-rating')) return;

            const faqItem = content.closest('.faq-item') || content.closest('[data-accordion]');
            const questionId = faqItem?.dataset.id || Math.random().toString(36).substr(2, 9);

            const ratingHtml = `
                <div class="faq-rating mt-4 pt-4 border-t border-stone-200 dark:border-stone-700 flex items-center justify-between">
                    <span class="text-sm text-stone-500 dark:text-stone-400">Was this answer helpful?</span>
                    <div class="flex gap-2">
                        <button class="faq-rate-btn px-3 py-1 text-sm border border-stone-200 dark:border-stone-600 rounded-lg hover:bg-emerald-50 dark:hover:bg-emerald-900/20 hover:border-emerald-500 hover:text-emerald-600 dark:hover:text-emerald-400 transition-all" data-helpful="yes" data-question="${questionId}">
                            <span class="flex items-center gap-1">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"/></svg>
                                Yes
                            </span>
                        </button>
                        <button class="faq-rate-btn px-3 py-1 text-sm border border-stone-200 dark:border-stone-600 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 hover:border-red-500 hover:text-red-600 dark:hover:text-red-400 transition-all" data-helpful="no" data-question="${questionId}">
                            <span class="flex items-center gap-1">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5"/></svg>
                                No
                            </span>
                        </button>
                    </div>
                </div>
            `;

            content.insertAdjacentHTML('beforeend', ratingHtml);
        });

        // Bind rating clicks
        document.addEventListener('click', (e) => {
            const rateBtn = e.target.closest('.faq-rate-btn');
            if (!rateBtn) return;

            const helpful = rateBtn.dataset.helpful === 'yes';
            const questionId = rateBtn.dataset.question;
            const container = rateBtn.closest('.faq-rating');

            // Save to localStorage
            const ratings = JSON.parse(localStorage.getItem('faqRatings') || '{}');
            ratings[questionId] = helpful;
            localStorage.setItem('faqRatings', JSON.stringify(ratings));

            // Show thank you
            container.innerHTML = `
                <div class="flex items-center gap-2 text-sm ${helpful ? 'text-emerald-600 dark:text-emerald-400' : 'text-stone-500 dark:text-stone-400'}">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                    <span>Thanks for your feedback!</span>
                </div>
            `;

            // Track analytics
            if (typeof analytics !== 'undefined') {
                analytics.track('faq_rated', { questionId, helpful });
            }
        });
    }

    // ============================================
    // ENHANCED FEATURE: Contact CTA Promo
    // ============================================
    function initContactPromo() {
        const container = document.getElementById('faq-contact-promo');
        if (!container) return;

        container.innerHTML = `
            <div class="bg-gradient-to-br from-stone-900 to-stone-800 dark:from-stone-800 dark:to-stone-900 text-white rounded-2xl p-6 md:p-8">
                <div class="flex flex-col md:flex-row items-center gap-6">
                    <div class="w-16 h-16 bg-primary-600 dark:bg-amber-600 rounded-2xl flex items-center justify-center flex-shrink-0">
                        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                    </div>
                    <div class="text-center md:text-left flex-1">
                        <h3 class="text-xl font-bold mb-2">Can't Find What You're Looking For?</h3>
                        <p class="text-stone-300 mb-4">Our support team is here to help. Get personalized assistance for your questions.</p>
                        <div class="flex flex-col sm:flex-row gap-3 justify-center md:justify-start">
                            <a href="/contact/" class="inline-flex items-center justify-center gap-2 px-6 py-3 bg-white text-stone-900 font-semibold rounded-xl hover:bg-stone-100 transition-colors">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                                Contact Support
                            </a>
                            <button id="open-chat-faq" class="inline-flex items-center justify-center gap-2 px-6 py-3 bg-primary-600 dark:bg-amber-600 text-white font-semibold rounded-xl hover:bg-primary-700 dark:hover:bg-amber-700 transition-colors">
                                <span class="relative flex h-2 w-2">
                                    <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
                                    <span class="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
                                </span>
                                Live Chat
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.getElementById('open-chat-faq')?.addEventListener('click', () => {
            document.dispatchEvent(new CustomEvent('chat:open'));
        });
    }

    // ============================================
    // ENHANCED FEATURE: Popular Questions
    // ============================================
    function initPopularQuestions() {
        const container = document.getElementById('popular-questions');
        if (!container) return;

        // Get popular questions from localStorage ratings
        const ratings = JSON.parse(localStorage.getItem('faqRatings') || '{}');
        const popularIds = Object.entries(ratings)
            .filter(([_, helpful]) => helpful)
            .slice(0, 5)
            .map(([id]) => id);

        // Find matching FAQ items
        const popularQuestions = [];
        document.querySelectorAll('.faq-item, [data-accordion]').forEach(item => {
            const id = item.dataset.id;
            if (popularIds.includes(id) || popularQuestions.length < 3) {
                const questionText = item.querySelector('button span, .accordion-toggle span')?.textContent?.trim();
                if (questionText) {
                    popularQuestions.push({ id, question: questionText, element: item });
                }
            }
        });

        if (popularQuestions.length === 0) return;

        container.innerHTML = `
            <div class="bg-primary-50 dark:bg-amber-900/20 rounded-2xl p-6">
                <h3 class="font-semibold text-stone-900 dark:text-white mb-4 flex items-center gap-2">
                    <svg class="w-5 h-5 text-primary-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>
                    Most Helpful Questions
                </h3>
                <ul class="space-y-2">
                    ${popularQuestions.slice(0, 5).map(q => `
                        <li>
                            <button class="popular-q-link text-left text-primary-600 dark:text-amber-400 hover:underline text-sm" data-target="${q.id}">
                                ${Templates.escapeHtml(q.question)}
                            </button>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;

        // Bind clicks to scroll and expand
        container.querySelectorAll('.popular-q-link').forEach(link => {
            link.addEventListener('click', () => {
                const targetId = link.dataset.target;
                const targetItem = document.querySelector(`[data-id="${targetId}"], .faq-item`);
                if (targetItem) {
                    targetItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    const trigger = targetItem.querySelector('.faq-trigger, .accordion-toggle');
                    if (trigger) trigger.click();
                }
            });
        });
    }

    // ============================================
    // ENHANCED FEATURE: Share Question
    // ============================================
    function initShareQuestion() {
        document.querySelectorAll('.faq-content, .accordion-content, .faq-answer').forEach(content => {
            if (content.querySelector('.faq-share')) return;

            const faqItem = content.closest('.faq-item') || content.closest('[data-accordion]');
            const questionText = faqItem?.querySelector('button span, .accordion-toggle span')?.textContent?.trim();
            
            if (!questionText) return;

            const shareHtml = `
                <div class="faq-share flex items-center gap-2 mt-3">
                    <span class="text-xs text-stone-400 dark:text-stone-500">Share:</span>
                    <button class="faq-share-btn p-1.5 hover:bg-stone-100 dark:hover:bg-stone-700 rounded transition-colors" data-platform="copy" title="Copy link">
                        <svg class="w-4 h-4 text-stone-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
                    </button>
                    <a href="https://twitter.com/intent/tweet?text=${encodeURIComponent('Q: ' + questionText)}&url=${encodeURIComponent(window.location.href)}" target="_blank" rel="noopener noreferrer" class="faq-share-btn p-1.5 hover:bg-stone-100 dark:hover:bg-stone-700 rounded transition-colors" title="Share on Twitter">
                        <svg class="w-4 h-4 text-stone-400" fill="currentColor" viewBox="0 0 24 24"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                    </a>
                    <a href="mailto:?subject=${encodeURIComponent('FAQ: ' + questionText)}&body=${encodeURIComponent(window.location.href)}" class="faq-share-btn p-1.5 hover:bg-stone-100 dark:hover:bg-stone-700 rounded transition-colors" title="Share via email">
                        <svg class="w-4 h-4 text-stone-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                    </a>
                </div>
            `;

            content.insertAdjacentHTML('beforeend', shareHtml);
        });

        // Copy link handler
        document.addEventListener('click', (e) => {
            const copyBtn = e.target.closest('.faq-share-btn[data-platform="copy"]');
            if (!copyBtn) return;

            navigator.clipboard.writeText(window.location.href).then(() => {
                const originalHtml = copyBtn.innerHTML;
                copyBtn.innerHTML = `<svg class="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>`;
                setTimeout(() => { copyBtn.innerHTML = originalHtml; }, 2000);
            });
        });
    }

    // ============================================
    // ENHANCED FEATURE: Keyboard Navigation
    // ============================================
    function initKeyboardNavigation() {
        const faqItems = document.querySelectorAll('.faq-item, [data-accordion]');
        let currentIndex = -1;

        document.addEventListener('keydown', (e) => {
            // Only if we're in FAQ context
            if (!document.querySelector('#faq-container, #faq-list')) return;

            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                
                if (e.key === 'ArrowDown') {
                    currentIndex = Math.min(currentIndex + 1, faqItems.length - 1);
                } else {
                    currentIndex = Math.max(currentIndex - 1, 0);
                }

                const item = faqItems[currentIndex];
                if (item) {
                    item.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    const trigger = item.querySelector('.faq-trigger, .accordion-toggle, button');
                    if (trigger) trigger.focus();
                }
            }

            if (e.key === 'Enter' && currentIndex >= 0) {
                const item = faqItems[currentIndex];
                const trigger = item?.querySelector('.faq-trigger, .accordion-toggle');
                if (trigger) trigger.click();
            }

            // '/' to focus search
            if (e.key === '/' && document.activeElement?.tagName !== 'INPUT') {
                e.preventDefault();
                document.getElementById('faq-search')?.focus();
            }
        });
    }

    // ============================================
    // ENHANCED FEATURE: FAQ Analytics
    // ============================================
    function trackFAQAnalytics() {
        // Track which questions are viewed
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const item = entry.target;
                    const questionText = item.querySelector('button span, .accordion-toggle span')?.textContent?.trim();
                    
                    // Track view
                    const views = JSON.parse(localStorage.getItem('faqViews') || '{}');
                    const id = item.dataset.id || questionText?.substring(0, 30);
                    if (id) {
                        views[id] = (views[id] || 0) + 1;
                        localStorage.setItem('faqViews', JSON.stringify(views));
                    }
                }
            });
        }, { threshold: 0.5 });

        document.querySelectorAll('.faq-item, [data-accordion]').forEach(item => {
            observer.observe(item);
        });

        // Track accordion opens
        document.addEventListener('click', (e) => {
            const trigger = e.target.closest('.faq-trigger, .accordion-toggle');
            if (trigger) {
                const item = trigger.closest('.faq-item, [data-accordion]');
                const questionText = trigger.querySelector('span')?.textContent?.trim();
                
                if (typeof analytics !== 'undefined') {
                    analytics.track('faq_opened', { question: questionText?.substring(0, 100) });
                }
            }
        });
    }

    function bindServerRenderedContent() {
        // Bind category tabs
        const categoryBtns = document.querySelectorAll('.category-tab');
        const categoryGroups = document.querySelectorAll('.faq-category');

        categoryBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Update active state
                categoryBtns.forEach(b => {
                    b.classList.remove('bg-primary-600', 'dark:bg-amber-600', 'text-white');
                    b.classList.add('bg-stone-100', 'dark:bg-stone-800', 'text-stone-700', 'dark:text-stone-300');
                });
                btn.classList.add('bg-primary-600', 'dark:bg-amber-600', 'text-white');
                btn.classList.remove('bg-stone-100', 'dark:bg-stone-800', 'text-stone-700', 'dark:text-stone-300');

                const category = btn.dataset.category;

                // Show/hide categories
                if (category === 'all') {
                    categoryGroups.forEach(cat => cat.classList.remove('hidden'));
                } else {
                    categoryGroups.forEach(cat => {
                        cat.classList.toggle('hidden', cat.dataset.category !== category);
                    });
                }

                // Clear search and show all items
                const searchInput = document.getElementById('faq-search');
                if (searchInput) searchInput.value = '';
                document.querySelectorAll('.faq-item').forEach(item => item.classList.remove('hidden'));
            });
        });

        // Bind accordion toggles
        const accordionToggles = document.querySelectorAll('.accordion-toggle');
        accordionToggles.forEach(toggle => {
            toggle.addEventListener('click', () => {
                const item = toggle.closest('[data-accordion]');
                const content = item.querySelector('.accordion-content');
                const icon = item.querySelector('.accordion-icon');
                const isOpen = !content.classList.contains('hidden');

                // Close all others
                document.querySelectorAll('[data-accordion]').forEach(otherItem => {
                    if (otherItem !== item) {
                        otherItem.querySelector('.accordion-content')?.classList.add('hidden');
                        otherItem.querySelector('.accordion-icon')?.classList.remove('rotate-180');
                    }
                });

                // Toggle current
                if (!isOpen) {
                    content.classList.remove('hidden');
                    icon.classList.add('rotate-180');
                } else {
                    content.classList.add('hidden');
                    icon.classList.remove('rotate-180');
                }
            });
        });
    }

    async function loadFAQs() {
        const container = document.getElementById('faq-container');
        if (!container) return;

        Loader.show(container, 'skeleton');

        try {
            const response = await PagesApi.getFAQs();
            const faqs = response.data || [];
            allFAQs = faqs;

            if (faqs.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-12">
                        <svg class="w-16 h-16 text-stone-300 dark:text-stone-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                        </svg>
                        <p class="text-stone-500 dark:text-stone-400">No FAQs available at the moment.</p>
                    </div>
                `;
                return;
            }

            categories = groupByCategory(faqs);
            renderFAQs(categories);
        } catch (error) {
            console.error('Failed to load FAQs:', error);
            container.innerHTML = `
                <div class="text-center py-12">
                    <svg class="w-16 h-16 text-red-300 dark:text-red-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    <p class="text-red-500 dark:text-red-400">Failed to load FAQs. Please try again later.</p>
                </div>
            `;
        }
    }

    function groupByCategory(faqs) {
        const grouped = {};
        
        faqs.forEach(faq => {
            const category = faq.category || 'General';
            if (!grouped[category]) {
                grouped[category] = [];
            }
            grouped[category].push(faq);
        });

        return grouped;
    }

    function renderFAQs(categorizedFaqs, searchTerm = '') {
        const container = document.getElementById('faq-container');
        if (!container) return;

        const categoryNames = Object.keys(categorizedFaqs);

        if (categoryNames.length === 0) {
            container.innerHTML = `
                <div class="text-center py-12">
                    <svg class="w-16 h-16 text-stone-300 dark:text-stone-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    <p class="text-stone-500 dark:text-stone-400">No FAQs found${searchTerm ? ` for "${Templates.escapeHtml(searchTerm)}"` : ''}.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <!-- Category Tabs -->
            <div class="mb-8 overflow-x-auto scrollbar-hide">
                <div class="flex gap-2 pb-2">
                    <button class="faq-category-btn px-4 py-2 bg-primary-600 dark:bg-amber-600 text-white rounded-full text-sm font-medium whitespace-nowrap transition-colors" data-category="all">
                        All
                    </button>
                    ${categoryNames.map(cat => `
                        <button class="faq-category-btn px-4 py-2 bg-stone-100 dark:bg-stone-800 hover:bg-stone-200 dark:hover:bg-stone-700 text-stone-600 dark:text-stone-300 rounded-full text-sm font-medium whitespace-nowrap transition-colors" data-category="${Templates.escapeHtml(cat)}">
                            ${Templates.escapeHtml(cat)}
                        </button>
                    `).join('')}
                </div>
            </div>

            <!-- FAQ Accordion -->
            <div id="faq-list" class="space-y-6">
                ${categoryNames.map(category => `
                    <div class="faq-category" data-category="${Templates.escapeHtml(category)}">
                        <h2 class="text-lg font-semibold text-stone-900 dark:text-white mb-4">${Templates.escapeHtml(category)}</h2>
                        <div class="space-y-3">
                            ${categorizedFaqs[category].map(faq => renderFAQItem(faq, searchTerm)).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        bindCategoryTabs();
        bindAccordion();
        
        // Re-init enhanced features for dynamic content
        initFAQRating();
        initShareQuestion();
    }

    function renderFAQItem(faq, searchTerm = '') {
        let question = Templates.escapeHtml(faq.question);
        let answer = faq.answer;

        if (searchTerm) {
            const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
            question = question.replace(regex, '<mark class="bg-yellow-200 dark:bg-yellow-800">$1</mark>');
            answer = answer.replace(regex, '<mark class="bg-yellow-200 dark:bg-yellow-800">$1</mark>');
        }

        return `
            <div class="faq-item border border-stone-200 dark:border-stone-700 rounded-xl overflow-hidden bg-white dark:bg-stone-800" data-id="${faq.id || ''}">
                <button class="faq-trigger w-full px-6 py-4 text-left flex items-center justify-between hover:bg-stone-50 dark:hover:bg-stone-700/50 transition-colors">
                    <span class="font-medium text-stone-900 dark:text-white pr-4">${question}</span>
                    <svg class="faq-icon w-5 h-5 text-stone-500 dark:text-stone-400 flex-shrink-0 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                    </svg>
                </button>
                <div class="faq-content hidden px-6 pb-4">
                    <div class="prose prose-sm dark:prose-invert max-w-none text-stone-600 dark:text-stone-300">
                        ${answer}
                    </div>
                </div>
            </div>
        `;
    }

    function bindCategoryTabs() {
        const categoryBtns = document.querySelectorAll('.faq-category-btn');
        const categoryGroups = document.querySelectorAll('.faq-category');

        categoryBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                categoryBtns.forEach(b => {
                    b.classList.remove('bg-primary-600', 'dark:bg-amber-600', 'text-white');
                    b.classList.add('bg-stone-100', 'dark:bg-stone-800', 'text-stone-600', 'dark:text-stone-300');
                });
                btn.classList.add('bg-primary-600', 'dark:bg-amber-600', 'text-white');
                btn.classList.remove('bg-stone-100', 'dark:bg-stone-800', 'text-stone-600', 'dark:text-stone-300');

                const category = btn.dataset.category;

                categoryGroups.forEach(group => {
                    if (category === 'all' || group.dataset.category === category) {
                        group.classList.remove('hidden');
                    } else {
                        group.classList.add('hidden');
                    }
                });
            });
        });
    }

    function bindAccordion() {
        const triggers = document.querySelectorAll('.faq-trigger');

        triggers.forEach(trigger => {
            trigger.addEventListener('click', () => {
                const item = trigger.closest('.faq-item');
                const content = item.querySelector('.faq-content');
                const icon = item.querySelector('.faq-icon');
                const isOpen = !content.classList.contains('hidden');

                // Close all others
                document.querySelectorAll('.faq-item').forEach(otherItem => {
                    if (otherItem !== item) {
                        otherItem.querySelector('.faq-content')?.classList.add('hidden');
                        otherItem.querySelector('.faq-icon')?.classList.remove('rotate-180');
                    }
                });

                // Toggle current
                content.classList.toggle('hidden');
                icon.classList.toggle('rotate-180');
            });
        });
    }

    function initSearch() {
        const searchInput = document.getElementById('faq-search');
        if (!searchInput) return;

        let debounceTimer = null;

        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim().toLowerCase();

            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                // Check if we're working with server-rendered or JS-rendered content
                const serverRendered = document.querySelector('.accordion-toggle');
                
                if (serverRendered) {
                    // Server-rendered: filter existing DOM elements
                    filterServerRenderedFAQs(query);
                } else if (categories && Object.keys(categories).length > 0) {
                    // JS-rendered: re-render filtered content
                    if (query.length < 2) {
                        renderFAQs(categories);
                        return;
                    }

                    const filtered = {};
                    Object.entries(categories).forEach(([category, faqs]) => {
                        const matchingFaqs = faqs.filter(faq => 
                            faq.question.toLowerCase().includes(query) ||
                            faq.answer.toLowerCase().includes(query)
                        );
                        if (matchingFaqs.length > 0) {
                            filtered[category] = matchingFaqs;
                        }
                    });

                    renderFAQs(filtered, query);
                }
            }, 300);
        });

        // Show keyboard shortcut hint
        const hint = document.createElement('span');
        hint.className = 'absolute right-3 top-1/2 -translate-y-1/2 text-xs text-stone-400 dark:text-stone-500 hidden md:block';
        hint.textContent = 'Press / to search';
        if (searchInput.parentElement) {
            searchInput.parentElement.style.position = 'relative';
            searchInput.parentElement.appendChild(hint);

            searchInput.addEventListener('focus', () => hint.classList.add('hidden'));
            searchInput.addEventListener('blur', () => hint.classList.remove('hidden'));
        }
    }

    function filterServerRenderedFAQs(query) {
        const faqItems = document.querySelectorAll('.faq-item');
        const categoryGroups = document.querySelectorAll('.faq-category');
        const noResults = document.getElementById('no-results');
        
        let visibleCount = 0;
        
        faqItems.forEach(item => {
            const questionEl = item.querySelector('.accordion-toggle span, button span');
            const contentEl = item.querySelector('.accordion-content');
            
            const question = questionEl ? questionEl.textContent.toLowerCase() : '';
            const answer = contentEl ? contentEl.textContent.toLowerCase() : '';
            
            if (!query || question.includes(query) || answer.includes(query)) {
                item.classList.remove('hidden');
                visibleCount++;
            } else {
                item.classList.add('hidden');
            }
        });
        
        // Hide empty categories
        categoryGroups.forEach(cat => {
            const visibleItems = cat.querySelectorAll('.faq-item:not(.hidden)');
            cat.classList.toggle('hidden', visibleItems.length === 0);
        });
        
        // Show no results message
        if (noResults) {
            noResults.classList.toggle('hidden', visibleCount > 0);
        }
    }

    function destroy() {
        categories = [];
        allFAQs = [];
    }

    return {
        init,
        destroy
    };
})();

window.FAQPage = FAQPage;
export default FAQPage;
