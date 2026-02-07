/**
 * Bunoraa Service Worker
 * Enhanced PWA with offline support, push notifications, and background sync
 * @version 2.0.0
 */

const VERSION = '2.0.0';
const CACHE_NAME = `bunoraa-v${VERSION}`;
const STATIC_CACHE = `bunoraa-static-v${VERSION}`;
const DYNAMIC_CACHE = `bunoraa-dynamic-v${VERSION}`;
const IMAGE_CACHE = `bunoraa-images-v${VERSION}`;
const API_CACHE = `bunoraa-api-v${VERSION}`;

// Static assets to precache (only include files that definitely exist)
const PRECACHE_URLS = [
  '/static/css/styles.css',
  '/static/js/app.bundle.js',
  '/static/manifest.json',
  '/static/images/logo.svg',
  '/static/images/favicon.svg',
];

// API endpoints to cache
const API_CACHE_URLS = [
  '/api/v1/catalog/homepage/',
  '/api/v1/catalog/categories/',
];

// Cache duration settings (in seconds)
const CACHE_DURATION = {
  static: 86400 * 30, // 30 days
  dynamic: 86400,     // 1 day
  api: 300,           // 5 minutes
  images: 86400 * 7,  // 7 days
};

/**
 * Install event - Cache static assets
 */
self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker v' + VERSION);
  
  event.waitUntil(
    Promise.all([
      // Cache static assets (with error handling for missing files)
      caches.open(STATIC_CACHE).then((cache) => {
        console.log('[SW] Caching static assets');
        // Use individual requests instead of addAll to handle failures gracefully
        return Promise.allSettled(
          PRECACHE_URLS.map(url =>
            fetch(url).then(response => {
              if (response.ok) {
                return cache.put(url, response);
              }
              console.warn('[SW] Failed to cache:', url, response.status);
            }).catch(err => console.warn('[SW] Error caching:', url, err.message))
          )
        );
      }),
      // Cache API responses
      caches.open(API_CACHE).then((cache) => {
        console.log('[SW] Caching API responses');
        return Promise.all(
          API_CACHE_URLS.map(url => 
            fetch(url).then(response => {
              if (response.ok) {
                return cache.put(url, response);
              }
            }).catch(() => {})
          )
        );
      }),
    ]).then(() => {
      console.log('[SW] Installation complete');
      self.skipWaiting();
    })
  );
});

/**
 * Activate event - Clean old caches
 */
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker v' + VERSION);
  
  const currentCaches = [CACHE_NAME, STATIC_CACHE, DYNAMIC_CACHE, IMAGE_CACHE, API_CACHE];
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (!currentCaches.includes(cacheName)) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('[SW] Activation complete');
      return self.clients.claim();
    })
  );
});

/**
 * Fetch event - Handle requests with appropriate strategies
 */
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') return;
  
  // Skip cross-origin requests (except CDN assets)
  if (url.origin !== location.origin && !url.hostname.includes('bunoraa')) return;
  
  // Handle different request types
  if (request.mode === 'navigate') {
    // Navigation requests - Network first with offline fallback
    event.respondWith(handleNavigationRequest(request));
  } else if (url.pathname.startsWith('/api/')) {
    // API requests - Network first with cache fallback
    event.respondWith(handleApiRequest(request));
  } else if (isStaticAsset(request)) {
    // Static assets - Cache first
    event.respondWith(handleStaticRequest(request));
  } else if (isImageRequest(request)) {
    // Images - Cache first with size limits
    event.respondWith(handleImageRequest(request));
  } else {
    // Other requests - Network first
    event.respondWith(handleDynamicRequest(request));
  }
});

/**
 * Handle navigation requests
 */
async function handleNavigationRequest(request) {
  try {
    const response = await fetch(request);
    return response;
  } catch (error) {
    // Try to return cached version
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    // Return offline page
    return caches.match('/offline/');
  }
}

/**
 * Handle API requests - Network first with cache fallback
 */
async function handleApiRequest(request) {
  try {
    const response = await fetch(request);
    
    if (response.ok) {
      // Cache successful responses
      const cache = await caches.open(API_CACHE);
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    // Return cached version
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return error response
    return new Response(JSON.stringify({ error: 'Offline' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Handle static asset requests - Cache first
 */
async function handleStaticRequest(request) {
  const cachedResponse = await caches.match(request);
  
  if (cachedResponse) {
    // Return cached and update in background
    updateCache(request, STATIC_CACHE);
    return cachedResponse;
  }
  
  try {
    const response = await fetch(request);
    
    if (response.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    return new Response('', { status: 404 });
  }
}

/**
 * Handle image requests with lazy caching
 */
async function handleImageRequest(request) {
  const cachedResponse = await caches.match(request);
  
  if (cachedResponse) {
    return cachedResponse;
  }
  
  try {
    const response = await fetch(request);
    
    if (response.ok) {
      const cache = await caches.open(IMAGE_CACHE);
      
      // Limit image cache size
      limitCacheSize(IMAGE_CACHE, 100);
      
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    // Return transparent 1x1 pixel for failed images
    return new Response(
      Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='), c => c.charCodeAt(0)),
      { status: 200, headers: { 'Content-Type': 'image/png' } }
    );
  }
}

/**
 * Handle dynamic requests - Network first
 */
async function handleDynamicRequest(request) {
  try {
    const response = await fetch(request);
    
    if (response.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    return caches.match(request);
  }
}

/**
 * Update cache in background
 */
async function updateCache(request, cacheName) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, response);
    }
  } catch (error) {
    // Ignore update errors
  }
}

/**
 * Limit cache size
 */
async function limitCacheSize(cacheName, maxItems) {
  const cache = await caches.open(cacheName);
  const keys = await cache.keys();
  
  if (keys.length > maxItems) {
    // Delete oldest items
    const deleteCount = keys.length - maxItems;
    for (let i = 0; i < deleteCount; i++) {
      await cache.delete(keys[i]);
    }
  }
}

/**
 * Check if request is for static asset
 */
function isStaticAsset(request) {
  const url = new URL(request.url);
  return url.pathname.startsWith('/static/') ||
         request.destination === 'script' ||
         request.destination === 'style' ||
         request.destination === 'font';
}

/**
 * Check if request is for image
 */
function isImageRequest(request) {
  return request.destination === 'image' ||
         /\.(png|jpg|jpeg|gif|webp|svg|ico)$/i.test(request.url);
}

/**
 * Push notification event
 */
self.addEventListener('push', (event) => {
  console.log('[SW] Push notification received');
  
  let data = { title: 'Bunoraa', body: 'নতুন আপডেট আছে!' };
  
  if (event.data) {
    try {
      data = event.data.json();
    } catch (e) {
      data.body = event.data.text();
    }
  }
  
  const options = {
    body: data.body,
    icon: '/static/images/icons/icon-192x192.png',
    badge: '/static/images/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: data.data || {},
    actions: data.actions || [
      { action: 'open', title: 'দেখুন' },
      { action: 'close', title: 'বন্ধ করুন' }
    ],
    tag: data.tag || 'bunoraa-notification',
    renotify: true,
    requireInteraction: false,
  };
  
  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

/**
 * Notification click event
 */
self.addEventListener('notificationclick', (event) => {
  console.log('[SW] Notification clicked:', event.action);
  
  event.notification.close();
  
  if (event.action === 'close') {
    return;
  }
  
  const urlToOpen = event.notification.data?.url || '/';
  
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then((clientList) => {
        // Check if already open
        for (const client of clientList) {
          if (client.url.includes(urlToOpen) && 'focus' in client) {
            return client.focus();
          }
        }
        // Open new window
        if (clients.openWindow) {
          return clients.openWindow(urlToOpen);
        }
      })
  );
});

/**
 * Background sync event
 */
self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync:', event.tag);
  
  if (event.tag === 'sync-cart') {
    event.waitUntil(syncCart());
  } else if (event.tag === 'sync-wishlist') {
    event.waitUntil(syncWishlist());
  } else if (event.tag === 'sync-analytics') {
    event.waitUntil(syncAnalytics());
  }
});

/**
 * Sync cart with server
 */
async function syncCart() {
  try {
    const cache = await caches.open('bunoraa-offline-data');
    const cartRequest = await cache.match('/offline/cart');
    
    if (cartRequest) {
      const cartData = await cartRequest.json();
      
      await fetch('/api/v1/commerce/cart/sync/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cartData)
      });
      
      await cache.delete('/offline/cart');
    }
  } catch (error) {
    console.error('[SW] Cart sync failed:', error);
  }
}

/**
 * Sync wishlist with server
 */
async function syncWishlist() {
  try {
    const cache = await caches.open('bunoraa-offline-data');
    const wishlistRequest = await cache.match('/offline/wishlist');
    
    if (wishlistRequest) {
      const wishlistData = await wishlistRequest.json();
      
      await fetch('/api/v1/commerce/wishlist/sync/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(wishlistData)
      });
      
      await cache.delete('/offline/wishlist');
    }
  } catch (error) {
    console.error('[SW] Wishlist sync failed:', error);
  }
}

/**
 * Sync analytics data
 */
async function syncAnalytics() {
  try {
    const cache = await caches.open('bunoraa-offline-data');
    const analyticsRequest = await cache.match('/offline/analytics');
    
    if (analyticsRequest) {
      const analyticsData = await analyticsRequest.json();
      
      await fetch('/api/v1/analytics/batch/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(analyticsData)
      });
      
      await cache.delete('/offline/analytics');
    }
  } catch (error) {
    console.error('[SW] Analytics sync failed:', error);
  }
}

/**
 * Message event - Communicate with main thread
 */
self.addEventListener('message', (event) => {
  console.log('[SW] Message received:', event.data);
  
  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  } else if (event.data.type === 'CACHE_URLS') {
    event.waitUntil(
      caches.open(DYNAMIC_CACHE).then((cache) => {
        return cache.addAll(event.data.urls);
      })
    );
  } else if (event.data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.keys().then((keys) => {
        return Promise.all(keys.map((key) => caches.delete(key)));
      })
    );
  }
});

console.log('[SW] Service Worker loaded v' + VERSION);