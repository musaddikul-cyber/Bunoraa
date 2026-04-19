/**
 * Bunoraa Service Worker
 * Provides offline support, caching strategies, and background sync.
 * 
 * @version 1.0.0
 */

const CACHE_VERSION = 'v1';
const STATIC_CACHE = `bunoraa-static-${CACHE_VERSION}`;
const IMAGE_CACHE = `bunoraa-images-${CACHE_VERSION}`;
const API_CACHE = `bunoraa-api-${CACHE_VERSION}`;
const PAGES_CACHE = `bunoraa-pages-${CACHE_VERSION}`;

// Assets to cache on install
const STATIC_ASSETS = [
  '/',
  '/offline.html',
  '/styles/offline.css',
  '/icon.png',
  '/apple-icon.png',
  '/favicon.ico',
  '/site.webmanifest',
];

// Cache strategies
const CACHE_STRATEGIES = {
  // Network first for API calls
  api: {
    maxAge: 60 * 1000, // 1 minute
    maxEntries: 100,
  },
  // Cache first with network fallback for static assets
  static: {
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
    maxEntries: 200,
  },
  // Stale while revalidate for images
  images: {
    maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
    maxEntries: 500,
  },
  // Cache first for pages with short TTL
  pages: {
    maxAge: 5 * 60 * 1000, // 5 minutes
    maxEntries: 50,
  },
};

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[SW] Installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS).catch((err) => {
          console.warn('[SW] Failed to cache some static assets:', err);
          // Continue installation even if some assets fail
          return Promise.resolve();
        });
      })
      .then(() => {
        console.log('[SW] Installation complete');
        return self.skipWaiting();
      })
      .catch((err) => {
        console.error('[SW] Installation failed:', err);
        throw err;
      })
  );
});

// Activate event - cleanup old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating...');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => {
              // Delete old version caches
              return name.startsWith('bunoraa-') && 
                     !name.includes(CACHE_VERSION);
            })
            .map((name) => {
              console.log('[SW] Deleting old cache:', name);
              return caches.delete(name);
            })
        );
      })
      .then(() => {
        console.log('[SW] Activation complete');
        return self.clients.claim();
      })
  );
});

// Fetch event - handle caching strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip extensions and chrome URLs
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return;
  }
  
  // Route to appropriate cache strategy
  if (isAssetRequest(request)) {
    event.respondWith(assetStrategy(request));
  } else if (isImageRequest(request)) {
    event.respondWith(imageStrategy(request));
  } else if (isAPIRequest(request)) {
    event.respondWith(apiStrategy(request));
  } else if (isPageRequest(request)) {
    event.respondWith(pageStrategy(request));
  }
});

// Helper: Check if request is for static assets
function isAssetRequest(request) {
  return /\.(js|css|woff2?|json)$/.test(request.url);
}

// Helper: Check if request is for images
function isImageRequest(request) {
  return /\.(png|jpg|jpeg|gif|svg|webp|avif|ico)$/.test(request.url);
}

// Helper: Check if request is for API
function isAPIRequest(request) {
  const url = new URL(request.url);
  return url.pathname.startsWith('/api/') || 
         url.pathname.startsWith('/graphql');
}

// Helper: Check if request is for page
function isPageRequest(request) {
  const acceptHeader = request.headers.get('accept') || '';
  return acceptHeader.includes('text/html');
}

// Strategy: Cache first for static assets
async function assetStrategy(request) {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);
  
  if (cached) {
    // Return cached and refresh in background
    fetchAndCache(request, cache);
    return cached;
  }
  
  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.error('[SW] Asset fetch failed:', error);
    return cached || new Response('Asset unavailable', { status: 503 });
  }
}

// Strategy: Stale while revalidate for images
async function imageStrategy(request) {
  const cache = await caches.open(IMAGE_CACHE);
  const cached = await cache.match(request);
  
  // Always try to fetch fresh in background
  const networkPromise = fetch(request)
    .then((response) => {
      if (response.ok) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => cached); // Fallback to cache on network error
  
  // Return cached immediately if available, then update
  if (cached) {
    // Check cache age
    const cacheDate = cached.headers.get('date');
    const cacheAge = cacheDate ? Date.now() - new Date(cacheDate).getTime() : Infinity;
    
    if (cacheAge < CACHE_STRATEGIES.images.maxAge) {
      fetchAndCache(request, cache); // Refresh in background
      return cached;
    }
  }
  
  return networkPromise;
}

// Strategy: Network first for API calls
async function apiStrategy(request) {
  const cache = await caches.open(API_CACHE);
  
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      // Cache successful API responses
      const responseClone = networkResponse.clone();
      cache.put(request, responseClone);
      
      // Enforce cache size limits
      enforceCacheLimit(cache, CACHE_STRATEGIES.api.maxEntries);
      
      return networkResponse;
    }
    
    throw new Error('Network response not ok');
  } catch (error) {
    // Network failed, try cache
    const cached = await cache.match(request);
    
    if (cached) {
      console.log('[SW] Serving cached API response');
      return cached;
    }
    
    // Return offline error for API
    return new Response(
      JSON.stringify({ 
        error: 'Offline', 
        message: 'You are currently offline. Please check your connection.' 
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

// Strategy: Network first with offline fallback for pages
async function pageStrategy(request) {
  const cache = await caches.open(PAGES_CACHE);
  
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
      enforceCacheLimit(cache, CACHE_STRATEGIES.pages.maxEntries);
      return networkResponse;
    }
    
    throw new Error('Network response not ok');
  } catch (error) {
    const cached = await cache.match(request);
    
    if (cached) {
      return cached;
    }
    
    // Return offline page
    const offlinePage = await caches.match('/offline.html');
    if (offlinePage) {
      return offlinePage;
    }
    
    // Generic offline response
    return new Response(
      `<!DOCTYPE html>
      <html>
        <head>
          <title>Offline - Bunoraa</title>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            body { font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; background: #f9fafb; }
            .container { text-align: center; padding: 2rem; }
            h1 { color: #1f2937; margin-bottom: 1rem; }
            p { color: #6b7280; margin-bottom: 1.5rem; }
            a { color: #059669; text-decoration: none; }
            a:hover { text-decoration: underline; }
          </style>
        </head>
        <body>
          <div class="container">
            <h1>You're Offline</h1>
            <p>Please check your internet connection and try again.</p>
            <a href="/">← Back to Home</a>
          </div>
        </body>
      </html>`,
      {
        status: 200,
        headers: { 'Content-Type': 'text/html' },
      }
    );
  }
}

// Helper: Fetch and cache in background
async function fetchAndCache(request, cache) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response);
    }
  } catch (error) {
    // Silently fail background updates
  }
}

// Helper: Enforce cache size limit
async function enforceCacheLimit(cache, maxEntries) {
  const keys = await cache.keys();
  if (keys.length > maxEntries) {
    // Delete oldest entries
    const keysToDelete = keys.slice(0, keys.length - maxEntries);
    for (const key of keysToDelete) {
      cache.delete(key);
    }
  }
}

// Background sync for offline form submissions
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-forms') {
    event.waitUntil(syncFormSubmissions());
  }
});

// Sync queued form submissions
async function syncFormSubmissions() {
  // This would integrate with IndexedDB to queue and retry form submissions
  console.log('[SW] Syncing offline form submissions');
  // Implementation depends on your form handling strategy
}

// Push notification support
self.addEventListener('push', (event) => {
  let payload = {};
  try {
    payload = event.data ? event.data.json() : {};
  } catch {
    payload = { body: event.data ? event.data.text() : '' };
  }
  
  const title = payload.title || 'Bunoraa';
  const options = {
    body: payload.body || '',
    icon: payload.icon || '/icon.png',
    badge: payload.badge || '/favicon.ico',
    data: payload.data || {},
    actions: payload.actions || [],
    requireInteraction: false,
  };
  
  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  const targetUrl = event.notification.data?.url || '/';
  
  event.waitUntil(
    clients.matchAll({ type: 'window' }).then((clientList) => {
      // Focus existing window if open
      for (const client of clientList) {
        if (client.url === targetUrl && 'focus' in client) {
          return client.focus();
        }
      }
      // Open new window
      if (clients.openWindow) {
        return clients.openWindow(targetUrl);
      }
    })
  );
});

// Message handling for app-level communication
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data && event.data.type === 'CLEAR_CACHE') {
    caches.keys().then((cacheNames) => {
      return Promise.all(cacheNames.map((name) => caches.delete(name)));
    });
  }
});

console.log('[SW] Service Worker loaded');
