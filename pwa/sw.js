// ═══════════════════════════════════════════════════════════════
//  AI Malaria Lab Assistant — Service Worker (sw.js)
//  Enables offline access and PWA install prompt
// ═══════════════════════════════════════════════════════════════

const CACHE_NAME = 'malaria-ai-v1';

// Assets to pre-cache for offline use
const PRECACHE_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
];

// ── Install event: cache shell assets ────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('[SW] Pre-caching shell assets');
      return cache.addAll(PRECACHE_ASSETS);
    }).then(() => self.skipWaiting())
  );
});

// ── Activate event: clean old caches ─────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

// ── Fetch event: network-first for Streamlit, cache for shell ─
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // For Streamlit API calls and websocket: always go to network
  if (url.pathname.startsWith('/_stcore') ||
      url.pathname.startsWith('/stream') ||
      url.protocol === 'ws:' ||
      url.protocol === 'wss:') {
    return; // Let browser handle directly
  }

  // For shell assets: cache-first
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;

      return fetch(event.request).then(response => {
        // Only cache successful same-origin responses
        if (response && response.status === 200 && response.type === 'basic') {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      }).catch(() => {
        // Offline fallback
        if (event.request.destination === 'document') {
          return caches.match('/index.html');
        }
      });
    })
  );
});
