const CACHE = 'aroaxinping-v3';
const ASSETS = ['/', '/index.html', '/css/style.css', '/js/main.js', '/favicon.svg'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  // páginas: red primero (para que los deploys lleguen), caché solo como fallback offline
  if (e.request.mode === 'navigate') {
    e.respondWith(
      fetch(e.request)
        .then(r => {
          const copy = r.clone();
          caches.open(CACHE).then(c => c.put(e.request, copy));
          return r;
        })
        .catch(() => caches.match(e.request))
    );
    return;
  }
  // assets: caché primero (ya llevan ?v= para invalidar)
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
});
