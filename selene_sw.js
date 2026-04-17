// selene_sw.js — Service Worker da Selene PWA
// Versão do cache: incrementar para forçar atualização.
const CACHE = 'selene-v1';
const SHELL = ['/mobile', '/manifest.json'];

// Instala e pré-cacheia o shell da aplicação
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL))
  );
  self.skipWaiting();
});

// Ativa e limpa caches antigos
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Estratégia: network-first para rotas da Selene, cache-fallback para o shell
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // Nunca intercepta WebSocket (ws://)
  if (url.protocol === 'ws:' || url.protocol === 'wss:') return;
  // Fontes externas: ignora (passa direto)
  if (!url.hostname.includes(self.location.hostname) && url.hostname !== self.location.hostname) return;

  e.respondWith(
    fetch(e.request)
      .then(res => {
        // Atualiza cache com resposta fresca (apenas GETs)
        if (e.request.method === 'GET') {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
