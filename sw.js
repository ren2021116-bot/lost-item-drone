// 最小のService Worker。常にネットワークを優先し、つながらないときだけ前回の内容を使う
// (更新したのに古い画面が出る、という事故を避けるため)
const CACHE = "findrone-v1";

self.addEventListener("install", () => { self.skipWaiting(); });

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;
  event.respondWith((async () => {
    try {
      const res = await fetch(req);
      if (res.ok && res.status === 200) {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy));
      }
      return res;
    } catch (err) {
      const hit = await caches.match(req);
      if (hit) return hit;
      throw err;
    }
  })());
});
