/* Cross-origin isolation shim for hosts that cannot send response headers
 * (GitHub Pages). The wasm build needs SharedArrayBuffer, which needs
 * COOP/COEP; a service worker can add those headers to every response it
 * proxies. First visit: register + one reload, after which the page is
 * crossOriginIsolated. Served with real headers (wasm/serve.py), this file
 * does nothing. */
if (typeof window === "undefined") {
  /* Service-worker context: proxy fetches, stamping the isolation headers. */
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", (e) => {
    const req = e.request;
    if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
    e.respondWith(fetch(req).then((res) => {
      if (res.status === 0) return res; /* opaque: pass through untouched */
      const h = new Headers(res.headers);
      h.set("Cross-Origin-Opener-Policy", "same-origin");
      h.set("Cross-Origin-Embedder-Policy", "require-corp");
      h.set("Cross-Origin-Resource-Policy", "cross-origin");
      return new Response(res.body, {
        status: res.status, statusText: res.statusText, headers: h,
      });
    }));
  });
} else if (!window.crossOriginIsolated && window.isSecureContext &&
           navigator.serviceWorker) {
  /* Page context, not yet isolated: install the worker and reload once.
   * The sessionStorage guard stops a reload loop if isolation still fails
   * (e.g. an extension interferes). */
  const KEY = "coiReloadedOnce";
  navigator.serviceWorker.register(document.currentScript.src).then((reg) => {
    const reload = () => {
      if (sessionStorage.getItem(KEY)) return;
      sessionStorage.setItem(KEY, "1");
      location.reload();
    };
    if (reg.active && !navigator.serviceWorker.controller) reload();
    navigator.serviceWorker.addEventListener("controllerchange", reload);
  }).catch((err) => console.warn("coi-serviceworker:", err));
} else if (window.crossOriginIsolated) {
  sessionStorage.removeItem("coiReloadedOnce");
}
