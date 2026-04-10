/*! coi-serviceworker v0.1.7 - Guido Zuidhof, licensed under MIT */
/*
 * Service Worker that injects Cross-Origin-Opener-Policy and
 * Cross-Origin-Embedder-Policy headers into all responses.
 * This enables SharedArrayBuffer on hosts that don't support
 * custom HTTP headers (e.g., GitHub Pages).
 *
 * Required for WASM pthreads (multi-threaded inference).
 */
if (typeof window === 'undefined') {
    // Service Worker scope
    self.addEventListener("install", () => self.skipWaiting());
    self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

    self.addEventListener("fetch", (e) => {
        // Only intercept same-origin or navigation requests
        if (
            e.request.cache === "only-if-cached" &&
            e.request.mode !== "same-origin"
        ) {
            return;
        }

        e.respondWith(
            fetch(e.request).then((response) => {
                // Can't modify opaque responses
                if (response.status === 0) return response;

                const newHeaders = new Headers(response.headers);
                newHeaders.set("Cross-Origin-Embedder-Policy", "credentialless");
                newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");

                return new Response(response.body, {
                    status: response.status,
                    statusText: response.statusText,
                    headers: newHeaders,
                });
            }).catch((err) => {
                console.error("coi-serviceworker fetch error:", err);
                return new Response("Service Worker fetch error", { status: 500 });
            })
        );
    });
} else {
    // Window scope — register the service worker
    (async () => {
        if (!window.crossOriginIsolated) {
            const reg = await navigator.serviceWorker.register(
                window.document.currentScript.src
            );
            if (reg.active && !navigator.serviceWorker.controller) {
                // Service worker installed but not controlling — reload to activate
                window.location.reload();
            } else if (!reg.active) {
                // Wait for the service worker to activate, then reload
                const sw = reg.installing || reg.waiting;
                sw.addEventListener("statechange", () => {
                    if (sw.state === "activated") window.location.reload();
                });
            }
        }
    })();
}
