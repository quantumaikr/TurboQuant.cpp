/**
 * inference-worker.js — Web Worker that runs WASM inference off the main thread.
 *
 * Eliminates ASYNCIFY entirely: the worker can block on quant_generate()
 * while the main thread stays responsive. Tokens stream via postMessage().
 *
 * Protocol:
 *   Main → Worker: { type: 'load', bytes: ArrayBuffer }
 *   Main → Worker: { type: 'generate', prompt: string, temperature: number, maxTokens: number }
 *   Worker → Main: { type: 'status', msg: string }
 *   Worker → Main: { type: 'token', text: string }
 *   Worker → Main: { type: 'done', nTokens: number, elapsed: number }
 *   Worker → Main: { type: 'ready' }
 */

/* Load the Emscripten glue code. Module is configured before loading. */
var Module = {
    onToken: null,
    onDone: null,
    onStatus: null,
    print: function(text) { /* suppress stdout in worker */ },
    printErr: function(text) { /* suppress stderr in worker */ },
    onRuntimeInitialized: function() {
        postMessage({ type: 'ready' });
    }
};

importScripts('quant.js');

onmessage = function(e) {
    const msg = e.data;

    if (msg.type === 'load') {
        try {
            const bytes = new Uint8Array(msg.bytes);
            Module.FS.writeFile('/model.gguf', bytes);
            const rc = Module._wasm_load_model(Module.allocateUTF8('/model.gguf'));
            if (rc === 0) {
                const info = Module._wasm_model_info();
                postMessage({ type: 'status', msg: 'Model loaded! Ready to chat. (' + msg.name + ')' });
                postMessage({ type: 'loaded', size: bytes.length, name: msg.name });
            } else {
                postMessage({ type: 'status', msg: 'Error: failed to load model' });
            }
        } catch (err) {
            postMessage({ type: 'status', msg: 'Error: ' + err.message });
        }
        return;
    }

    if (msg.type === 'generate') {
        postMessage({ type: 'status', msg: 'thinking' });

        /* Set up per-token callback — posts each token to main thread */
        Module.onToken = function(text) {
            postMessage({ type: 'token', text: text });
        };

        const promptPtr = Module.allocateUTF8(msg.prompt);
        /* This blocks the worker (not the main thread!) until generation completes */
        Module._wasm_generate(promptPtr, msg.temperature || 0.7, msg.maxTokens || 256);
        Module._free(promptPtr);

        postMessage({ type: 'done' });
        return;
    }
};
