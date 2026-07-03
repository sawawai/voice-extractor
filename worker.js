/* ============================================================
   worker.js — runs the WASM path off the main thread.
   WebGPU stays on the main thread (it garbles in a worker);
   WASM runs here so demucs-web's synchronous STFT doesn't block
   the UI or trip Firefox's slow-script watchdog. Same engine
   core as the main thread, forced to backend 'wasm'.
   ============================================================ */
import { runSeparation } from "./engine.js";

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type !== "separate") return;

  const hooks = {
    status: (text) => postMessage({ type: "status", text }),
    progress: (value, seg, total) => postMessage({ type: "progress", value, seg, total }),
    download: (loaded, total) => postMessage({ type: "download", loaded, total }),
    note: (text) => postMessage({ type: "note", text }),
    engine: (name) => postMessage({ type: "engine", name }),
  };

  try {
    const res = await runSeparation(msg.channels, msg.sampleRate, "quality", hooks, { backend: "wasm" });
    const transfer = [res.left.buffer];
    if (res.right && res.right.buffer !== res.left.buffer) transfer.push(res.right.buffer);
    postMessage({ type: "done", left: res.left, right: res.right || null }, transfer);
  } catch (err) {
    postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
