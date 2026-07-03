/* ============================================================
   engine.js — shared inference core (runs on the MAIN THREAD for
   WebGPU, and inside a Web Worker for WASM).
   ------------------------------------------------------------
   Why the split:
     • ORT's WebGPU EP garbles Demucs when run in a Worker, but is
       correct on the main thread — so WebGPU runs on main thread.
     • demucs-web's synchronous STFT blocks the main thread long
       enough that Firefox's slow-script watchdog kills it mid-run
       (truncated/garbled output) — so WASM runs in a Worker.
   app.js picks the path via verifyWebGPU() — a one-time on-GPU
   correctness probe (no UA sniffing); the worker calls this with
   { backend: 'wasm' }.

   runSeparation(channels, sampleRate, mode, hooks, { backend })
     hooks = { progress, download, status, note, engine }
     -> { left: Float32Array, right: Float32Array | null }

   Dev overrides (URL params): ?webgpu=off  force WASM
                               ?probe=off|reset  skip / re-run GPU probe
                               ?ort=1.25.1  try another ORT on the WebGPU path
   ============================================================ */

import {
  PROBE_SAMPLES, PROBE_REF_FILE, makeProbeSignal, compareProbe, upcastFp16,
} from "./probe.js";

function qp(name) {
  try { return new URL(self.location.href).searchParams.get(name); } catch { return null; }
}

const ORT_VER      = qp("ort") || "1.21.0";
const ORT_ALL      = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/ort.all.mjs`;      // WebGPU (jsep)
const ORT_WASM_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/ort.wasm.min.mjs`; // plain WASM
const DEMUCS_URL   = "https://esm.sh/demucs-web@1.0.2";

const ENABLE_WEBGPU = true;
// One shared fp16 weights blob (~85MB) serves BOTH backends; each backend gets
// a small graph file. WebGPU upcasts the blob to fp32 in JS so the GPU runs
// pure-fp32 compute (fp16 compute garbles — ORT #26732, still open); WASM
// feeds the blob directly (its CPU kernels upcast internally — proven correct).
const MODEL_GRAPH_FP32 = "./models/htdemucs_graph_fp32.onnx"; // ~2.6MB — WebGPU graph
const MODEL_GRAPH_FP16 = "./models/htdemucs_graph_fp16.onnx"; // ~2.6MB — WASM graph
const WEIGHTS_FP16     = "./models/weights_fp16.bin";         // ~85MB  — shared weights
const PROBE_REF_URL    = "./models/" + PROBE_REF_FILE;        // ~170KB — GPU probe reference
// Legacy monolithic models — fallback when the split artifacts aren't deployed.
const MODEL_FP32 = "./models/htdemucs_embedded.onnx";        // ~172MB
const MODEL_FP16 = "./models/htdemucs_embedded_fp16.onnx";   // ~86MB

const MODEL_CACHE = "voice-model-v1";
const FFT_SIZE = 4096;
const HOP = FFT_SIZE / 4;

function abs(url) { return new URL(url, self.location.href).href; }

/* ---------- backend decision (call on the main thread) ---------- */
export async function canUseWebGPU() {
  try {
    if (!ENABLE_WEBGPU || qp("webgpu") === "off") return false;
    if (typeof navigator === "undefined" || !navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

/* ---------- WebGPU correctness probe (main thread) ----------
   No UA sniffing: separate a deterministic test signal on the GPU once and
   compare against the CPU-computed reference (models/probe_ref_v1.bin).
   Catches Firefox's WebGPU miscomputation AND buggy drivers (e.g. some Intel
   iGPUs, ORT #24442), and auto-enables Firefox the day its WebGPU is fixed.
   The verdict is cached per GPU + ORT version; the session it builds is
   reused by the real run, so a passing probe costs one extra segment (~2s). */
export async function verifyWebGPU(hooks) {
  if (!(await canUseWebGPU())) return false;
  if (qp("probe") === "off") return true;

  let key = null;
  try {
    key = `voiceext:webgpu-ok:${PROBE_REF_FILE}:${ORT_VER}:${await adapterKey()}`;
    if (qp("probe") !== "reset") {
      const v = localStorage.getItem(key);
      if (v === "ok") return true;
      if (v === "bad") return false;
    }
  } catch { /* no localStorage — probe every time */ }

  try {
    hooks.status("初回のみ：GPU が正しく計算できるか確認しています…");
    const refBuf = await fetchModelCached(abs(PROBE_REF_URL), hooks);
    const sig = makeProbeSignal(PROBE_SAMPLES);
    const quiet = { ...hooks, progress: () => {}, engine: () => {} };
    const res = await runDemucs([sig.left, sig.right], quiet, "webgpu");
    const { ok, relErr } = compareProbe(res.left, res.right || res.left, new Float32Array(refBuf));
    console.info(`[webgpu probe] relErr=${relErr.toFixed(4)} -> ${ok ? "ok" : "wasm fallback"}`);
    try { if (key) localStorage.setItem(key, ok ? "ok" : "bad"); } catch { /* ignore */ }
    if (!ok) await releaseProcessor("webgpu"); // don't hold GPU memory we'll never use
    return ok;
  } catch (err) {
    // init/download failures may be transient — fall back now, don't cache
    console.warn("[webgpu probe] failed:", err);
    await releaseProcessor("webgpu");
    return false;
  }
}

async function releaseProcessor(backend) {
  const p = _procs.get(backend);
  _procs.delete(backend);
  try { if (p && p.session && p.session.release) await p.session.release(); } catch { /* ignore */ }
}

async function adapterKey() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
    if (info) return [info.vendor, info.architecture, info.device, info.description].join("|");
  } catch { /* fall through */ }
  return "unknown-gpu";
}

/* ---------- public entry ---------- */
export async function runSeparation(channels, sampleRate, mode, hooks, opts = {}) {
  const backend = opts.backend || "wasm";
  if (mode === "fast") {
    hooks.engine("center-extract");
    return centerExtract(channels, hooks);
  }
  try {
    hooks.engine("demucs");
    return await runDemucs(channels, hooks, backend);
  } catch (err) {
    hooks.note("Demucs を利用できません。簡易エンジンに切り替えます（" + String((err && err.message) || err) + "）");
    hooks.engine("center-extract");
    return centerExtract(channels, hooks);
  }
}

/* ---------- ORT / demucs-web loading ---------- */
let _loaded = null;
async function load(backend) {
  if (_loaded) return _loaded;
  const dm = await import(/* @vite-ignore */ DEMUCS_URL);

  let ort, localOrt = false;
  if (backend === "webgpu") {
    // Main thread: WebGPU EP. CDN is fine (no pthread workers spawned).
    ort = await import(/* @vite-ignore */ ORT_ALL);
    ort.env.wasm.numThreads = (self.navigator && navigator.hardwareConcurrency) || 4;
    try { ort.env.webgpu.powerPreference = "high-performance"; } catch { /* no webgpu env */ }
  } else {
    const r = await loadWasmOrt();   // worker: same-origin ORT so threads spawn
    ort = r.ort; localOrt = r.localOrt;
  }

  _loaded = { ort, DemucsProcessor: dm.DemucsProcessor, CONSTANTS: dm.CONSTANTS, backend, localOrt };
  return _loaded;
}

async function loadWasmOrt() {
  // Prefer the self-hosted ./ort/ (same-origin) so the pthread pool can spawn
  // inside the worker; fall back to CDN single-thread if it's not populated.
  const base = new URL("./ort/", self.location.href).href;
  try {
    const r = await fetch(base + "READY", { cache: "no-store" });
    if (r.ok) {
      const avail = (await r.text()).split("\n").map((s) => s.trim()).filter(Boolean);
      const prefer = ["ort.wasm.min.mjs", "ort.wasm.bundle.min.mjs", "ort.min.mjs"];
      const entry = prefer.find((f) => avail.includes(f)) || avail.find((f) => /wasm/.test(f)) || avail[0];
      if (entry) {
        const ort = await import(/* @vite-ignore */ base + entry);
        ort.env.wasm.wasmPaths = base;
        const cores = (self.navigator && navigator.hardwareConcurrency) || 4;
        ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(cores, 12) : 1;
        return { ort, localOrt: true };
      }
    }
  } catch { /* not populated */ }

  const ort = await import(/* @vite-ignore */ ORT_WASM_CDN);
  ort.env.wasm.numThreads = 1;      // CDN pthreads hang under COEP in a worker
  return { ort, localOrt: false };
}

const _procs = new Map();     // backend -> loaded DemucsProcessor (probe + runs share it)
let _progressSink = () => {}; // cached processors report to whichever run is active

async function runDemucs(channels, hooks, backend) {
  const { ort, DemucsProcessor } = await load(backend);
  const gpu = backend === "webgpu";
  _progressSink = (p, seg, total) => hooks.progress(p, seg, total);

  let processor = _procs.get(backend);
  if (!processor) {
    const { graphBuf, externalData } = await loadModelArtifacts(gpu, hooks);
    hooks.status("モデルを初期化しています…（少し時間がかかります）");
    processor = new DemucsProcessor({
      ort,
      // merged over demucs-web's defaults (executionProviders, graphOptimizationLevel)
      sessionOptions: externalData ? { externalData } : {},
      onProgress: ({ progress, currentSegment, totalSegments }) =>
        _progressSink(progress, currentSegment, totalSegments),
    });
    await processor.loadModel(graphBuf);
    _procs.set(backend, processor);
  }

  hooks.status("声を抽出しています…");
  const L = channels[0];
  const R = channels[1] || channels[0];
  const result = await processor.separate(L, R);
  return { left: result.vocals.left, right: result.vocals.right };
}

/* Split artifacts: small graph + shared fp16 blob. WebGPU reconstructs the
   fp32 external file in memory by upcasting the blob (`weights_fp32.bin` /
   `weights_fp16.bin` must match the `location` written by
   scripts/convert_external_fp16.py). Falls back to the legacy monolithic
   models if the split files aren't deployed. */
async function loadModelArtifacts(gpu, hooks) {
  try {
    const graphBuf = await fetchModelCached(abs(gpu ? MODEL_GRAPH_FP32 : MODEL_GRAPH_FP16), hooks);
    const blob16 = await fetchModelCached(abs(WEIGHTS_FP16), hooks);
    const weights = gpu ? new Uint8Array(upcastFp16(blob16).buffer) : new Uint8Array(blob16);
    dropLegacyCache();
    return {
      graphBuf,
      externalData: [{ data: weights, path: gpu ? "weights_fp32.bin" : "weights_fp16.bin" }],
    };
  } catch {
    const buf = await fetchModelCached(abs(gpu ? MODEL_FP32 : MODEL_FP16), hooks);
    return { graphBuf: buf, externalData: null };
  }
}

let _droppedLegacy = false;
function dropLegacyCache() {
  if (_droppedLegacy) return;
  _droppedLegacy = true; // frees ~260MB for users who cached the old monolithic models
  try {
    caches.open(MODEL_CACHE)
      .then((c) => { c.delete(abs(MODEL_FP32)); c.delete(abs(MODEL_FP16)); })
      .catch(() => {});
  } catch { /* no Cache API */ }
}

/* ---------- streamed + cached + retried model fetch ---------- */
async function fetchModelCached(url, hooks) {
  let cache = null;
  try { cache = await caches.open(MODEL_CACHE); } catch { /* private mode */ }

  if (cache) {
    const hit = await cache.match(url);
    if (hit) { hooks.status("モデルを読み込んでいます…"); return await hit.arrayBuffer(); }
  }

  let lastErr;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      hooks.status(attempt === 1 ? "モデルをダウンロードしています…" : `再試行しています…（${attempt}/3）`);
      const resp = await fetch(url);
      if (!resp.ok) throw new Error("HTTP " + resp.status);

      const total = Number(resp.headers.get("content-length")) || 0;
      const reader = resp.body.getReader();
      const chunks = [];
      let received = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        hooks.download(received, total);
      }

      const bytes = new Uint8Array(received);
      let pos = 0;
      for (const c of chunks) { bytes.set(c, pos); pos += c.length; }

      if (cache) {
        // the Response constructor copies a BufferSource body (Fetch spec,
        // "extract a body"), so no defensive bytes.slice() — that would be
        // an extra ~85MB transient allocation for the weights blob
        try { await cache.put(url, new Response(bytes, { headers: { "content-length": String(received) } })); }
        catch { /* quota */ }
      }
      return bytes.buffer;
    } catch (e) {
      lastErr = e;
      if (/HTTP 4\d\d/.test(String(e && e.message))) break; // e.g. 404 — don't retry, let caller fall back
    }
  }
  throw new Error("モデルのダウンロードに失敗しました: " + String((lastErr && lastErr.message) || lastErr));
}

/* ============================================================
   centerExtract — STFT soft-mask centre isolation (fallback)
   ============================================================ */
async function centerExtract(channels, hooks) {
  const L = channels[0];
  const R = channels[1] || channels[0];
  const N = L.length;

  const win = hann(FFT_SIZE);
  const out = new Float32Array(N);
  const norm = new Float32Array(N);

  const fft = new FFT(FFT_SIZE);
  const midRe = new Float32Array(FFT_SIZE), midIm = new Float32Array(FFT_SIZE);
  const sideRe = new Float32Array(FFT_SIZE), sideIm = new Float32Array(FFT_SIZE);
  const half = FFT_SIZE / 2;
  const eps = 1e-9;

  const frames = Math.ceil((N + FFT_SIZE) / HOP);
  let frame = 0, lastReport = -1;

  for (let start = -FFT_SIZE; start < N; start += HOP, frame++) {
    for (let i = 0; i < FFT_SIZE; i++) {
      const idx = start + i;
      let l = 0, r = 0;
      if (idx >= 0 && idx < N) { l = L[idx]; r = R[idx]; }
      const w = win[i];
      midRe[i] = 0.5 * (l + r) * w;
      sideRe[i] = 0.5 * (l - r) * w;
      midIm[i] = 0; sideIm[i] = 0;
    }

    fft.forward(midRe, midIm);
    fft.forward(sideRe, sideIm);

    for (let k = 0; k <= half; k++) {
      const m2 = midRe[k] * midRe[k] + midIm[k] * midIm[k];
      const s2 = sideRe[k] * sideRe[k] + sideIm[k] * sideIm[k];
      let mask = m2 / (m2 + s2 + eps);
      mask = mask * mask;
      midRe[k] *= mask; midIm[k] *= mask;
      if (k > 0 && k < half) { const j = FFT_SIZE - k; midRe[j] = midRe[k]; midIm[j] = -midIm[k]; }
    }

    fft.inverse(midRe, midIm);

    for (let i = 0; i < FFT_SIZE; i++) {
      const idx = start + i;
      if (idx < 0 || idx >= N) continue;
      const w = win[i];
      out[idx] += midRe[i] * w;
      norm[idx] += w * w;
    }

    if (frame % 8 === 0) {
      const p = Math.min(0.99, frame / frames);
      if (p - lastReport > 0.01) { hooks.progress(p); lastReport = p; await tick(); }
    }
  }

  for (let i = 0; i < N; i++) out[i] = norm[i] > eps ? out[i] / norm[i] : 0;
  hooks.progress(1);
  return { left: out, right: null };
}

/* ---------- helpers ---------- */
function hann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

/* ---------- compact radix-2 FFT (in-place) ---------- */
class FFT {
  constructor(n) {
    this.n = n;
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let x = i, y = 0;
      for (let b = 0; b < bits; b++) { y = (y << 1) | (x & 1); x >>= 1; }
      this.rev[i] = y;
    }
    this.cos = new Float32Array(n / 2);
    this.sin = new Float32Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / n);
    }
  }
  _transform(re, im, inv) {
    const n = this.n, rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          let wr = this.cos[k], wi = this.sin[k];
          if (inv) wi = -wi;
          const a = j, b = j + half;
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
      }
    }
  }
  forward(re, im) { this._transform(re, im, false); }
  inverse(re, im) { this._transform(re, im, true); const n = this.n; for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; } }
}
