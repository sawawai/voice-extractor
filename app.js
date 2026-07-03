/* ============================================================
   app.js — UI orchestration for 音声抽出
   Intake -> decode+resample(44.1k stereo) -> engine (demucs-web,
   main thread) -> WAV export -> playback. State machine drives
   the panels.
   ============================================================ */

import { runSeparation, verifyWebGPU } from "./engine.js";

const TARGET_SR = 44100; // demucs-web requirement

const stage = document.getElementById("stage");
const $ = (id) => document.getElementById(id);

const els = {
  fileInput: $("fileInput"),
  dropzone: $("dropzone"),
  fileName: $("fileName"),
  fileInfo: $("fileInfo"),
  clearFile: $("clearFile"),
  extractBtn: $("extractBtn"),
  optNormalize: $("optNormalize"),
  optMono: $("optMono"),
  optHighpass: $("optHighpass"),
  doneNote: $("doneNote"),
  workStatus: $("workStatus"),
  progressBar: $("progressBar"),
  progressFill: $("progressFill"),
  progressPct: $("progressPct"),
  progressStep: $("progressStep"),
  backendNote: $("backendNote"),
  downloadNote: $("downloadNote"),
  downloadBtn: $("downloadBtn"),
  resetBtn: $("resetBtn"),
  errorMsg: $("errorMsg"),
  errorReset: $("errorReset"),
};

let state = {
  file: null,
  audioBuffer: null,    // decoded + resampled (44.1k stereo)
  decodePromise: null,  // resolves to audioBuffer (or null on failure)
  decodeStatus: "idle", // "idle" | "loading" | "ready" | "failed"
  loadId: 0,            // bumped on pick/clear — stale decode callbacks self-discard
  inputUrl: null,       // object URL of the uploaded file (preview + compare)
  resultUrl: null,      // object URL of exported WAV
  trim: null,           // { a, b } seconds, or null = whole file
};

/* ---------- waveform players (preview / result / compare) ----------
   A hidden <audio> drives playback; the canvas draws peak bars so the
   input/output difference is visible at a glance. Redraws happen only
   on timeupdate/seek/drag/resize — no free-running animation.

   Every player shows a VIEW [va, vb] of the file (default: all of it):
   the waveform spans exactly the view, playback is confined to it, and
   the time readout is relative to it. Trimming is a mode: beginTrim()
   shows edge handles for a SELECTION inside the view; the app applies
   it by calling setView(selection), which zooms the waveform in — so
   trimming long files can be repeated recursively, each pass on a
   bigger waveform. */
const TRIM_MIN_SEC = 1;

/* Theme colours for the canvases — the custom properties never change at
   runtime, so read them once instead of on every animation frame. */
let _theme = null;
function themeColors() {
  if (!_theme) {
    const css = getComputedStyle(document.documentElement);
    _theme = {
      accent: css.getPropertyValue("--accent").trim() || "#bc3f1d",
      wave: css.getPropertyValue("--wave").trim() || "#d9d4c8",
    };
  }
  return _theme;
}

/* Size the backing store to CSS px × dpr and return a cleared 2D context,
   or null while the canvas is hidden (zero-sized). */
function canvasContext(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (!w || !h) return null;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h, dpr };
}

const prefersReducedMotion = () =>
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function makePlayer(rootId, opts = {}) {
  const root = $(rootId);
  const audio = root.querySelector("audio");
  const btn = root.querySelector(".player__btn");
  const canvas = root.querySelector(".player__wave");
  const timeEl = root.querySelector(".player__time");
  let data = null;              // full decoded channel, for (re)computing peaks
  let peaks = null;             // peaks of the current view
  let va = 0, vb = Infinity;    // view, absolute seconds
  let sa = 0, sb = Infinity;    // selection while trimming, absolute seconds
  let selecting = false;
  let dragging = null;          // "a" | "b" while a handle is held

  const dur = () => audio.duration || 0;
  const viewEnd = () => Math.min(vb, dur() || Infinity);
  // playback window = the selection while trimming, else the view
  const winStart = () => (selecting ? sa : va);
  const winEnd = () => (selecting ? Math.min(sb, dur()) : viewEnd());

  function recomputePeaks() {
    if (!data) { peaks = null; return; }
    if (!dur() || (va <= 0 && viewEnd() >= dur() - 0.01)) { peaks = computePeaks(data); return; }
    const i0 = Math.max(0, Math.floor((va / dur()) * data.length));
    const i1 = Math.min(data.length, Math.ceil((viewEnd() / dur()) * data.length));
    peaks = i1 > i0 ? computePeaks(data.subarray(i0, i1)) : null;
  }

  let loading = false;
  let loadingRaf = 0;
  let loadingT0 = 0;

  /* Decode-in-progress placeholder: the same bar floor a silent file
     would render as (minimum-height ticks), so the real peaks literally
     grow out of it when they arrive. A flat solid 朱 segment sweeps
     through — no gradient, and not before 2s so short decodes never
     flash a moving animation. Runs only while decoding, and
     decodeAudioData is off the main thread, so nothing can stutter it. */
  function drawLoadingFrame(now) {
    const c = canvasContext(canvas);
    if (!c) return;
    const { ctx, w, h, dpr } = c;
    const { accent, wave: rest } = themeColors();
    const bins = 160;
    const bw = w / bins;
    const bh = 2 * dpr; // the same floor height silent audio renders at
    const y = (h - bh) / 2;

    const elapsed = now - loadingT0;
    // sweep motion ported from screenshot-denoiser's #prog-bar.indeterminate:
    // a solid 28%-wide segment wipes left→right in 0.85s ease-in-out,
    // entering and exiting fully off the ends, with a short rest between
    // sweeps. Motion still waits 2s so short decodes never flash an animation.
    const SEG = 0.28;
    let segL = 2; // off-bar sentinel — no sweep drawn
    if (elapsed >= 2000) {
      const t = (elapsed - 2000) % 1200; // 0.85s sweep + 0.35s rest
      if (t < 850) {
        const e = 0.5 - 0.5 * Math.cos(Math.PI * (t / 850)); // ≈ CSS ease-in-out
        segL = -SEG + e * (1 + SEG);
      }
    }

    for (let b = 0; b < bins; b++) {
      const u = b / bins;
      ctx.fillStyle = u >= segL && u <= segL + SEG ? accent : rest;
      ctx.fillRect(b * bw + bw * 0.22, y, bw * 0.56, bh);
    }
  }

  function updateTimeReadout() {
    const D = dur();
    if (D) {
      const span = Math.max(viewEnd() - va, 1e-9);
      const wLen = Math.max(winEnd() - winStart(), 0);
      const rel = Math.min(Math.max(audio.currentTime - winStart(), 0), wLen);
      timeEl.textContent = `${fmtTime(rel, span)} / ${fmtTime(wLen, span)}`;
    } else {
      timeEl.textContent = "0:00";
    }
  }

  function render() {
    if (loading) {
      // placeholder owns the canvas; draw a frame here too so the static
      // (reduced-motion) placeholder appears once the panel is visible —
      // pinned to t0 there so it never shows a sweep
      updateTimeReadout();
      drawLoadingFrame(prefersReducedMotion() ? loadingT0 : performance.now());
      return;
    }
    const c = canvasContext(canvas);
    if (!c) return; // panel hidden — setPanel re-renders on show
    const { ctx, w, h, dpr } = c;
    const { accent, wave: rest } = themeColors();
    const D = dur();
    const span = Math.max(viewEnd() - va, 1e-9);

    if (peaks) {
      const bins = peaks.length;
      const bw = w / bins;
      for (let b = 0; b < bins; b++) {
        const bh = Math.max(2 * dpr, peaks[b] * h * 0.9);
        // played-colour only from the window start, and only during real
        // playback while trimming — handle-drags can't paint the waveform
        const t = va + ((b + 0.5) / bins) * span;
        const played = D && !(selecting && audio.paused) && t >= winStart() && t <= audio.currentTime;
        ctx.fillStyle = played ? accent : rest;
        ctx.fillRect(b * bw + bw * 0.22, (h - bh) / 2, bw * 0.56, bh);
      }
    }

    if (selecting && D) {
      const xa = ((sa - va) / span) * w;
      const xb = ((Math.min(sb, viewEnd()) - va) / span) * w;
      // dim outside the selection
      ctx.fillStyle = "rgba(247, 245, 240, 0.78)";
      ctx.fillRect(0, 0, xa, h);
      ctx.fillRect(xb, 0, w - xb, h);
      // edge handles: line + grip
      ctx.fillStyle = accent;
      for (const x of [xa, xb]) {
        ctx.fillRect(Math.min(Math.max(x - dpr, 0), w - 2 * dpr), 0, 2 * dpr, h);
        const gw = 7 * dpr, gh = h * 0.44;
        ctx.fillRect(Math.min(Math.max(x - gw / 2, 0), w - gw), (h - gh) / 2, gw, gh);
      }
    }

    // padded against the view length, which is constant while dragging a
    // selection — the readout never changes width mid-drag
    updateTimeReadout();
  }

  btn.addEventListener("click", () => {
    if (!audio.src) return;
    if (audio.paused) {
      if (audio.currentTime < winStart() || audio.currentTime >= winEnd() - 0.05) {
        audio.currentTime = winStart();
      }
      audio.play();
    } else {
      audio.pause();
    }
  });
  audio.addEventListener("play", () => root.classList.add("is-playing"));
  audio.addEventListener("pause", () => root.classList.remove("is-playing"));
  audio.addEventListener("ended", () => root.classList.remove("is-playing"));
  audio.addEventListener("timeupdate", () => {
    // stop at the window edge (natural `ended` covers winEnd == duration)
    if (!audio.paused && dur() && winEnd() < dur() - 0.01 && audio.currentTime >= winEnd() - 0.02) {
      audio.pause();
      audio.currentTime = winStart();
    }
    render();
  });
  audio.addEventListener("seeked", render);
  audio.addEventListener("loadedmetadata", () => { recomputePeaks(); render(); });
  window.addEventListener("resize", render);

  const xToTime = (x) =>
    va + (Math.min(Math.max(x, 0), canvas.clientWidth) / canvas.clientWidth) * (viewEnd() - va);
  const timeToX = (t) => ((t - va) / Math.max(viewEnd() - va, 1e-9)) * canvas.clientWidth;
  const HIT = 12; // CSS px around a handle that grabs it

  canvas.addEventListener("pointerdown", (e) => {
    if (loading || !dur()) return;
    if (selecting) {
      const dA = Math.abs(e.offsetX - timeToX(sa));
      const dB = Math.abs(e.offsetX - timeToX(Math.min(sb, viewEnd())));
      if (dA <= HIT || dB <= HIT) {
        dragging = dA <= dB ? "a" : "b";
        canvas.setPointerCapture(e.pointerId);
        e.preventDefault();
        return;
      }
    }
    audio.currentTime = Math.min(Math.max(xToTime(e.offsetX), winStart()), Math.max(winStart(), winEnd() - 0.05));
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dur()) return;
    if (dragging) {
      const t = xToTime(e.offsetX);
      const minLen = Math.min(TRIM_MIN_SEC, (viewEnd() - va) / 4);
      if (dragging === "a") sa = Math.min(Math.max(va, t), sb - minLen);
      else sb = Math.max(Math.min(viewEnd(), t), sa + minLen);
      if (audio.currentTime < sa || audio.currentTime > sb) audio.currentTime = sa;
      render();
      if (opts.onTrim) opts.onTrim(sa, Math.min(sb, viewEnd()));
    } else if (selecting) {
      const near =
        Math.abs(e.offsetX - timeToX(sa)) <= HIT ||
        Math.abs(e.offsetX - timeToX(Math.min(sb, viewEnd()))) <= HIT;
      canvas.style.cursor = near ? "ew-resize" : "pointer";
    }
  });
  for (const ev of ["pointerup", "pointercancel"]) {
    canvas.addEventListener(ev, () => { dragging = null; });
  }

  return {
    audio,
    render,
    setView(a, b) {
      va = Math.max(0, a); vb = b;
      selecting = false; dragging = null;
      canvas.style.cursor = "pointer";
      if (dur() && (audio.currentTime < winStart() || audio.currentTime > winEnd())) {
        audio.currentTime = winStart();
      }
      recomputePeaks();
      render();
    },
    beginTrim() {
      if (!dur() || viewEnd() - va <= TRIM_MIN_SEC * 2) return false;
      selecting = true;
      sa = va; sb = viewEnd();
      audio.pause();
      audio.currentTime = va; // park the playhead: no leftover played colour
      render();
      return true;
    },
    cancelTrim() {
      selecting = false; dragging = null;
      canvas.style.cursor = "pointer";
      render();
    },
    getSelection() {
      return { a: sa, b: Math.min(sb, viewEnd()) };
    },
    isTrimming() { return selecting; },
    setSource(url, channelData) {
      if (url) audio.src = url; else audio.removeAttribute("src");
      va = 0; vb = Infinity;
      selecting = false; dragging = null;
      this.setData(channelData);
    },
    setData(channelData) {
      data = channelData || null;
      recomputePeaks();
      render();
    },
    setLoading(on) {
      on = !!on;
      if (loading === on) return;
      loading = on;
      btn.disabled = on; // no playback until the waveform is in
      cancelAnimationFrame(loadingRaf);
      if (on) {
        loadingT0 = performance.now();
        if (prefersReducedMotion()) {
          drawLoadingFrame(loadingT0); // one static placeholder frame, no sweep
        } else {
          const step = (now) => {
            if (!loading) return;
            drawLoadingFrame(now);
            loadingRaf = requestAnimationFrame(step);
          };
          loadingRaf = requestAnimationFrame(step);
        }
      } else {
        render();
      }
    },
    reset() {
      audio.pause();
      audio.removeAttribute("src");
      root.classList.remove("is-playing");
      loading = false;
      btn.disabled = false;
      cancelAnimationFrame(loadingRaf);
      data = null; peaks = null;
      va = 0; vb = Infinity; selecting = false; dragging = null;
      timeEl.textContent = "0:00";
      render();
    },
  };
}

function computePeaks(data, bins = 160) {
  const step = Math.floor(data.length / bins) || 1;
  const out = new Float32Array(bins);
  for (let b = 0; b < bins; b++) {
    let peak = 0;
    const end = Math.min((b + 1) * step, data.length);
    for (let i = b * step; i < end; i++) {
      const v = Math.abs(data[i]);
      if (v > peak) peak = v;
    }
    out[b] = Math.min(1, peak); // raw amplitude — quiet output SHOULD look quiet
  }
  return out;
}

const players = {
  preview: makePlayer("playerPreview", { onTrim: refreshTrimInfo }),
  out: makePlayer("playerOut"),
  compare: makePlayer("playerCompare"),
};

/* ---------- input trim (mode with apply/cancel, repeatable) ----------
   範囲を切り出す shows the handles; 適用 zooms the preview into the
   selection (so a long file can be trimmed again on the zoomed
   waveform); 切り出しを解除 goes back to the whole file. Only the
   APPLIED range (state.trim) is extracted. */
const trimEls = {
  info: $("trimInfo"),
  start: $("trimStart"),
  apply: $("trimApply"),
  cancel: $("trimCancel"),
  clear: $("trimClear"),
};

function setTrimUI() {
  const trimming = players.preview.isTrimming();
  trimEls.start.hidden = trimming;
  trimEls.start.disabled = state.decodeStatus !== "ready"; // no trimming before the waveform exists
  trimEls.apply.hidden = !trimming;
  trimEls.cancel.hidden = !trimming;
  trimEls.clear.hidden = !state.trim; // stays visible while re-trimming an applied trim
}

function refreshTrimInfo() {
  const dur = players.preview.audio.duration || 0;
  if (players.preview.isTrimming()) {
    const s = players.preview.getSelection();
    trimEls.info.textContent = `${fmtTime(s.a, dur)} – ${fmtTime(s.b, dur)}（${fmtTime(s.b - s.a, dur)}）を選択中`;
    trimEls.info.classList.add("is-active");
    return;
  }
  trimEls.info.classList.toggle("is-active", !!state.trim);
  if (state.trim) {
    trimEls.info.textContent = `${fmtTime(state.trim.a, dur)} – ${fmtTime(state.trim.b, dur)} を抽出します`;
  } else if (state.decodeStatus === "loading") {
    trimEls.info.textContent = "波形を解析しています…";
  } else if (state.decodeStatus === "ready") {
    trimEls.info.textContent = "波形の解析が完了しました";
  } else if (state.decodeStatus === "failed") {
    trimEls.info.textContent = "波形を解析できませんでした";
  } else {
    trimEls.info.textContent = "";
  }
}

function syncTrimUI() {
  refreshTrimInfo();
  setTrimUI();
}

trimEls.start.addEventListener("click", () => {
  if (!players.preview.beginTrim()) return; // no metadata yet / too short
  syncTrimUI();
});
trimEls.apply.addEventListener("click", () => {
  const sel = players.preview.getSelection();
  players.preview.setView(sel.a, sel.b);
  const dur = players.preview.audio.duration || 0;
  state.trim = sel.a <= 0.01 && sel.b >= dur - 0.01 ? null : { a: sel.a, b: sel.b };
  syncTrimUI();
});
trimEls.cancel.addEventListener("click", () => {
  players.preview.cancelTrim();
  syncTrimUI();
});
trimEls.clear.addEventListener("click", () => {
  players.preview.setView(0, Infinity);
  state.trim = null;
  syncTrimUI();
});

// only one player at a time (preview vs result-vs-original comparison)
for (const p of Object.values(players)) {
  p.audio.addEventListener("play", () => {
    for (const q of Object.values(players)) if (q !== p) q.audio.pause();
  });
}

/* ---------- state machine ---------- */
function setPanel(name) {
  stage.dataset.state = name;
  for (const p of document.querySelectorAll("[data-panel]")) {
    p.hidden = p.dataset.panel !== name;
  }
  // canvases can't draw while hidden — redraw whichever just became visible
  requestAnimationFrame(() => { for (const p of Object.values(players)) p.render(); });
}

/* ---------- file intake ---------- */
function onFilePicked(file) {
  if (!file) return;
  if (
    !/^(audio|video)\//.test(file.type) &&
    !/\.(wav|mp3|m4a|flac|ogg|oga|aac|opus|weba|mp4|m4v|webm|mov|mkv)$/i.test(file.name)
  ) {
    return showError("対応していないファイル形式です。音声ファイルまたは動画ファイルをお試しください。");
  }
  state.file = file;
  els.fileName.textContent = file.name;
  els.fileInfo.textContent = formatBytes(file.size);
  if (state.inputUrl) URL.revokeObjectURL(state.inputUrl);
  state.inputUrl = URL.createObjectURL(file);
  players.preview.setSource(state.inputUrl, null);
  state.trim = null;
  // decode now (instead of at extraction) so the preview gets its waveform;
  // failures surface when 抽出する is pressed. decodeAudioData can't be
  // aborted, so a stale decode (file cleared / replaced meanwhile) is
  // detected via loadId and discards itself instead of clobbering the UI.
  const loadId = ++state.loadId;
  players.preview.setLoading(true);
  state.decodeStatus = "loading";
  syncTrimUI();
  state.audioBuffer = null;
  state.decodePromise = decodeFile(file)
    .catch(() => null)
    .then((buf) => {
      if (loadId !== state.loadId) return null; // stale — a newer file (or none) took over
      if (buf) {
        state.audioBuffer = buf;
        players.preview.setData(buf.getChannelData(0));
      }
      players.preview.setLoading(false);
      state.decodeStatus = buf ? "ready" : "failed";
      syncTrimUI(); // 解析しています… → 完了しました; enables 範囲を切り出す
      return buf;
    });
  setPanel("ready");
}

els.fileInput.addEventListener("change", (e) => onFilePicked(e.target.files[0]));

const dz = els.dropzone;
["dragenter", "dragover"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-drag"); })
);
dz.addEventListener("drop", (e) => onFilePicked(e.dataTransfer.files[0]));
dz.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); }
});

els.clearFile.addEventListener("click", resetAll);

/* ---------- extraction ---------- */
els.extractBtn.addEventListener("click", startExtraction);

/* Demucs needs more memory than mobile browsers give a tab (iOS Safari
   kills the page mid-inference even on short clips), so phones/tablets
   are shown a blocking notice at page load instead of the tool. */
const IS_MOBILE = (() => {
  try {
    if (navigator.userAgentData && navigator.userAgentData.mobile) return true;
    if (/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) return true;
    return /Mac/.test(navigator.userAgent) && navigator.maxTouchPoints > 1; // iPadOS reports as macOS
  } catch {
    return false;
  }
})();

async function startExtraction() {
  if (players.preview.isTrimming()) { // unapplied selection is not extracted
    players.preview.cancelTrim();
    syncTrimUI();
  }
  players.preview.audio.pause();
  // the format is now actually being used — make it the default for next visit
  try { localStorage.setItem("voiceext:format", getFormat()); } catch { /* ignore */ }
  setPanel("work");
  setProgress(0);
  els.workStatus.textContent = "音声を読み込んでいます…";
  els.backendNote.textContent = "";
  els.downloadNote.textContent = "";
  els.doneNote.textContent = "";

  try {
    const buf = await state.decodePromise; // 44.1k stereo, started at file pick
    if (!buf) throw new Error("音声を読み込めませんでした。対応していない形式か、ファイルが壊れている可能性があります。");
    els.workStatus.textContent = "声を抽出しています…";

    let channels = [buf.getChannelData(0), buf.getChannelData(1)];
    if (state.trim) {
      const s = Math.floor(state.trim.a * buf.sampleRate);
      const e = Math.min(buf.length, Math.ceil(state.trim.b * buf.sampleRate));
      channels = channels.map((c) => c.subarray(s, e));
    }

    const res = await runInference(channels, buf.sampleRate);

    els.workStatus.textContent = "仕上げ処理をしています…";
    await new Promise((r) => setTimeout(r, 0)); // let the status paint before the sync DSP/encode below

    const outChannels = buildOutput(
      res, els.optMono.checked, els.optNormalize.checked, els.optHighpass.checked, buf.sampleRate
    );

    let blob, ext;
    if (getFormat() === "mp3") {
      els.workStatus.textContent = "MP3 に変換しています…";
      setProgress(0);
      try {
        blob = await encodeMp3(outChannels, buf.sampleRate);
        setProgress(1);
        ext = "mp3";
      } catch (err) {
        console.warn("MP3 encode failed — falling back to WAV:", err);
        els.doneNote.textContent = "MP3 への変換に失敗したため、WAV で保存します。";
        blob = encodeWav(outChannels, buf.sampleRate);
        ext = "wav";
      }
    } else {
      blob = encodeWav(outChannels, buf.sampleRate);
      ext = "wav";
    }
    if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
    state.resultUrl = URL.createObjectURL(blob);

    presentResult(outChannels[0], ext);
  } catch (err) {
    console.error(err);
    showError(err && err.message ? err.message : "処理中に問題が発生しました。");
  }
}

function uiHooks() {
  return {
    status: (text) => { els.workStatus.textContent = text; },
    progress: (p, seg, total) => {
      setProgress(p, seg, total);
      els.downloadNote.textContent = "";
      // segments done — what remains is assembly/encode, not extraction
      if (p >= 0.999) els.workStatus.textContent = "仕上げ処理をしています…";
    },
    download: (loaded, total) => {
      const mb = (loaded / 1048576).toFixed(0);
      // loaded can pass total if a proxy reports a stale/compressed size —
      // drop to the size-unknown display instead of showing >100%
      if (total && loaded <= total) {
        const pct = Math.round((loaded / total) * 100);
        els.downloadNote.textContent = `モデルをダウンロード中… ${pct}%（${mb} / ${(total / 1048576).toFixed(0)} MB・初回のみ）`;
      } else {
        els.downloadNote.textContent = `モデルをダウンロード中… ${mb} MB（初回のみ）`;
      }
    },
  };
}

// WebGPU runs on the main thread (garbles in a worker); WASM runs in a worker
// (so demucs-web's synchronous STFT can't block the UI / trip Firefox's
// slow-script watchdog). verifyWebGPU separates a known test signal on the
// GPU once (verdict cached) so miscomputing GPUs — Firefox's WebGPU, buggy
// drivers — fall back to WASM automatically instead of shipping garbled audio.
async function runInference(channels, sampleRate) {
  const hooks = uiHooks();
  // definitive only now — detection alone can't tell (e.g. Firefox detects
  // WebGPU but fails the correctness probe and runs on CPU)
  if (await verifyWebGPU(hooks)) {
    els.backendNote.textContent = "WebGPU（GPU）で処理しています";
    return runSeparation(channels, sampleRate, hooks, { backend: "webgpu" });
  }
  els.backendNote.textContent = "CPU で処理しています（WebGPU が利用できないため、時間がかかります）";
  return runInWorker(channels, sampleRate, hooks);
}

function runInWorker(channels, sampleRate, hooks) {
  return new Promise((resolve, reject) => {
    const worker = new Worker("worker.js", { type: "module" });
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "status") hooks.status(m.text);
      else if (m.type === "progress") hooks.progress(m.value, m.seg, m.total);
      else if (m.type === "download") hooks.download(m.loaded, m.total);
      else if (m.type === "done") {
        worker.terminate();
        resolve({ left: new Float32Array(m.left), right: m.right ? new Float32Array(m.right) : null });
      } else if (m.type === "error") {
        worker.terminate();
        reject(new Error(m.message));
      }
    };
    worker.onerror = (e) => { worker.terminate(); reject(new Error(e.message)); };

    // copy channels so transferring them doesn't detach the AudioBuffer
    // (still needed for the result waveform).
    const copies = channels.map((c) => c.slice());
    worker.postMessage({ type: "separate", channels: copies, sampleRate }, copies.map((c) => c.buffer));
  });
}

/* ---------- decode + resample to 44.1k stereo ---------- */
async function decodeFile(file) {
  const arrayBuf = await file.arrayBuffer();
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(arrayBuf);
  } finally {
    ctx.close();
  }
  return resampleToTarget(decoded);
}

async function resampleToTarget(buf) {
  const needResample = buf.sampleRate !== TARGET_SR;
  const needStereo = buf.numberOfChannels < 2;
  if (!needResample && !needStereo) return buf;

  const length = Math.ceil(buf.duration * TARGET_SR);
  const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const octx = new OAC(2, length, TARGET_SR); // 2ch dest upmixes mono automatically
  const src = octx.createBufferSource();
  src.buffer = buf;
  src.connect(octx.destination);
  src.start();
  return await octx.startRendering();
}

/* ---------- progress ---------- */
function setProgress(p, seg, total) {
  const pct = Math.round(Math.max(0, Math.min(1, p)) * 100); // demucs-web can overshoot 100%
  els.progressFill.style.width = pct + "%";
  els.progressPct.textContent = pct;
  els.progressBar.setAttribute("aria-valuenow", pct);
  // zero-pad the current segment to the total's width (9/20 → 09/20) so the
  // readout never changes size mid-run — same treatment as the time display
  els.progressStep.textContent =
    seg && total ? `区間 ${String(seg).padStart(String(total).length, "0")} / ${total}` : "";
}

/* ---------- assemble output channels ---------- */
function buildOutput(res, mono, doNormalize, doHighpass, sampleRate) {
  const L = res.left;
  const R = res.right || res.left; // fallback engine returns mono (right=null)

  let channels;
  if (mono) {
    const m = new Float32Array(L.length);
    for (let i = 0; i < L.length; i++) m[i] = (L[i] + R[i]) * 0.5;
    channels = [m];
  } else {
    // ensure independent buffers if right was aliased to left
    channels = [L, res.right ? R : R.slice()];
  }
  // filter BEFORE normalizing so subsonic junk doesn't eat the gain headroom
  if (doHighpass) for (const ch of channels) highpassInPlace(ch, sampleRate);
  if (doNormalize) normalizeChannels(channels);
  return channels;
}

/* 2nd-order Butterworth high-pass (RBJ biquad), fc ≈ 30Hz: removes DC
   offset (0Hz by definition) and subsonic rumble without touching the
   voice band. */
function highpassInPlace(x, fs, fc = 30) {
  const w0 = (2 * Math.PI * fc) / fs;
  const cw = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
  const a0 = 1 + alpha;
  const b0 = (1 + cw) / 2 / a0;
  const b1 = -(1 + cw) / a0;
  const b2 = b0;
  const a1 = (-2 * cw) / a0;
  const a2 = (1 - alpha) / a0;
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = xi;
    y2 = y1; y1 = yi;
    x[i] = yi;
  }
}

/* ---------- result presentation ---------- */
/* MP3 encode runs in a worker (lamejs is sync JS — a long encode would
   freeze the page). Channels are copied+transferred; the originals stay
   usable for the result waveform. */
function encodeMp3(channels, sampleRate) {
  return new Promise((resolve, reject) => {
    const w = new Worker("mp3-worker.js", { type: "module" });
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") setProgress(m.value);
      else if (m.type === "done") { w.terminate(); resolve(m.blob); }
      else if (m.type === "error") { w.terminate(); reject(new Error(m.message)); }
    };
    w.onerror = (err) => { w.terminate(); reject(new Error(err.message || "MP3 worker error")); };
    const copies = channels.map((c) => c.slice());
    w.postMessage({ channels: copies, sampleRate, kbps: 192 }, copies.map((c) => c.buffer));
  });
}

function presentResult(resultChannel, ext) {
  players.out.setSource(state.resultUrl, resultChannel);
  // original alongside the result, zoomed to the trim so both waveforms
  // span exactly the processed range
  players.compare.setSource(state.inputUrl, state.audioBuffer ? state.audioBuffer.getChannelData(0) : null);
  if (state.trim) players.compare.setView(state.trim.a, state.trim.b);
  const base = state.file.name.replace(/\.[^.]+$/, "");
  els.downloadBtn.href = state.resultUrl;
  els.downloadBtn.download = `${base}_声.${ext || "wav"}`;
  setPanel("done");
}

/* ---------- reset / error ---------- */
els.resetBtn.addEventListener("click", resetAll);
els.errorReset.addEventListener("click", resetAll);

function resetAll() {
  state.loadId++; // orphan any in-flight decode so it can't repaint the preview
  if (state.resultUrl) { URL.revokeObjectURL(state.resultUrl); state.resultUrl = null; }
  if (state.inputUrl) { URL.revokeObjectURL(state.inputUrl); state.inputUrl = null; }
  for (const p of Object.values(players)) p.reset();
  els.fileInput.value = "";
  state.file = null; state.audioBuffer = null; state.decodePromise = null; state.trim = null;
  state.decodeStatus = "idle";
  syncTrimUI();
  setPanel("idle");
}

function showError(msg) {
  els.errorMsg.textContent = msg;
  setPanel("error");
}

/* ============================================================
   DSP + encoding
   ============================================================ */
function normalizeChannels(channels) {
  let peak = 0;
  for (const ch of channels)
    for (let i = 0; i < ch.length; i++) { const a = Math.abs(ch[i]); if (a > peak) peak = a; }
  if (peak < 1e-6) return;
  const gain = 0.97 / peak; // shared gain preserves stereo balance
  for (const ch of channels)
    for (let i = 0; i < ch.length; i++) ch[i] *= gain;
}

function encodeWav(channels, sampleRate) {
  const numCh = channels.length;
  const n = channels[0].length;
  const blockAlign = numCh * 2;      // 16-bit
  const dataLen = n * blockAlign;
  const buffer = new ArrayBuffer(44 + dataLen);
  const view = new DataView(buffer);
  const wr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };

  wr(0, "RIFF");
  view.setUint32(4, 36 + dataLen, true);
  wr(8, "WAVE");
  wr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);              // PCM
  view.setUint16(22, numCh, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  wr(36, "data");
  view.setUint32(40, dataLen, true);

  // Interleave samples through an aligned Int16Array view (offset 44 is
  // even) — much faster than per-sample DataView calls. WAV wants
  // little-endian, which is what every platform a browser runs on uses
  // (WASM — required by this app — mandates LE memory).
  const pcm = new Int16Array(buffer, 44);
  let off = 0;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < numCh; c++) {
      const s = Math.max(-1, Math.min(1, channels[c][i]));
      pcm[off++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/* ---------- formatters ---------- */
function formatBytes(b) {
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(0) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}
/* Format seconds like the reference duration `ref` (usually the file
   length): H:MM:SS when ref is an hour or more, else M:SS — with the
   leading field zero-padded to ref's width so "09:59 / 20:00" and
   "10:00 / 20:00" are the same length and the layout never shifts. */
function fmtTime(s, ref) {
  s = Math.max(0, Math.floor(s || 0));
  const r = Math.max(Math.floor(ref || 0), s);
  const sec = String(s % 60).padStart(2, "0");
  if (r >= 3600) {
    const h = String(Math.floor(s / 3600)).padStart(String(Math.floor(r / 3600)).length, "0");
    const min = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    return `${h}:${min}:${sec}`;
  }
  const min = String(Math.floor(s / 60)).padStart(String(Math.floor(r / 60)).length || 1, "0");
  return `${min}:${sec}`;
}

/* ---------- boot ---------- */
// We no longer use COEP, so unregister any coi-serviceworker left over from
// earlier versions — it re-streams every fetch and breaks large downloads.
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.getRegistrations().then((regs) => {
    if (!regs.length) return;
    Promise.all(regs.map((r) => r.unregister())).then(() => {
      if (navigator.serviceWorker.controller) location.reload();
    });
  }).catch(() => {});
}

// output format: WAV/MP3 segmented toggle. Remembered across visits, but
// only a choice actually used for an extraction is saved (startExtraction) —
// an idly clicked toggle doesn't change future visits' default.
const fmtButtons = Array.from(document.querySelectorAll("#optFormat .seg__btn"));

function getFormat() {
  const on = fmtButtons.find((b) => b.classList.contains("is-on"));
  return on ? on.dataset.value : "wav";
}
function setFormat(v) {
  for (const b of fmtButtons) {
    const on = b.dataset.value === v;
    b.classList.toggle("is-on", on);
    b.setAttribute("aria-pressed", String(on));
  }
}
for (const b of fmtButtons) {
  b.addEventListener("click", () => setFormat(b.dataset.value));
}
try {
  if (localStorage.getItem("voiceext:format") === "mp3") setFormat("mp3");
} catch { /* no localStorage */ }

// mobile browsers can't run the separation model (per-tab memory limits kill
// the page mid-inference) — show the notice instead of the tool
setPanel(IS_MOBILE ? "mobile" : "idle");
