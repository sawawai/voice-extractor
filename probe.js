/* ============================================================
   probe.js — shared by engine.js (browser) and
   scripts/make_probe_ref.mjs (Node).

   The WebGPU correctness probe: a deterministic "voice + music"
   test signal is separated once on the GPU and compared against a
   reference computed offline on the ONNX Runtime CPU EP (which is
   the proven-correct path). Genuine backend differences are tiny
   (~1e-3 relative); miscomputation (Firefox WebGPU, buggy drivers,
   fp16 overflow) is enormous — so a loose tolerance separates them
   cleanly and Math.sin ulp differences across JS engines don't
   matter.

   Also home to upcastFp16: the fp16 weights blob -> fp32 expansion
   that lets WebGPU run a pure-fp32 graph from an fp16-sized
   download.
   ============================================================ */

export const PROBE_SAMPLES = 343980;       // demucs-web TRAINING_SAMPLES: exactly one model segment
export const PROBE_DECIMATE = 16;          // reference stores every 16th output sample
export const PROBE_REF_FILE = "probe_ref_v1.bin"; // bump the name when the signal/model changes
export const PROBE_TOLERANCE = 0.25;       // rel L2; correct backends land ~1e-3..1e-2

/* Deterministic probe input: a swept harmonic "voice" in the centre
   (vibrato + syllable-rate envelope, so Demucs assigns it to vocals)
   over a side-panned chord + seeded noise "music" bed. No
   Math.random — must be reproducible in Node and every browser. */
export function makeProbeSignal(n = PROBE_SAMPLES) {
  const sr = 44100;
  const left = new Float32Array(n);
  const right = new Float32Array(n);

  let seed = 0x9e3779b9 | 0;
  const rand = () => {
    seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  let phase = 0;
  const chordF = [130.81, 164.81, 196.0]; // C3-E3-G3
  const chordPhase = [0, 0, 0];

  for (let i = 0; i < n; i++) {
    const t = i / sr;

    // "voice": f0 sweeps 90..210Hz with 5.5Hz vibrato, 10 harmonics, 2.6Hz syllables
    const f0 = 150 + 60 * Math.sin(2 * Math.PI * 0.35 * t) + 5 * Math.sin(2 * Math.PI * 5.5 * t);
    phase += (2 * Math.PI * f0) / sr;
    let v = 0;
    for (let k = 1; k <= 10; k++) v += Math.sin(phase * k) / k;
    const syll = Math.max(0, Math.sin(2 * Math.PI * 2.6 * t)) ** 0.5;
    v *= 0.2 * syll;

    // "music": sustained chord + noise, panned off-centre
    let mus = 0;
    for (let c = 0; c < 3; c++) {
      chordPhase[c] += (2 * Math.PI * chordF[c]) / sr;
      mus += Math.sin(chordPhase[c]);
    }
    mus *= 0.1;
    const noise = (rand() * 2 - 1) * 0.04;

    left[i] = v + mus * 0.9 + noise;
    right[i] = v + mus * 0.4 - noise;
  }
  return { left, right };
}

export function probeDecimate(arr) {
  const out = new Float32Array(Math.floor(arr.length / PROBE_DECIMATE));
  for (let i = 0; i < out.length; i++) out[i] = arr[i * PROBE_DECIMATE];
  return out;
}

/* ref layout: Float32Array [decimated left..., decimated right...] */
export function compareProbe(left, right, ref) {
  const half = ref.length >> 1;
  const dl = probeDecimate(left);
  const dr = probeDecimate(right);
  let num = 0, den = 0;
  for (let i = 0; i < half; i++) {
    const a = dl[i] - ref[i];
    const b = dr[i] - ref[half + i];
    num += a * a + b * b;
    den += ref[i] * ref[i] + ref[half + i] * ref[half + i];
  }
  const relErr = Math.sqrt(num / (den + 1e-12));
  return { ok: Number.isFinite(relErr) && relErr < PROBE_TOLERANCE, relErr };
}

/* fp16 -> fp32 expansion of the weights blob (~85MB -> ~170MB, well
   under a second). Float16Array where available, bit-twiddling
   fallback elsewhere. */
export function upcastFp16(buf) {
  const n = buf.byteLength >>> 1;
  const out = new Float32Array(n);
  if (typeof Float16Array !== "undefined") {
    out.set(new Float16Array(buf, 0, n));
    return out;
  }
  const u16 = new Uint16Array(buf, 0, n);
  const u32 = new Uint32Array(out.buffer);
  for (let i = 0; i < n; i++) {
    const h = u16[i];
    const s = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    if (e === 0) {
      if (m === 0) { u32[i] = s; continue; }        // signed zero
      do { m <<= 1; e--; } while ((m & 0x400) === 0); // subnormal -> normalize
      u32[i] = s | ((e + 113) << 23) | ((m & 0x3ff) << 13);
    } else if (e === 31) {
      u32[i] = s | 0x7f800000 | (m << 13);           // inf / nan
    } else {
      u32[i] = s | ((e + 112) << 23) | (m << 13);
    }
  }
  return out;
}
