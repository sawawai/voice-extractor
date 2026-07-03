/* ============================================================
   make_probe_ref.mjs — generate models/probe_ref_v1.bin, the
   ground-truth output for the browser's WebGPU correctness probe.

   Runs the SAME pipeline the browser runs (demucs-web separate()
   over the fp32 graph + upcast fp16 weights), but on onnxruntime-
   node's CPU EP — the numerically trusted path. Also cross-checks
   against the original monolithic fp32 model to prove the split
   artifacts + JS upcast are sound end-to-end.

   Usage:
     npm i            (installs demucs-web + onnxruntime-node dev deps)
     python scripts/convert_external_fp16.py   (must run first)
     node scripts/make_probe_ref.mjs
   ============================================================ */
import { readFileSync, writeFileSync, rmSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ort from "onnxruntime-node";
import { DemucsProcessor } from "demucs-web";
import {
  PROBE_SAMPLES, PROBE_REF_FILE, makeProbeSignal, probeDecimate, upcastFp16,
} from "../probe.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODELS = path.join(root, "models");
const GRAPH_FP32 = path.join(MODELS, "htdemucs_graph_fp32.onnx");
const BLOB_FP16 = path.join(MODELS, "weights_fp16.bin");
const BLOB_FP32 = path.join(MODELS, "weights_fp32.bin"); // temp, deleted after
const ORIGINAL = path.join(MODELS, "htdemucs_embedded.onnx");
const OUT = path.join(MODELS, PROBE_REF_FILE);

async function separateWith(modelPath) {
  const session = await ort.InferenceSession.create(modelPath, {
    graphOptimizationLevel: "basic", // mirror demucs-web's browser default
  });
  const processor = new DemucsProcessor({ ort });
  processor.session = session; // bypass loadModel: Node resolves external data from disk by path
  const { left, right } = makeProbeSignal(PROBE_SAMPLES);
  const t0 = Date.now();
  const res = await processor.separate(left, right);
  console.log(`  separated with ${path.basename(modelPath)} in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  if (session.release) await session.release();
  return res.vocals;
}

function relL2(a, b) {
  let num = 0, den = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
  return Math.sqrt(num / (den + 1e-12));
}
const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);

if (!existsSync(GRAPH_FP32) || !existsSync(BLOB_FP16)) {
  console.error("split artifacts missing — run scripts/convert_external_fp16.py first");
  process.exit(1);
}

// Reconstruct weights_fp32.bin exactly as the browser does (JS upcast), so this
// run also end-to-end-tests upcastFp16 + the fp32 graph.
const blob16 = readFileSync(BLOB_FP16);
const f32 = upcastFp16(blob16.buffer.slice(blob16.byteOffset, blob16.byteOffset + blob16.byteLength));
writeFileSync(BLOB_FP32, Buffer.from(f32.buffer));

try {
  console.log("reference run: fp32 graph + upcast(fp16 blob), CPU EP ...");
  const vocals = await separateWith(GRAPH_FP32);

  const dl = probeDecimate(vocals.left);
  const dr = probeDecimate(vocals.right);
  const out = new Float32Array(dl.length + dr.length);
  out.set(dl, 0);
  out.set(dr, dl.length);
  writeFileSync(OUT, Buffer.from(out.buffer));
  console.log(`wrote ${OUT} (${(out.byteLength / 1024).toFixed(0)} KB), vocals RMS=${rms(dl).toFixed(4)}`);
  if (rms(dl) < 1e-3) {
    console.error("vocals output is near-silent — probe signal needs a stronger voice component");
    process.exit(1);
  }

  if (existsSync(ORIGINAL)) {
    console.log("cross-check: original monolithic fp32 model ...");
    const ref = await separateWith(ORIGINAL);
    const err = relL2(vocals.left, ref.left);
    console.log(`  rel L2 err (split+upcast vs original) = ${err.toFixed(5)}`);
    if (!(err < 0.05)) {
      console.error("CROSS-CHECK FAILED: split artifacts diverge from the original model");
      process.exit(1);
    }
  } else {
    console.log("(original model not present — skipping cross-check)");
  }
  console.log("OK");
} finally {
  rmSync(BLOB_FP32, { force: true }); // never shipped; the browser builds it in memory
}
