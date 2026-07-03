/* Copy onnxruntime-web's dist into ./ort/ (same-origin) so threaded WASM can
   spawn its pthread worker pool under cross-origin isolation.

   Run:
     npm i onnxruntime-web@1.21.0     # must match ORT_VER in worker.js
     node scripts/fetch-ort.mjs

   Writes ./ort/ (the runtime files) + ./ort/READY (a sentinel the worker probes
   to decide it can enable threads).
*/
import { cp, writeFile, readFile, access, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const srcDist = join(root, "node_modules", "onnxruntime-web", "dist");
const outDir = join(root, "ort");

try {
  await access(srcDist);
} catch {
  console.error(`Not found: ${srcDist}`);
  console.error("Run `npm i onnxruntime-web@1.21.0` first (match ORT_VER in worker.js).");
  process.exit(1);
}

await cp(srcDist, outDir, { recursive: true });

let ver = "unknown";
try {
  ver = JSON.parse(
    await readFile(join(root, "node_modules", "onnxruntime-web", "package.json"), "utf8")
  ).version;
} catch { /* ignore */ }

// Write the list of available ESM bundles (ort.*.mjs) to ort/READY, one per
// line. The worker picks which to import via ORT_BUNDLE_PREFERENCE (currently
// WASM-only, so the WebGPU EP can't run). Exclude ort-wasm-* runtime files
// (those are loaded by ORT itself via wasmPaths).
const files = await readdir(outDir);
const bundles = files.filter((f) => f.startsWith("ort.") && f.endsWith(".mjs"));

if (!bundles.length) {
  console.error("No ort.*.mjs ESM bundle found in dist — cannot write READY.");
  console.error("Files ending in .mjs:", files.filter((f) => f.endsWith(".mjs")).join(", "));
  process.exit(1);
}

await writeFile(join(outDir, "READY"), bundles.join("\n") + "\n");
console.log(`copied onnxruntime-web@${ver} dist -> ort/`);
console.log(`ORT ESM bundles available: ${bundles.join(", ")}`);
console.log(`(worker picks WASM-only first; wrote list to ort/READY)`);
if (ver !== "unknown") console.log(`Keep ORT_VER in worker.js ("${ver}") for the CDN fallback.`);
