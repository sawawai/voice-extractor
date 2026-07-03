# models/

Shipping artifacts (everyone downloads ~85 MB total, both backends):

    htdemucs_graph_fp32.onnx   (~2.6 MB)  WebGPU graph — pure-fp32 compute
    htdemucs_graph_fp16.onnx   (~2.6 MB)  WASM graph
    weights_fp16.bin           (~85 MB)   shared external weights blob
    probe_ref_v1.bin           (~170 KB)  GPU correctness-probe reference

Both graphs reference the same blob: the WebGPU path upcasts it to fp32 in JS
(fp16 *compute* garbles on WebGPU — ORT #26732 — but fp16 *storage* is safe),
WASM uses it directly.

Build them from demucs-web's fp32 model (see repo README for details):

```bash
pip install onnx onnxconverter-common onnxruntime numpy
curl -L -o htdemucs_embedded.onnx \
  https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx
python ../scripts/convert_external_fp16.py htdemucs_embedded.onnx   # run from repo root
node ../scripts/make_probe_ref.mjs                                  # run from repo root
```

`htdemucs_embedded.onnx` (~172 MB) and `htdemucs_embedded_fp16.onnx` (~86 MB)
are the build input / legacy runtime fallback — keep them locally for
rebuilds, but they don't need to be deployed.

Fetched same-origin, streamed with progress, Cache-API cached (downloads once).
