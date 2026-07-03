/* ============================================================
   mp3-worker.js — encodes the extracted audio to MP3 off the main
   thread (lamejs is pure-JS and synchronous; a long encode would
   otherwise freeze the page). Loaded lazily — only when the user
   picks MP3 as the output format.
   ============================================================ */
import * as lame from "https://esm.sh/@breezystack/lamejs@1.2.7";

const Mp3Encoder = lame.Mp3Encoder || (lame.default && lame.default.Mp3Encoder);

function toInt16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

self.onmessage = (e) => {
  const { channels, sampleRate, kbps } = e.data;
  try {
    if (!Mp3Encoder) throw new Error("MP3 encoder unavailable");
    const enc = new Mp3Encoder(channels.length, sampleRate, kbps);
    const L = toInt16(channels[0]);
    const R = channels[1] ? toInt16(channels[1]) : null;
    const BLOCK = 1152;
    const parts = [];
    let lastP = 0;
    for (let i = 0; i < L.length; i += BLOCK) {
      const l = L.subarray(i, i + BLOCK);
      const buf = R ? enc.encodeBuffer(l, R.subarray(i, i + BLOCK)) : enc.encodeBuffer(l);
      if (buf.length) parts.push(buf);
      const p = i / L.length;
      if (p - lastP >= 0.01) {
        postMessage({ type: "progress", value: p });
        lastP = p;
      }
    }
    const end = enc.flush();
    if (end.length) parts.push(end);
    postMessage({ type: "done", blob: new Blob(parts, { type: "audio/mpeg" }) });
  } catch (err) {
    postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
