import * as ort from "https://esm.sh/onnxruntime-web@1.20.1";

const MODEL_URL =
  "https://github.com/MegaCode111REAL/resonance/releases/download/yourmt3-browser-v1/yourmt3-mc13-forward.onnx";

const SAMPLE_RATE = 16000;
const SEGMENT_SAMPLES = 32767;
const CHANNELS = 13;
const START_TOKEN = 0;
const MAX_TOKENS = 256;

let sessionPromise = null;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

async function getSession() {
  if (!sessionPromise) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    sessionPromise = (async () => {
      post("progress", { value: 2, message: "Downloading YourMT3 browser model…" });
      const response = await fetch(MODEL_URL, { cache: "force-cache" });
      if (!response.ok) throw new Error("YourMT3 browser model download failed: HTTP " + response.status);

      const total = Number(response.headers.get("content-length")) || 0;
      const reader = response.body?.getReader();
      let bytes;
      if (reader) {
        const chunks = [];
        let loaded = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          loaded += value.byteLength;
          post("progress", {
            value: total ? 2 + 18 * loaded / total : 10,
            message: total
              ? "Downloading YourMT3 browser model… " +
                (loaded / 1048576).toFixed(0) + " / " +
                (total / 1048576).toFixed(0) + " MB"
              : "Downloading YourMT3 browser model…"
          });
        }
        bytes = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) {
          bytes.set(chunk, offset);
          offset += chunk.byteLength;
        }
      } else {
        bytes = new Uint8Array(await response.arrayBuffer());
      }

      post("progress", { value: 22, message: "Loading YourMT3 into WebAssembly…" });
      const session = await ort.InferenceSession.create(bytes.buffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
        enableCpuMemArena: false,
        enableMemPattern: false
      });
      post("progress", { value: 28, message: "YourMT3 browser model ready" });
      return session;
    })().catch(error => {
      sessionPromise = null;
      throw error;
    });
  }
  return sessionPromise;
}

function normalizeAudio(input) {
  const output = new Float32Array(SEGMENT_SAMPLES);
  const source = input instanceof Float32Array ? input : new Float32Array(input);
  const length = Math.min(source.length, output.length);
  output.set(source.subarray(0, length));
  return output;
}

function argmax(logits, offset, size) {
  let best = offset;
  let bestValue = logits[offset];
  for (let i = 1; i < size; i++) {
    const value = logits[offset + i];
    if (value > bestValue) {
      bestValue = value;
      best = offset + i;
    }
  }
  return best - offset;
}

async function decodeSegment(audio) {
  const session = await getSession();
  const normalized = normalizeAudio(audio);
  const audioTensor = new ort.Tensor("float32", normalized, [1, 1, SEGMENT_SAMPLES]);

  // Each channel starts with the T5 decoder start token. The model predicts
  // all 13 channels in parallel. We feed the complete prefix back on every
  // step; this is slower than KV-cache generation, but keeps the exported
  // graph portable and entirely browser-side.
  const tokens = new Int32Array(CHANNELS * 1);
  tokens.fill(START_TOKEN);

  for (let step = 0; step < MAX_TOKENS; step++) {
    const tokenTensor = new ort.Tensor(
      "int64",
      BigInt64Array.from(tokens, value => BigInt(value)),
      [1, CHANNELS, step + 1]
    );

    const outputs = await session.run({ audio: audioTensor, tokens: tokenTensor });
    const logits = outputs.logits.data;
    const vocabSize = outputs.logits.dims[3];
    const next = new Int32Array(CHANNELS);

    for (let channel = 0; channel < CHANNELS; channel++) {
      const offset = (channel * (step + 1) + step) * vocabSize;
      next[channel] = argmax(logits, offset, vocabSize);
    }

    const expanded = new Int32Array(CHANNELS * (step + 2));
    for (let channel = 0; channel < CHANNELS; channel++) {
      expanded.set(
        tokens.subarray(channel * (step + 1), (channel + 1) * (step + 1)),
        channel * (step + 2)
      );
      expanded[(channel + 1) * (step + 2) - 1] = next[channel];
    }
    tokens.set(expanded.subarray(0, tokens.length));
    if (step + 1 === MAX_TOKENS) {
      return { tokens: expanded, tokenLength: step + 2 };
    }
    // Rebind because the prefix grew.
    tokens.length = expanded.length;
  }

  return { tokens, tokenLength: MAX_TOKENS };
}

self.onmessage = async event => {
  if (event.data?.type !== "transcribe") return;

  try {
    post("progress", { value: 0, message: "Starting YourMT3 browser transcription…" });
    const { audio } = event.data;
    const result = await decodeSegment(audio);
    post("result", {
      tokens: Array.from(result.tokens),
      channels: CHANNELS,
      tokenLength: result.tokenLength,
      sampleRate: SAMPLE_RATE
    });
  } catch (error) {
    post("error", { message: error?.stack || error?.message || String(error) });
  }
};
