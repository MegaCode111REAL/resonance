import * as ort from "https://esm.sh/onnxruntime-web@1.20.1";

const ENCODER_URL = "https://github.com/MegaCode111REAL/resonance/releases/download/yourmt3-browser-v2/yourmt3-encoder.onnx";
const DECODER_URL = "https://github.com/MegaCode111REAL/resonance/releases/download/yourmt3-browser-v2/yourmt3-decoder.onnx";
const VOCAB_URL = "https://github.com/MegaCode111REAL/resonance/releases/download/yourmt3-browser-v2/yourmt3-vocab.json";

const SAMPLE_RATE = 16000;
const SEGMENT_SAMPLES = 32767;
const CHANNELS = 13;
const START_TOKEN = 0;
const MAX_TOKENS = 256;

let sessionsPromise = null;
let vocabPromise = null;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

async function downloadModel(url, label, progressStart, progressSpan) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(label + " download failed: HTTP " + response.status);
  }

  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body?.getReader();

  if (!reader) {
    return new Uint8Array(await response.arrayBuffer());
  }

  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    post("progress", {
      value: total
        ? progressStart + progressSpan * loaded / total
        : progressStart + progressSpan * 0.5,
      message: total
        ? "Downloading " + label + "… " +
          (loaded / 1048576).toFixed(0) + " / " +
          (total / 1048576).toFixed(0) + " MB"
        : "Downloading " + label + "…"
    });
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function getVocab() {
  if (!vocabPromise) {
    vocabPromise = fetch(VOCAB_URL, { cache: "force-cache" })
      .then(response => {
        if (!response.ok) {
          throw new Error("YourMT3 vocabulary download failed: HTTP " + response.status);
        }
        return response.json();
      });
  }
  return vocabPromise;
}

async function getSessions() {
  if (!sessionsPromise) {
    sessionsPromise = (async () => {
      ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
      ort.env.wasm.simd = true;

      post("progress", { value: 1, message: "Downloading YourMT3 encoder…" });
      const encoderBytes = await downloadModel(ENCODER_URL, "YourMT3 encoder", 2, 28);

      post("progress", { value: 30, message: "Downloading YourMT3 decoder…" });
      const decoderBytes = await downloadModel(DECODER_URL, "YourMT3 decoder", 31, 28);

      post("progress", { value: 60, message: "Loading YourMT3 into WebAssembly…" });

      const options = {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
        enableCpuMemArena: false,
        enableMemPattern: false
      };

      const encoder = await ort.InferenceSession.create(encoderBytes.buffer, options);
      post("progress", { value: 70, message: "YourMT3 encoder ready; loading decoder…" });
      const decoder = await ort.InferenceSession.create(decoderBytes.buffer, options);

      post("progress", { value: 75, message: "YourMT3 browser models ready" });
      return { encoder, decoder };
    })().catch(error => {
      sessionsPromise = null;
      throw error;
    });
  }
  return sessionsPromise;
}

function normalizeAudio(input) {
  const output = new Float32Array(SEGMENT_SAMPLES);
  const source = input instanceof Float32Array ? input : new Float32Array(input);
  output.set(source.subarray(0, Math.min(source.length, output.length)));
  return output;
}

function argmax(logits, offset, size) {
  let best = 0;
  let bestValue = logits[offset];
  for (let i = 1; i < size; i++) {
    if (logits[offset + i] > bestValue) {
      bestValue = logits[offset + i];
      best = i;
    }
  }
  return best;
}

function makeTokenTensor(tokens) {
  return new ort.Tensor(
    "int64",
    BigInt64Array.from(tokens, value => BigInt(value)),
    [1, CHANNELS, tokens.length / CHANNELS]
  );
}

function appendTokens(tokens, next) {
  const oldLength = tokens.length / CHANNELS;
  const expanded = new Int32Array(CHANNELS * (oldLength + 1));

  for (let channel = 0; channel < CHANNELS; channel++) {
    expanded.set(
      tokens.subarray(channel * oldLength, (channel + 1) * oldLength),
      channel * (oldLength + 1)
    );
    expanded[(channel + 1) * (oldLength + 1) - 1] = next[channel];
  }

  return expanded;
}

async function decodeSegment(audio) {
  const { encoder, decoder } = await getSessions();

  const audioTensor = new ort.Tensor(
    "float32",
    normalizeAudio(audio),
    [1, 1, SEGMENT_SAMPLES]
  );

  post("progress", { value: 76, message: "Encoding audio segment…" });
  const encoded = await encoder.run({ audio: audioTensor });
  const encoderHiddenStates = encoded.encoder_hidden_states;

  let tokens = new Int32Array(CHANNELS);
  tokens.fill(START_TOKEN);

  for (let step = 0; step < MAX_TOKENS; step++) {
    const tokenTensor = makeTokenTensor(tokens);
    const outputs = await decoder.run({
      encoder_hidden_states: encoderHiddenStates,
      target_tokens: tokenTensor
    });

    const logits = outputs.logits;
    const sequenceLength = logits.dims[2];
    const vocabSize = logits.dims[3];
    const next = new Int32Array(CHANNELS);

    for (let channel = 0; channel < CHANNELS; channel++) {
      const offset = (channel * sequenceLength + sequenceLength - 1) * vocabSize;
      next[channel] = argmax(logits.data, offset, vocabSize);
    }

    tokens = appendTokens(tokens, next);

    if (next.every(token => token === 1)) break;

    post("progress", {
      value: 76 + 14 * (step + 1) / MAX_TOKENS,
      message: "YourMT3 decoding token " + (step + 1) + " / " + MAX_TOKENS
    });
  }

  return tokens;
}

function decodeEvents(tokenIds, vocab, segmentStart) {
  const notes = [];
  const active = new Map();
  let time = segmentStart;
  let program = 0;
  let velocity = 1;
  let pitch = null;
  let isDrum = false;

  const finish = (key, end) => {
    const item = active.get(key);
    if (!item) return;
    if (end > item.start) {
      notes.push({
        midi: item.pitch,
        time: item.start,
        duration: end - item.start,
        velocity: item.velocity,
        program: item.program,
        isDrum: item.isDrum
      });
    }
    active.delete(key);
  };

  for (const token of tokenIds) {
    const event = vocab.events[token];
    if (!event || event.type == null) continue;

    const type = String(event.type).toLowerCase();
    const value = Number(event.value);

    if (type.includes("shift")) {
      time += value / 100;
    } else if (type.includes("program")) {
      program = value;
      isDrum = program === 128;
    } else if (type.includes("velocity")) {
      if (value <= 0 && pitch != null) {
        finish(program + ":" + pitch + ":" + isDrum, time);
        pitch = null;
      } else {
        velocity = Math.max(0.01, Math.min(1, value / 127));
      }
    } else if (type.includes("pitch")) {
      pitch = value;
      const key = program + ":" + pitch + ":" + isDrum;
      if (active.has(key)) finish(key, time);
      active.set(key, {
        pitch,
        start: time,
        velocity,
        program,
        isDrum
      });
    } else if (type.includes("tie")) {
      continue;
    } else if (type.includes("end")) {
      break;
    }
  }

  for (const [key] of active) finish(key, time);
  return notes;
}

async function transcribe(audio) {
  const vocab = await getVocab();
  const source = audio instanceof Float32Array ? audio : new Float32Array(audio);
  const segmentCount = Math.max(1, Math.ceil(source.length / SEGMENT_SAMPLES));
  const notes = [];

  for (let segment = 0; segment < segmentCount; segment++) {
    const start = segment * SEGMENT_SAMPLES;
    const chunk = source.subarray(start, Math.min(source.length, start + SEGMENT_SAMPLES));
    const tokens = await decodeSegment(chunk);
    const length = tokens.length / CHANNELS;

    for (let channel = 0; channel < CHANNELS; channel++) {
      const channelTokens = [];
      for (let i = 0; i < length; i++) {
        channelTokens.push(tokens[channel * length + i]);
      }
      notes.push(...decodeEvents(channelTokens, vocab, start / SAMPLE_RATE));
    }

    post("progress", {
      value: 90 + 10 * (segment + 1) / segmentCount,
      message: "YourMT3 transcribed segment " + (segment + 1) + " / " + segmentCount
    });
  }

  notes.sort((a, b) => a.time - b.time || a.program - b.program || a.midi - b.midi);
  const deduped = [];

  for (const note of notes) {
    const duplicate = deduped.find(existing =>
      existing.program === note.program &&
      existing.midi === note.midi &&
      Math.abs(existing.time - note.time) < 0.025
    );

    if (duplicate) {
      duplicate.duration = Math.max(duplicate.duration, note.duration);
      duplicate.velocity = Math.max(duplicate.velocity, note.velocity);
    } else {
      deduped.push(note);
    }
  }

  return deduped;
}

self.onmessage = async event => {
  if (event.data?.type !== "transcribe") return;

  try {
    post("progress", { value: 0, message: "Starting YourMT3 browser transcription…" });
    const notes = await transcribe(event.data.audio);
    post("result", { notes, sampleRate: SAMPLE_RATE });
  } catch (error) {
    post("error", { message: error?.stack || error?.message || String(error) });
  }
};
