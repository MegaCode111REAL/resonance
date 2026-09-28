import { Midi } from "https://esm.sh/@tonejs/midi@2.0.28";
import { BasicPitch, outputToNotesPoly, addPitchBendsToNoteEvents, noteFramesToTime } from "https://esm.sh/@spotify/basic-pitch@1.0.1";
import * as ort from "https://esm.sh/onnxruntime-web@1.20.1";
import { DemucsProcessor, CONSTANTS as DEMUCS_CONSTANTS } from "https://esm.sh/demucs-web@1.0.2";
import { detect as detectBeatGrid } from "https://esm.sh/@audio/beat@2.1.3";

const WEBMSCORE_URL = "https://cdn.jsdelivr.net/npm/webmscore@1.2.1/webmscore.mjs";

const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);
const KEY_NAMES = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"];

const INSTRUMENTS = {
  piano: { name: "Piano", low: 21, high: 108, preferredLow: 36, preferredHigh: 84, program: 0 },
  marimba: { name: "Marimba", low: 45, high: 96, preferredLow: 48, preferredHigh: 84, program: 12 },
  vibraphone: { name: "Vibraphone", low: 53, high: 89, preferredLow: 55, preferredHigh: 81, program: 11 },
  xylophone: { name: "Xylophone", low: 65, high: 108, preferredLow: 67, preferredHigh: 96, program: 13 },
  violin: { name: "Violin", low: 55, high: 103, preferredLow: 60, preferredHigh: 96, program: 40 },
  cello: { name: "Cello", low: 36, high: 76, preferredLow: 40, preferredHigh: 69, program: 42 },
  flute: { name: "Flute", low: 60, high: 96, preferredLow: 67, preferredHigh: 88, program: 73 },
  "acoustic-guitar": { name: "Acoustic Guitar", low: 40, high: 88, preferredLow: 45, preferredHigh: 79, program: 25 },
  "electric-guitar": { name: "Electric Guitar", low: 40, high: 88, preferredLow: 45, preferredHigh: 79, program: 27 },
  bass: { name: "Bass", low: 28, high: 55, preferredLow: 31, preferredHigh: 50, program: 33 }
};

const PITCHED_STEMS = ["vocals", "bass", "other"];

const BASIC_PITCH_PROFILES = {
  vocals: { minFreq: 65.41, maxFreq: 1046.5 },
  bass: { minFreq: 27.5, maxFreq: 440 },
  other: { minFreq: 27.5, maxFreq: 4186.01 }
};

const BASIC_PITCH_ONSET_THRESHOLD = 0.5;
const BASIC_PITCH_FRAME_THRESHOLD = 0.4;
const BASIC_PITCH_MIN_NOTE_LEN_FRAMES = 11;
const BASIC_PITCH_DUPLICATE_WINDOW_SECONDS = 0.035;

// The arranger currently writes MIDI at 120 BPM.
// A 1/16 note is one quarter of a beat = 0.125 seconds.
const DEFAULT_TEMPO = 120;
const QUANTIZE_GRID_BEATS = 1 / 4;

const INSTRUMENT_ORDER = Object.keys(INSTRUMENTS);

const audioInput = document.querySelector("#audio-file");
const dropzone = document.querySelector("#dropzone");
const fileInfo = document.querySelector("#file-info");
const panel = document.querySelector("#arrangement-panel");
const analysisPanel = document.querySelector("#analysis-panel");
const instrumentList = document.querySelector("#instrument-list");
const tempoElement = document.querySelector("#tempo");
const arrangeButton = document.querySelector("#arrange");
const transposeButton = document.querySelector("#auto-transpose");
const originalDownloadButton = document.querySelector("#download-original");
const downloadButton = document.querySelector("#download");
const status = document.querySelector("#status");
const progress = document.querySelector("#progress");
const progressFill = document.querySelector("#progress-fill");
const progressText = document.querySelector("#progress-text");
const result = document.querySelector("#result");
const stats = document.querySelector("#stats");
const stemsElement = document.querySelector("#stems");
const notesPreview = document.querySelector("#notes-preview");

let sourceFile = null;
let transcription = null;
let arrangedMidi = null;
let arrangedNotes = [];
let downloadBytes = null;
let demucsProcessor = null;
let basicPitchModel = null;
let detectedTempo = DEFAULT_TEMPO;
let webMscorePromise = null;
let webMscoreScore = null;

async function getWebMscore() {
  if (!webMscorePromise) {
    console.info("[Resonance WebMscore] Initializing MuseScore renderer");
    const started = performance.now();
    webMscorePromise = import(WEBMSCORE_URL).then(async module => {
      const WebMscore = module.default;
      await WebMscore.ready;
      console.info("[Resonance WebMscore] Renderer ready in " + ((performance.now() - started) / 1000).toFixed(1) + "s");
      return WebMscore;
    }).catch(error => {
      webMscorePromise = null;
      console.error("[Resonance WebMscore] Initialization failed", error);
      throw error;
    });
  }

  return webMscorePromise;
}

function warmWebMscore() {
  const start = () => {
    getWebMscore().catch(() => {});
  };

  if ("requestIdleCallback" in window) {
    window.requestIdleCallback(start, { timeout: 2500 });
  } else {
    setTimeout(start, 1200);
  }
}

async function renderScorePreview() {
  if (!arrangedMidi) return;

  const viewer = document.querySelector("#score-viewer");
  const viewerStatus = document.querySelector("#score-viewer-status");
  const scorePages = document.querySelector("#score-pages");
  if (!viewer || !viewerStatus || !scorePages) return;

  viewer.classList.remove("hidden");
  scorePages.innerHTML = "";
  viewerStatus.textContent = "Loading MuseScore renderer…";

  try {
    const WebMscore = await getWebMscore();

    if (webMscoreScore) {
      webMscoreScore.destroy();
      webMscoreScore = null;
    }

    viewerStatus.textContent = "Engraving score…";
    const midiBytes = arrangedMidi.toArray();
    webMscoreScore = await WebMscore.load("midi", midiBytes, [], true);

    const pageCount = await webMscoreScore.npages();
    for (let page = 0; page < pageCount; page++) {
      const svg = await webMscoreScore.saveSvg(page, false);
      const pageElement = document.createElement("div");
      pageElement.className = "score-page";
      pageElement.innerHTML = svg;
      scorePages.appendChild(pageElement);
    }

    viewerStatus.textContent = pageCount + (pageCount === 1 ? " page" : " pages") + " · MuseScore engraving";
  } catch (error) {
    console.error("[Resonance WebMscore] Rendering failed", error);
    viewerStatus.textContent = "Score preview failed: " + (error?.message || error);
  }
}

function setStatus(message) {
  status.textContent = message;
}

function setProgress(value, message) {
  progress.classList.remove("hidden");
  progressFill.style.width = Math.max(0, Math.min(100, value)) + "%";
  progressText.textContent = message;
}

function hideProgress() {
  progress.classList.add("hidden");
}

function clampPitch(pitch, instrument, partIndex, partCount) {
  const span = instrument.high - instrument.low;
  const preferredCenter = instrument.preferredLow + (instrument.preferredHigh - instrument.preferredLow) / 2;
  let value = Math.round(pitch);

  while (value < instrument.low) value += 12;
  while (value > instrument.high) value -= 12;

  const offset = partCount <= 1 ? 0 : (partIndex - (partCount - 1) / 2) * Math.min(7, span / Math.max(1, partCount * 2));
  const target = preferredCenter + offset;
  while (value < target - 12 && value + 12 <= instrument.high) value += 12;
  while (value > target + 12 && value - 12 >= instrument.low) value -= 12;

  return Math.max(instrument.low, Math.min(instrument.high, value));
}

function noteName(midi) {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  return names[midi % 12] + (Math.floor(midi / 12) - 1);
}

function renderAnalysis(notes, duration, stemCounts) {

  if (!notes.length) return;

  if (tempoElement) tempoElement.textContent = Math.round(detectedTempo) + " BPM";

  const pitches = notes.map(n => n.midi);
  const min = Math.min(...pitches);
  const max = Math.max(...pitches);
  const unique = new Set(pitches.map(p => p % 12)).size;

  stats.innerHTML = [
    ["Notes", notes.length],
    ["Duration", duration.toFixed(1) + " s"],
    ["Pitch range", noteName(min) + "–" + noteName(max)],
    ["Pitch classes", unique + " / 12"]
  ].map(([label, value]) => "<div><span>" + label + "</span><strong>" + value + "</strong></div>").join("");

  const labels = {
    vocals: "Vocals",
    bass: "Bass",
    other: "Other instruments",
    drums: "Drums"
  };

  stemsElement.innerHTML = Object.entries(stemCounts)
    .map(([stem, count]) => "<div class=\"stem\"><span>" + (labels[stem] || stem) + "</span><strong>" + (count ? count + " notes" : "rhythm only") + "</strong></div>")
    .join("");

  notesPreview.textContent = notes.slice(0, 28).map(n => noteName(n.midi)).join(" · ") + (notes.length > 28 ? " · …" : "");
  analysisPanel.classList.remove("hidden");
}

function quantizeNotes(notes, tempo = detectedTempo) {
  const secondsPerBeat = 60 / tempo;
  const gridSeconds = secondsPerBeat * QUANTIZE_GRID_BEATS;

  return notes
    .map(note => {
      const start = Math.max(0, Math.round(note.time / gridSeconds) * gridSeconds);
      const rawEnd = Math.max(note.time + note.duration, note.time + 0.05);
      const end = Math.max(start + gridSeconds, Math.round(rawEnd / gridSeconds) * gridSeconds);

      return {
        ...note,
        time: start,
        duration: Math.max(gridSeconds, end - start)
      };
    })
    .sort((a, b) => a.time - b.time || b.midi - a.midi);
}

function cleanTranscribedNotes(notes) {
  const sorted = [...notes]
    .filter(note => Number.isFinite(note.midi) && Number.isFinite(note.time) && Number.isFinite(note.duration))
    .filter(note => (note.velocity ?? 0) >= 0.12)
    .sort((a, b) => a.time - b.time || b.velocity - a.velocity);

  const cleaned = [];

  for (const note of sorted) {
    const duplicateIndex = cleaned.findIndex(existing =>
      existing.midi === note.midi &&
      existing.stem === note.stem &&
      Math.abs(existing.time - note.time) <= BASIC_PITCH_DUPLICATE_WINDOW_SECONDS
    );

    if (duplicateIndex >= 0) {
      if ((note.velocity ?? 0) > (cleaned[duplicateIndex].velocity ?? 0)) {
        cleaned[duplicateIndex] = note;
      }
      continue;
    }

    cleaned.push(note);
  }

  return quantizeNotes(cleaned);
}

function selectedEnsemble() {
  return INSTRUMENT_ORDER
    .map(key => {
      const input = instrumentList?.querySelector('[data-instrument="' + key + '"]');
      return {
        key,
        parts: Math.max(0, Math.min(8, Number(input?.value) || 0))
      };
    })
    .filter(item => item.parts > 0);
}

function chooseTargetInstrument(note, ensemble) {
  const preferred = {
    bass: ["bass", "cello", "acoustic-guitar"],
    vocals: ["flute", "violin", "cello", "piano", "vibraphone"],
    other: ["piano", "marimba", "acoustic-guitar", "electric-guitar", "vibraphone", "violin", "flute", "cello", "xylophone", "bass"]
  };

  const ordered = preferred[note.stem] || preferred.other;
  for (const key of ordered) {
    if (ensemble.some(item => item.key === key)) return key;
  }

  const byRange = [...ensemble].sort((a, b) => {
    const ia = INSTRUMENTS[a.key];
    const ib = INSTRUMENTS[b.key];
    const da = note.midi < ia.low ? ia.low - note.midi : note.midi > ia.high ? note.midi - ia.high : 0;
    const db = note.midi < ib.low ? ib.low - note.midi : note.midi > ib.high ? note.midi - ib.high : 0;
    return da - db;
  });
  return byRange[0].key;
}

function buildEnsembleArrangement(notes, ensemble, tempo = detectedTempo) {
  const midi = new Midi();
  midi.header.setTempo(tempo);

  const tracks = [];
  for (const item of ensemble) {
    const instrument = INSTRUMENTS[item.key];
    for (let part = 0; part < item.parts; part++) {
      const track = midi.addTrack();
      track.name = instrument.name + " " + (part + 1);
      track.channel = tracks.length % 16;
      track.instrument.number = instrument.program;
      tracks.push({ track, key: item.key, part, parts: item.parts });
    }
  }

  const buckets = new Map();
  for (const note of notes) {
    const targetKey = chooseTargetInstrument(note, ensemble);
    const candidates = tracks.filter(track => track.key === targetKey);
    const timeBucket = Math.round(note.time / (60 / tempo / 4));
    const bucketKey = targetKey + ":" + timeBucket;
    const index = buckets.get(bucketKey) ?? 0;
    buckets.set(bucketKey, index + 1);
    const target = candidates[index % candidates.length];
    const instrument = INSTRUMENTS[target.key];
    const pitch = clampPitch(note.midi, instrument, target.part, target.parts);

    target.track.addNote({
      midi: pitch,
      time: note.time,
      duration: Math.max(0.05, note.duration),
      velocity: Math.max(0.08, Math.min(1, note.velocity ?? 0.75))
    });
  }

  return midi;
}

function renderInstrumentChoices() {
  if (!instrumentList) return;
  instrumentList.innerHTML = INSTRUMENT_ORDER.map(key => {
    const instrument = INSTRUMENTS[key];
    const defaultParts = key === "marimba" ? 2 : key === "piano" ? 1 : 0;
    return '<label class="instrument-choice">' +
      '<span><input type="checkbox" data-toggle="' + key + '"' + (defaultParts ? " checked" : "") + '><strong>' + instrument.name + '</strong></span>' +
      '<input data-instrument="' + key + '" type="number" min="0" max="8" value="' + defaultParts + '" aria-label="' + instrument.name + ' parts">' +
      '</label>';
  }).join("");

  instrumentList.querySelectorAll("[data-toggle]").forEach(toggle => {
    toggle.addEventListener("change", () => {
      const input = instrumentList.querySelector('[data-instrument="' + toggle.dataset.toggle + '"]');
      if (input) input.value = toggle.checked ? Math.max(1, Number(input.value) || 1) : 0;
    });
  });

  instrumentList.querySelectorAll("[data-instrument]").forEach(input => {
    input.addEventListener("input", () => {
      const toggle = instrumentList.querySelector('[data-toggle="' + input.dataset.instrument + '"]');
      if (toggle) toggle.checked = Number(input.value) > 0;
    });
  });
}

function buildArrangement(notes, instrumentName, partCount, tempo = detectedTempo) {
  const instrument = INSTRUMENTS[instrumentName];
  const midi = new Midi();
  midi.header.setTempo(tempo);

  const tracks = Array.from({ length: partCount }, (_, index) => {
    const track = midi.addTrack();
    track.name = instrument.name + " " + (index + 1);
    track.channel = index % 16;
    track.instrument.number = instrument.program;
    return track;
  });

  const sorted = [...notes].sort((a, b) => a.time - b.time || b.midi - a.midi);
  const activeAt = new Map();

  for (const [index, note] of sorted.entries()) {
    const timeBucket = Math.round(note.time * 20) / 20;
    const bucket = activeAt.get(timeBucket) ?? 0;
    const part = (index + bucket) % partCount;
    activeAt.set(timeBucket, bucket + 1);

    const pitch = clampPitch(note.midi, instrument, part, partCount);
    tracks[part].addNote({
      midi: pitch,
      time: note.time,
      duration: Math.max(0.05, note.duration),
      velocity: Math.max(0.08, Math.min(1, note.velocity ?? 0.75))
    });
  }

  return midi;
}

function scoreShift(notes, shift, instrument) {
  let invalid = 0;
  let black = 0;
  let rangePenalty = 0;

  for (const note of notes) {
    const pitch = note.midi + shift;
    if (pitch < 0 || pitch > 127) {
      invalid++;
      continue;
    }
    if (BLACK_KEYS.has(pitch % 12)) black++;
    if (instrument && (pitch < instrument.low || pitch > instrument.high)) rangePenalty++;
  }

  return { invalid, black, rangePenalty };
}

function chooseBestShift(notes, instrumentName) {
  const instrument = INSTRUMENTS[instrumentName];
  const candidates = [];

  for (let shift = -6; shift < 6; shift++) {
    const scored = scoreShift(notes, shift, instrument);
    candidates.push({
      shift,
      ...scored,
      distance: Math.abs(shift),
      unchanged: shift === 0 ? 0 : 1
    });
  }

  candidates.sort((a, b) =>
    a.invalid - b.invalid ||
    a.rangePenalty - b.rangePenalty ||
    a.black - b.black ||
    a.distance - b.distance ||
    a.unchanged - b.unchanged ||
    a.shift - b.shift
  );

  return candidates[0];
}

function applyShiftToArrangement(midi, shift) {
  for (const track of midi.tracks) {
    for (const note of track.notes) note.midi += shift;
  }
}

function getArrangementNotes(midi) {
  return midi.tracks.flatMap(track => track.notes.map(note => ({
    midi: note.midi,
    time: note.time,
    duration: note.duration,
    velocity: note.velocity
  })));
}

function buildOriginalMidi(notes, tempo = detectedTempo) {
  const midi = new Midi();
  midi.header.setTempo(tempo);

  const stems = ["vocals", "bass", "other"];
  const labels = {
    vocals: "Vocals",
    bass: "Bass",
    other: "Other"
  };

  for (const stem of stems) {
    const trackNotes = notes.filter(note => note.stem === stem);
    if (!trackNotes.length) continue;

    const track = midi.addTrack();
    track.name = labels[stem] + " (Unarranged)";
    track.channel = midi.tracks.length - 1;
    track.instrument.number = stem === "bass" ? 33 : 0;

    for (const note of trackNotes) {
      track.addNote({
        midi: note.midi,
        time: note.time,
        duration: Math.max(0.05, note.duration),
        velocity: Math.max(0.08, Math.min(1, note.velocity ?? 0.75))
      });
    }
  }

  return midi;
}

function downloadMidi(midi, filename) {
  const bytes = midi.toArray();
  const blob = new Blob([bytes], { type: "audio/midi" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setDownload() {
  downloadBytes = arrangedMidi.toArray();
  downloadButton.disabled = false;
}

function showResult(text) {
  result.classList.remove("hidden");
  result.textContent = text;
}

function mixToMono(left, right) {
  const length = Math.min(left.length, right.length);
  const mono = new Float32Array(length);
  for (let i = 0; i < length; i++) mono[i] = (left[i] + right[i]) * 0.5;
  return mono;
}

async function resampleToMono22050(source) {
  const targetSampleRate = 22050;

  if (source.sampleRate === targetSampleRate && source.numberOfChannels === 1) {
    return source.getChannelData(0);
  }

  const frameCount = Math.ceil(source.duration * targetSampleRate);
  const offlineContext = new OfflineAudioContext(1, frameCount, targetSampleRate);
  const bufferSource = offlineContext.createBufferSource();
  bufferSource.buffer = source;
  bufferSource.connect(offlineContext.destination);
  bufferSource.start(0);

  const rendered = await offlineContext.startRendering();
  return rendered.getChannelData(0);
}

async function resampleStereo44100(decodedAudio) {
  if (decodedAudio.sampleRate === 44100) {
    return {
      left: decodedAudio.getChannelData(0),
      right: decodedAudio.numberOfChannels > 1 ? decodedAudio.getChannelData(1) : decodedAudio.getChannelData(0)
    };
  }

  const frameCount = Math.ceil(decodedAudio.duration * 44100);
  const offlineContext = new OfflineAudioContext(2, frameCount, 44100);
  const source = offlineContext.createBufferSource();
  source.buffer = decodedAudio;
  source.connect(offlineContext.destination);
  source.start(0);

  const rendered = await offlineContext.startRendering();
  return {
    left: rendered.getChannelData(0),
    right: rendered.numberOfChannels > 1 ? rendered.getChannelData(1) : rendered.getChannelData(0)
  };
}

const DEMUCS_CACHE_NAME = "resonance-models-v1";
const DEMUCS_MODEL_URL = DEMUCS_CONSTANTS.DEFAULT_MODEL_URL;

async function loadCachedDemucsModel() {
  const cache = await caches.open(DEMUCS_CACHE_NAME);
  const cached = await cache.match(DEMUCS_MODEL_URL);

  if (cached) {
    const size = Number(cached.headers.get("content-length")) || 0;
    setProgress(35, size ? "Loading cached separation model… " + (size / 1048576).toFixed(0) + " MB" : "Loading cached separation model…");
    return cached.arrayBuffer();
  }

  setProgress(1, "Downloading separation model…");
  const response = await fetch(DEMUCS_MODEL_URL, { cache: "reload" });

  if (!response.ok) {
    throw new Error("Could not download the Demucs separation model (" + response.status + ").");
  }

  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body?.getReader();

  if (!reader) {
    const buffer = await response.arrayBuffer();
    await cache.put(DEMUCS_MODEL_URL, new Response(buffer, {
      headers: { "content-type": "application/octet-stream" }
    }));
    setProgress(35, "Separation model cached");
    return buffer;
  }

  const chunks = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;

    const percent = total ? loaded / total : 0;
    const loadedMB = (loaded / 1048576).toFixed(0);
    const totalMB = total ? (total / 1048576).toFixed(0) : "?";
    setProgress(percent * 35, "Downloading separation model… " + loadedMB + " / " + totalMB + " MB");
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }

  await cache.put(DEMUCS_MODEL_URL, new Response(buffer, {
    headers: {
      "content-type": response.headers.get("content-type") || "application/octet-stream",
      "content-length": String(buffer.byteLength)
    }
  }));

  setProgress(35, "Separation model cached");
  return buffer.buffer;
}

async function getDemucsProcessor() {
  if (demucsProcessor) return demucsProcessor;

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  // GitHub Pages does not provide the COOP/COEP isolation required by
  // ONNX Runtime's more advanced browser backends. Force the stable WASM
  // backend here instead of probing WebGPU, which can throw NotFoundError
  // in Safari even when navigator.gpu exists.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  demucsProcessor = new DemucsProcessor({
    ort,
    sessionOptions: {
      enableCpuMemArena: false,
      enableMemPattern: false
    },
    onProgress: ({ progress: value, currentSegment, totalSegments }) => {
      const percent = Math.round(value * 100);
      setProgress(35 + percent * 0.35, "Separating stems… " + percent + "% (" + currentSegment + "/" + totalSegments + ")");
    },
    onLog: (phase, message) => console.debug("[Resonance Demucs]", phase, message)
  });

  const model = await loadCachedDemucsModel();
  setProgress(36, "Initializing separation model…");
  console.info("[Resonance Demucs] Loading ONNX model into ONNX Runtime");
  const started = performance.now();
  try {
    await demucsProcessor.loadModel(model);
  } catch (error) {
    console.error("[Resonance Demucs] Model initialization failed", error);
    throw new Error("Demucs model initialization failed: " + (error?.message || error));
  }
  console.info("[Resonance Demucs] ONNX model ready in " + ((performance.now() - started) / 1000).toFixed(1) + "s");
  setProgress(40, "Separation model ready");
  return demucsProcessor;
}

async function getBasicPitchModel() {
  if (!basicPitchModel) {
    setProgress(41, "Loading note transcription model…");
    basicPitchModel = new BasicPitch("https://unpkg.com/@spotify/basic-pitch@1.0.1/model/model.json");
  }
  return basicPitchModel;
}

async function transcribeStem(stemName, stem, duration, basicPitch) {
  const mono = mixToMono(stem.left, stem.right);

  const sourceContext = new OfflineAudioContext(1, mono.length, 44100);
  const sourceBuffer = sourceContext.createBuffer(1, mono.length, 44100);
  sourceBuffer.copyToChannel(mono, 0);

  const frameCount = Math.ceil(duration * 22050);
  const targetContext = new OfflineAudioContext(1, frameCount, 22050);
  const source = targetContext.createBufferSource();
  source.buffer = sourceBuffer;
  source.connect(targetContext.destination);
  source.start(0);

  const rendered = await targetContext.startRendering();
  const audioData = rendered.getChannelData(0);

  const frames = [];
  const onsets = [];
  const contours = [];

  await basicPitch.evaluateModel(
    audioData,
    (frameChunk, onsetChunk, contourChunk) => {
      for (const row of frameChunk) frames.push(row);
      for (const row of onsetChunk) onsets.push(row);
      for (const row of contourChunk) contours.push(row);
    },
    () => {}
  );

  const profile = BASIC_PITCH_PROFILES[stemName] || BASIC_PITCH_PROFILES.other;
  const noteEvents = outputToNotesPoly(
    frames,
    onsets,
    BASIC_PITCH_ONSET_THRESHOLD,
    BASIC_PITCH_FRAME_THRESHOLD,
    BASIC_PITCH_MIN_NOTE_LEN_FRAMES,
    true,
    profile.maxFreq,
    profile.minFreq,
    true
  );
  const withBends = addPitchBendsToNoteEvents(contours, noteEvents);

  return noteFramesToTime(withBends).map(note => ({
    midi: Math.max(0, Math.min(127, Math.round(note.pitchMidi))),
    time: note.startTimeSeconds,
    duration: Math.max(0.05, note.durationSeconds),
    velocity: Math.max(0.08, Math.min(1, note.amplitude ?? 0.75)),
    stem: stemName
  }));
}

function gmProgramName(program) {
  const names = [
    "Piano", "Piano", "Electric Piano", "Honky-tonk Piano", "Electric Piano", "Electric Piano",
    "Harpsichord", "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer", "Organ", "Organ", "Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar", "Acoustic Guitar",
    "Electric Guitar", "Electric Guitar", "Electric Guitar", "Overdriven Guitar", "Distortion Guitar",
    "Guitar Harmonics", "Acoustic Bass", "Electric Bass", "Electric Bass", "Fretless Bass",
    "Slap Bass", "Slap Bass", "Synth Bass", "Synth Bass", "Violin", "Viola", "Cello",
    "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble", "String Ensemble", "Synth Strings", "Synth Strings", "Choir", "Voice",
    "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass", "Synth Brass", "Soprano Sax", "Alto Sax",
    "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo",
    "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead", "Lead", "Lead", "Lead", "Lead", "Lead", "Lead", "Lead", "Pad", "Pad", "Pad",
    "Pad", "Pad", "Pad", "Pad", "Pad", "FX", "FX", "FX", "FX", "FX", "FX", "FX", "FX",
    "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko", "Melodic Tom", "Synth Drum",
    "Reverse Cymbal", "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone", "Helicopter", "Applause", "Gunshot"
  ];
  return names[program] || (program === 128 ? "Drums" : "MIDI " + (program + 1));
}

function stemForProgram(program, isDrum) {
  if (isDrum || program === 128) return "drums";
  if (program >= 32 && program <= 39) return "bass";
  if (program >= 52 && program <= 55) return "vocals";
  return "other";
}

async function transcribeWithYourMT3(audioBuffer) {
  const targetRate = 16000;
  const frameCount = Math.ceil(audioBuffer.duration * targetRate);
  const context = new OfflineAudioContext(1, frameCount, targetRate);
  const source = context.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(context.destination);
  source.start(0);
  const rendered = await context.startRendering();
  const mono = rendered.getChannelData(0);

  return new Promise((resolve, reject) => {
    const worker = new Worker("./yourmt3_worker.js", { type: "module" });
    const cleanup = () => worker.terminate();

    worker.onmessage = event => {
      const message = event.data;
      if (message.type === "progress") {
        setProgress(message.value, message.message);
        return;
      }
      if (message.type === "error") {
        cleanup();
        reject(new Error(message.message));
        return;
      }
      if (message.type !== "result") return;

      const notes = message.notes.map(note => ({
        midi: Math.max(0, Math.min(127, Math.round(note.midi))),
        time: Math.max(0, note.time),
        duration: Math.max(0.025, note.duration),
        velocity: Math.max(0.08, Math.min(1, note.velocity)),
        stem: stemForProgram(note.program, note.isDrum),
        program: note.program,
        isDrum: Boolean(note.isDrum),
        sourcePart: note.isDrum ? "Drums" : gmProgramName(note.program)
      }));

      cleanup();
      resolve({
        notes: cleanTranscribedNotes(notes),
        duration: audioBuffer.duration,
        tempo: detectedTempo,
        stemCounts: {
          vocals: notes.filter(note => note.stem === "vocals").length,
          bass: notes.filter(note => note.stem === "bass").length,
          other: notes.filter(note => note.stem === "other").length,
          drums: notes.filter(note => note.stem === "drums").length
        }
      });
    };

    worker.onerror = event => {
      cleanup();
      reject(event.error || new Error(event.message || "YourMT3 worker failed."));
    };

    worker.postMessage({ type: "transcribe", audio: mono }, [mono.buffer]);
  });
}

async function transcribeAudio(file) {
  setProgress(2, "Decoding audio locally…");

  const decodeContext = new AudioContext();
  const decodedAudio = await decodeContext.decodeAudioData(await file.arrayBuffer());
  const originalDuration = decodedAudio.duration;

  setProgress(5, "Detecting tempo and beat grid…");
  try {
    const monoForBeat = decodedAudio.numberOfChannels > 1
      ? mixToMono(decodedAudio.getChannelData(0), decodedAudio.getChannelData(1))
      : decodedAudio.getChannelData(0);
    const sourceRate = decodedAudio.sampleRate;
    const beatResult = detectBeatGrid(monoForBeat, {
      fs: sourceRate,
      minBpm: 60,
      maxBpm: 200
    });
    detectedTempo = Number.isFinite(beatResult.bpm) ? beatResult.bpm : DEFAULT_TEMPO;
  } catch (error) {
    console.warn("[Resonance] Tempo detection failed; using default tempo", error);
    detectedTempo = DEFAULT_TEMPO;
  }

  setProgress(10, "Loading multi-instrument YourMT3 model…");
  const result = await transcribeWithYourMT3(decodedAudio);
  await decodeContext.close();

  if (!result.notes.length) {
    throw new Error("YourMT3 did not produce any musical notes for this recording.");
  }

  return {
    ...result,
    duration: originalDuration
  };
}

async function handleAudio(file) {
  if (!file) return;

  sourceFile = file;
  arrangedMidi = null;
  arrangedNotes = [];
  downloadBytes = null;
  transcription = null;

  transposeButton.disabled = true;
  downloadButton.disabled = true;
  arrangeButton.disabled = true;
  result.classList.add("hidden");
  originalDownloadButton.disabled = true;

  fileInfo.classList.remove("hidden");
  fileInfo.textContent = file.name + " · " + (file.size / 1048576).toFixed(2) + " MB";
  panel.classList.remove("hidden");
  setStatus("Transcribing all instruments…");

  try {
    transcription = await transcribeAudio(file);
    renderAnalysis(transcription.notes, transcription.duration, transcription.stemCounts);
    arrangeButton.disabled = false;
    originalDownloadButton.disabled = false;
    setProgress(100, "Separation and transcription complete");
    setStatus(transcription.notes.length + " notes from YourMT3 multi-instrument transcription");
    setTimeout(hideProgress, 900);
  } catch (error) {
    console.error(error);
    hideProgress();
    setStatus("Audio analysis failed");
    showResult(error?.message || "The audio could not be separated or transcribed in this browser.");
  }
}

audioInput.addEventListener("change", () => handleAudio(audioInput.files?.[0]));

["dragenter", "dragover"].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault();
  dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", event => {
  const file = event.dataTransfer.files?.[0];
  if (file) handleAudio(file);
});

arrangeButton.addEventListener("click", async () => {
  if (!transcription) return;

  const ensemble = selectedEnsemble();
  if (!ensemble.length) {
    showResult("Select at least one instrument and give it at least one part.");
    return;
  }

  arrangeButton.disabled = true;
  transposeButton.disabled = true;
  setProgress(92, "Building musical arrangement…");

  await new Promise(requestAnimationFrame);

  arrangedMidi = buildEnsembleArrangement(transcription.notes, ensemble, detectedTempo);
  arrangedNotes = getArrangementNotes(arrangedMidi);

  setProgress(100, "Arrangement complete");
  setStatus(ensemble.map(item => INSTRUMENTS[item.key].name + " × " + item.parts).join(" · "));
  setDownload();
  transposeButton.disabled = arrangedNotes.length === 0;
  await renderScorePreview();
  showResult("Created " + ensemble.reduce((sum, item) => sum + item.parts, 0) + " ensemble parts at " + Math.round(detectedTempo) + " BPM, quantized to 1/16 notes.");
  arrangeButton.disabled = false;
  setTimeout(hideProgress, 700);
});

transposeButton.addEventListener("click", async () => {
  if (!arrangedMidi) return;

  transposeButton.disabled = true;
  setProgress(10, "Testing all 12 chromatic transpositions…");
  await new Promise(requestAnimationFrame);

  const ensemble = selectedEnsemble();
  const rangeInstrument = ensemble.length ? ensemble[0].key : "piano";
  const best = chooseBestShift(arrangedNotes, rangeInstrument);
  applyShiftToArrangement(arrangedMidi, best.shift);
  arrangedNotes = getArrangementNotes(arrangedMidi);
  setDownload();

  const key = KEY_NAMES[((best.shift % 12) + 12) % 12];
  const direction = best.shift === 0 ? "no shift" : (best.shift > 0 ? "+" : "") + best.shift + " semitones";

  showResult(
    "Auto Transpose: " + key +
    " · " + direction +
    " · " + best.black + " black-key notes" +
    " · " + best.rangePenalty + " range adjustments"
  );

  setProgress(100, "Auto Transpose complete");
  setStatus("Optimized: " + key);
  transposeButton.disabled = false;
  setTimeout(hideProgress, 700);
});

originalDownloadButton.addEventListener("click", () => {
  if (!transcription?.notes?.length) return;

  const midi = buildOriginalMidi(transcription.notes, transcription.tempo);
  const base = sourceFile?.name?.replace(/\.[^.]+$/, "") || "resonance";
  downloadMidi(midi, base + "-unarranged.mid");
});

downloadButton.addEventListener("click", () => {
  if (!downloadBytes) return;

  const blob = new Blob([downloadBytes], { type: "audio/midi" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;

  const base = sourceFile?.name?.replace(/\\.[^.]+$/, "") || "resonance";
  link.download = base + "-arrangement.mid";
  link.click();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

function instrumentName(key) {
  return INSTRUMENTS[key]?.name || key;
}

renderInstrumentChoices();
warmWebMscore();
