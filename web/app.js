import { Midi } from "https://esm.sh/@tonejs/midi@2.0.28";
import {
  BasicPitch,
  outputToNotesPoly,
  addPitchBendsToNoteEvents,
  noteFramesToTime
} from "https://esm.sh/@spotify/basic-pitch@1.0.1";

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

const audioInput = document.querySelector("#audio-file");
const dropzone = document.querySelector("#dropzone");
const fileInfo = document.querySelector("#file-info");
const panel = document.querySelector("#arrangement-panel");
const analysisPanel = document.querySelector("#analysis-panel");
const instrumentInput = document.querySelector("#instrument");
const partsInput = document.querySelector("#parts");
const arrangeButton = document.querySelector("#arrange");
const transposeButton = document.querySelector("#auto-transpose");
const downloadButton = document.querySelector("#download");
const status = document.querySelector("#status");
const progress = document.querySelector("#progress");
const progressFill = document.querySelector("#progress-fill");
const progressText = document.querySelector("#progress-text");
const result = document.querySelector("#result");
const stats = document.querySelector("#stats");
const notesPreview = document.querySelector("#notes-preview");

let sourceFile = null;
let transcription = null;
let arrangedMidi = null;
let arrangedNotes = [];
let downloadBytes = null;

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

function renderAnalysis(notes, duration) {
  const pitches = notes.map(n => n.midi);
  const min = Math.min(...pitches);
  const max = Math.max(...pitches);
  const unique = new Set(pitches.map(p => p % 12)).size;
  stats.innerHTML = [
    ["Notes", notes.length],
    ["Duration", duration.toFixed(1) + " s"],
    ["Pitch range", noteName(min) + "–" + noteName(max)],
    ["Pitch classes", unique + " / 12"]
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");

  notesPreview.textContent = notes.slice(0, 28).map(n => noteName(n.midi)).join(" · ") + (notes.length > 28 ? " · …" : "");
  analysisPanel.classList.remove("hidden");
}

function buildArrangement(notes, instrumentName, partCount, duration, tempo = 120) {
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

function setDownload() {
  downloadBytes = arrangedMidi.toArray();
  downloadButton.disabled = false;
}

function showResult(text) {
  result.classList.remove("hidden");
  result.textContent = text;
}

async function resampleAudioBuffer(source, sampleRate) {
  if (source.sampleRate === sampleRate) return source;

  const frameCount = Math.ceil(source.duration * sampleRate);
  const offlineContext = new OfflineAudioContext(
    source.numberOfChannels,
    frameCount,
    sampleRate
  );
  const bufferSource = offlineContext.createBufferSource();
  bufferSource.buffer = source;
  bufferSource.connect(offlineContext.destination);
  bufferSource.start(0);

  return offlineContext.startRendering();
}

async function transcribeAudio(file) {
  setProgress(5, "Decoding audio locally…");
  const audioContext = new AudioContext();
  const decodedAudio = await audioContext.decodeAudioData(await file.arrayBuffer());

  setProgress(12, "Resampling audio to 22050 Hz…");
  const audioBuffer = await resampleAudioBuffer(decodedAudio, 22050);
  const modelUrl = "https://unpkg.com/@spotify/basic-pitch@1.0.1/model/model.json";
  const basicPitch = new BasicPitch(modelUrl);

  const frames = [];
  const onsets = [];
  const contours = [];

  setProgress(18, "Loading transcription model…");
  await basicPitch.evaluateModel(
    audioBuffer,
    (frameChunk, onsetChunk, contourChunk) => {
      for (const row of frameChunk) frames.push(row);
      for (const row of onsetChunk) onsets.push(row);
      for (const row of contourChunk) contours.push(row);
    },
    pct => setProgress(20 + pct * 0.58, "Transcribing audio… " + Math.round(pct * 100) + "%")
  );

  setProgress(82, "Converting model output into notes…");
  const noteEvents = outputToNotesPoly(frames, onsets, 0.25, 0.25, 5);
  const withBends = addPitchBendsToNoteEvents(contours, noteEvents);
  const notes = noteFramesToTime(withBends).map(note => ({
    midi: Math.max(0, Math.min(127, Math.round(note.pitchMidi))),
    time: note.startTimeSeconds,
    duration: Math.max(0.05, note.durationSeconds),
    velocity: Math.max(0.08, Math.min(1, note.amplitude ?? 0.75))
  }));

  await audioContext.close();
  if (!notes.length) throw new Error("The transcription model found no pitched notes in this recording.");

  return { notes, duration: audioBuffer.duration };
}

async function handleAudio(file) {
  if (!file) return;
  sourceFile = file;
  arrangedMidi = null;
  arrangedNotes = [];
  downloadBytes = null;
  transposeButton.disabled = true;
  downloadButton.disabled = true;
  arrangeButton.disabled = true;
  result.classList.add("hidden");

  fileInfo.classList.remove("hidden");
  fileInfo.textContent = file.name + " · " + (file.size / 1048576).toFixed(2) + " MB";
  panel.classList.remove("hidden");
  setStatus("Transcribing…");

  try {
    transcription = await transcribeAudio(file);
    renderAnalysis(transcription.notes, transcription.duration);
    arrangeButton.disabled = false;
    setProgress(100, "Transcription complete");
    setStatus(transcription.notes.length + " notes found");
    setTimeout(hideProgress, 700);
  } catch (error) {
    console.error(error);
    hideProgress();
    setStatus("Transcription failed");
    showResult(error?.message || "The audio could not be transcribed in this browser.");
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
  const parts = Math.max(1, Math.min(16, Number(partsInput.value) || 1));
  const instrument = instrumentInput.value;

  arrangeButton.disabled = true;
  transposeButton.disabled = true;
  setProgress(5, "Building musical arrangement…");

  await new Promise(requestAnimationFrame);
  arrangedMidi = buildArrangement(
    transcription.notes,
    instrument,
    parts,
    transcription.duration
  );
  arrangedNotes = getArrangementNotes(arrangedMidi);
  setProgress(100, "Arrangement complete");
  setStatus(instrumentName(instrument) + " × " + parts);
  setDownload();
  transposeButton.disabled = arrangedNotes.length === 0;
  showResult("Created " + parts + " " + INSTRUMENTS[instrument].name + " part" + (parts === 1 ? "" : "s") + " from the transcription.");
  arrangeButton.disabled = false;
  setTimeout(hideProgress, 700);
});

transposeButton.addEventListener("click", async () => {
  if (!arrangedMidi) return;
  transposeButton.disabled = true;
  setProgress(10, "Testing all 12 chromatic transpositions…");
  await new Promise(requestAnimationFrame);

  const instrument = instrumentInput.value;
  const best = chooseBestShift(arrangedNotes, instrument);
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

downloadButton.addEventListener("click", () => {
  if (!downloadBytes) return;
  const blob = new Blob([downloadBytes], { type: "audio/midi" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  const base = sourceFile?.name?.replace(/\.[^.]+$/, "") || "resonance";
  link.download = base + "-arrangement.mid";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

function instrumentName(key) {
  return INSTRUMENTS[key]?.name || key;
}
