(() => {
  "use strict";
  const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);
  const KEY_NAMES = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"];
  const fileInput = document.querySelector("#midi-file");
  const panel = document.querySelector("#arrangement-panel");
  const autoButton = document.querySelector("#auto-transpose");
  const downloadButton = document.querySelector("#download");
  const status = document.querySelector("#status");
  const score = document.querySelector("#score");
  const result = document.querySelector("#result");
  let midi = null;
  let sourceName = "arrangement.mid";
  let downloadBytes = null;

  function noteMessages() {
    return midi.tracks.flatMap(track => track.notes);
  }

  function scoreShift(notes, shift) {
    let invalid = 0;
    let black = 0;
    for (const note of notes) {
      const pitch = note.midi + shift;
      if (pitch < 0 || pitch > 127) invalid++;
      else if (BLACK_KEYS.has(pitch % 12)) black++;
    }
    return { invalid, black };
  }

  function chooseBestShift(notes) {
    const candidates = [];
    for (let shift = -11; shift <= 11; shift++) {
      const scored = scoreShift(notes, shift);
      candidates.push({ shift, ...scored, distance: Math.abs(shift), unchanged: shift === 0 ? 0 : 1 });
    }
    candidates.sort((a, b) =>
      a.invalid - b.invalid || a.black - b.black ||
      a.distance - b.distance || a.unchanged - b.unchanged || a.shift - b.shift
    );
    return candidates[0];
  }

  function applyShift(notes, shift) {
    for (const note of notes) note.midi += shift;
  }

  function renderResult(best, total) {
    const key = KEY_NAMES[((best.shift % 12) + 12) % 12];
    const direction = best.shift === 0 ? "No transposition needed" : (best.shift > 0 ? "+" : "") + best.shift + " semitones";
    score.textContent = best.black + " / " + total + " black-key notes";
    result.classList.remove("hidden");
    result.textContent = key + " · " + direction + " · " + best.black + " black-key notes";
  }

  function setDownload() {
    downloadBytes = midi.toArray();
    downloadButton.disabled = false;
  }

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    try {
      status.textContent = "Loading MIDI…";
      midi = new Midi(await file.arrayBuffer());
      sourceName = file.name;
      panel.classList.remove("hidden");
      autoButton.disabled = false;
      setDownload();
      result.classList.add("hidden");
      score.textContent = "—";
      status.textContent = midi.tracks.length + " tracks · " + noteMessages().length + " notes";
    } catch (error) {
      console.error(error);
      status.textContent = "Could not read that MIDI";
      panel.classList.add("hidden");
    }
  });

  autoButton.addEventListener("click", () => {
    if (!midi) return;
    const notes = noteMessages();
    if (!notes.length) {
      status.textContent = "No pitched notes found";
      return;
    }
    autoButton.disabled = true;
    autoButton.textContent = "Testing 12 keys…";
    requestAnimationFrame(() => {
      const best = chooseBestShift(notes);
      applyShift(notes, best.shift);
      renderResult(best, notes.length);
      setDownload();
      status.textContent = "Optimized to " + KEY_NAMES[((best.shift % 12) + 12) % 12];
      autoButton.disabled = false;
      autoButton.textContent = "Auto Transpose";
    });
  });

  downloadButton.addEventListener("click", () => {
    if (!downloadBytes) return;
    const blob = new Blob([downloadBytes], { type: "audio/midi" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = sourceName.replace(/\.midi?$/i, "") + "-auto-transposed.mid";
    link.click();
    URL.revokeObjectURL(url);
  });
})();
