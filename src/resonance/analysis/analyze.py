"""Feature extraction for Resonance's arrangement pipeline.

The first analyzer is deterministic and intentionally explainable. It turns a
MIDI file into a JSON-serializable musical representation that later learned
models can consume as features or training targets.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from itertools import pairwise

import mido

from ..midi import load_midi

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl-Schmuckler key profiles. These are useful as a baseline; the final
# Resonance key model will be learned from audio/MIDI training data.
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


@dataclass(frozen=True)
class Note:
    pitch: int
    start: int
    end: int
    velocity: int
    channel: int
    track: int

    @property
    def duration(self) -> int:
        return max(1, self.end - self.start)

    @property
    def pitch_class(self) -> int:
        return self.pitch % 12


@dataclass(frozen=True)
class Section:
    index: int
    start_tick: int
    end_tick: int
    label: str
    energy: float
    note_density: float


def _extract_notes(midi: mido.MidiFile) -> list[Note]:
    notes: list[Note] = []
    for track_index, track in enumerate(midi.tracks):
        absolute = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity > 0:
                active[(message.channel, message.note)].append((absolute, message.velocity))
            elif message.type in {"note_off", "note_on"}:
                if message.type == "note_on" and message.velocity != 0:
                    continue
                key = (message.channel, message.note)
                starts = active.get(key)
                if starts:
                    start, velocity = starts.pop(0)
                    notes.append(Note(message.note, start, absolute, velocity, message.channel, track_index))
    return sorted(notes, key=lambda n: (n.start, n.pitch, n.track))


def _weighted_pitch_classes(notes: list[Note]) -> list[float]:
    values = [0.0] * 12
    for note in notes:
        values[note.pitch_class] += note.duration * max(1, note.velocity) / 127.0
    return values


def _correlation(a: list[float], b: tuple[float, ...]) -> float:
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    da = [x - mean_a for x in a]
    db = [x - mean_b for x in b]
    denominator = math.sqrt(sum(x * x for x in da) * sum(x * x for x in db))
    return 0.0 if denominator == 0 else sum(x * y for x, y in zip(da, db)) / denominator


def detect_key(notes: list[Note]) -> dict[str, Any]:
    distribution = _weighted_pitch_classes(notes)
    candidates: list[tuple[float, str, str]] = []
    for tonic in range(12):
        rotated = distribution[tonic:] + distribution[:tonic]
        candidates.append((_correlation(rotated, MAJOR_PROFILE), NOTE_NAMES[tonic], "major"))
        candidates.append((_correlation(rotated, MINOR_PROFILE), NOTE_NAMES[tonic], "minor"))
    candidates.sort(reverse=True)
    score, tonic, mode = candidates[0]
    second = candidates[1][0] if len(candidates) > 1 else 0.0
    confidence = max(0.0, min(1.0, (score - second + 1.0) / 2.0))
    return {
        "tonic": tonic,
        "mode": mode,
        "key": f"{tonic} {mode}",
        "confidence": round(confidence, 4),
        "pitch_class_distribution": [round(x, 4) for x in distribution],
    }


def _bar_boundaries(midi: mido.MidiFile, end_tick: int) -> list[int]:
    beats = midi.ticks_per_beat
    numerator = 4
    denominator = 4
    changes: list[tuple[int, int, int]] = [(0, numerator, denominator)]
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += message.time
            if message.type == "time_signature":
                changes.append((absolute, message.numerator, message.denominator))
    changes.sort()
    boundaries = [0]
    current = 0
    change_index = 0
    while current < end_tick:
        while change_index + 1 < len(changes) and changes[change_index + 1][0] <= current:
            change_index += 1
            _, numerator, denominator = changes[change_index]
        bar_ticks = int(beats * numerator * 4 / denominator)
        bar_ticks = max(1, bar_ticks)
        current += bar_ticks
        boundaries.append(min(current, end_tick))
    return boundaries


def _chord_name(pitches: list[int]) -> tuple[str, float]:
    if not pitches:
        return "N.C.", 0.0
    pcs = sorted({p % 12 for p in pitches})
    best: tuple[float, str] = (0.0, "N.C.")
    templates = {
        "maj": {0, 4, 7}, "min": {0, 3, 7}, "dim": {0, 3, 6},
        "sus2": {0, 2, 7}, "sus4": {0, 5, 7},
        "7": {0, 4, 7, 10}, "maj7": {0, 4, 7, 11}, "min7": {0, 3, 7, 10},
    }
    for root in range(12):
        for suffix, template in templates.items():
            present = {(pc - root) % 12 for pc in pcs}
            intersection = len(present & template)
            extra = len(present - template)
            missing = len(template - present)
            score = intersection / len(template) - 0.12 * extra - 0.08 * missing
            if score > best[0]:
                best = (score, f"{NOTE_NAMES[root]}{suffix}")
    return best[1], round(max(0.0, min(1.0, best[0])), 4)


def detect_chords(notes: list[Note], bars: list[int]) -> list[dict[str, Any]]:
    chords: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(pairwise(bars)):
        pitches = [n.pitch for n in notes if n.start < end and n.end > start]
        name, confidence = _chord_name(pitches)
        chords.append({"bar": index + 1, "start_tick": start, "end_tick": end, "chord": name, "confidence": confidence})
    return chords


def extract_melody(notes: list[Note]) -> list[dict[str, Any]]:
    """Extract a conservative monophonic melody candidate from onset groups."""
    melody: list[dict[str, Any]] = []
    by_start: dict[int, list[Note]] = defaultdict(list)
    for note in notes:
        by_start[note.start].append(note)
    for start in sorted(by_start):
        candidates = by_start[start]
        note = max(candidates, key=lambda n: (n.pitch, n.duration, n.velocity))
        if melody and note.pitch == melody[-1]["pitch"] and start <= melody[-1]["end_tick"]:
            melody[-1]["end_tick"] = max(melody[-1]["end_tick"], note.end)
        else:
            melody.append({"pitch": note.pitch, "start_tick": note.start, "end_tick": note.end, "velocity": note.velocity})
    return melody


def _bar_features(notes: list[Note], start: int, end: int) -> tuple[float, float, tuple[int, ...]]:
    inside = [n for n in notes if n.start < end and n.end > start]
    if not inside:
        return 0.0, 0.0, tuple([0] * 12)
    density = len(inside) / max(1.0, (end - start))
    energy = sum(n.velocity * max(1, min(n.duration, end - start)) for n in inside) / max(1.0, len(inside) * 127 * (end - start))
    pcs = [0] * 12
    for n in inside:
        pcs[n.pitch_class] += 1
    return density, energy, tuple(pcs)


def detect_sections(notes: list[Note], bars: list[int]) -> list[Section]:
    if len(bars) <= 1:
        return []
    features = [_bar_features(notes, a, b) for a, b in pairwise(bars)]
    sections: list[Section] = []
    start_bar = 0
    previous = features[0]
    for bar in range(1, len(features)):
        density, energy, pcs = features[bar]
        pd, pe, ppcs = previous
        distance = abs(density - pd) / max(0.0001, density + pd) + abs(energy - pe)
        pc_distance = 1.0 - (sum(min(a, b) for a, b in zip(pcs, ppcs)) / max(1, sum(pcs) + sum(ppcs)))
        boundary = distance + pc_distance * 0.5 > 0.75
        if boundary and bar - start_bar >= 2:
            group = features[start_bar:bar]
            avg_energy = sum(x[1] for x in group) / len(group)
            avg_density = sum(x[0] for x in group) / len(group)
            sections.append(Section(len(sections) + 1, bars[start_bar], bars[bar], chr(65 + len(sections)), avg_energy, avg_density))
            start_bar = bar
        previous = features[bar]
    group = features[start_bar:]
    sections.append(Section(len(sections) + 1, bars[start_bar], bars[-1], chr(65 + len(sections)), sum(x[1] for x in group) / len(group), sum(x[0] for x in group) / len(group)))
    return sections


def _importance(notes: list[Note]) -> list[dict[str, Any]]:
    if not notes:
        return []
    counts = Counter(n.pitch for n in notes)
    max_count = max(counts.values())
    result = []
    for note in notes:
        repetition = counts[note.pitch] / max_count
        duration_score = min(1.0, note.duration / max(1, max(n.duration for n in notes)))
        register_score = 1.0 - abs(note.pitch - 72) / 48
        score = 0.45 * repetition + 0.30 * duration_score + 0.25 * max(0.0, register_score)
        result.append({"pitch": note.pitch, "start_tick": note.start, "end_tick": note.end, "score": round(max(0.0, min(1.0, score)), 4)})
    return result


def analyze_midi(path: Path) -> dict[str, Any]:
    """Analyze a MIDI file and return a JSON-serializable feature object."""
    midi = load_midi(path)
    notes = _extract_notes(midi)
    end_tick = max((n.end for n in notes), default=0)
    bars = _bar_boundaries(midi, end_tick)
    key = detect_key(notes) if notes else {"key": None, "tonic": None, "mode": None, "confidence": 0.0, "pitch_class_distribution": [0.0] * 12}
    sections = detect_sections(notes, bars)
    chords = detect_chords(notes, bars)
    tracks = []
    for index, track in enumerate(midi.tracks):
        track_notes = [n for n in notes if n.track == index]
        name = next((m.name for m in track if m.type == "track_name"), f"Track {index + 1}")
        tracks.append({"index": index, "name": name, "notes": len(track_notes), "lowest_pitch": min((n.pitch for n in track_notes), default=None), "highest_pitch": max((n.pitch for n in track_notes), default=None), "channels": sorted({n.channel for n in track_notes})})
    tempo = None
    for track in midi.tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo = round(mido.tempo2bpm(message.tempo), 3)
                break
        if tempo is not None:
            break
    return {
        "version": 1,
        "source": str(path),
        "ticks_per_beat": midi.ticks_per_beat,
        "duration_ticks": end_tick,
        "tempo_bpm": tempo,
        "note_count": len(notes),
        "track_count": len(midi.tracks),
        "tracks": tracks,
        "key": key,
        "melody": extract_melody(notes),
        "chords": chords,
        "sections": [asdict(section) for section in sections],
        "note_importance": _importance(notes),
    }
