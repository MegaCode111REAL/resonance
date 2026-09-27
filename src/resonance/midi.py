"""MIDI utilities used throughout Resonance."""

from __future__ import annotations

from pathlib import Path

import mido


def validate_midi_path(path: Path) -> Path:
    """Validate and return a MIDI path."""
    path = Path(path)
    if path.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError(f"Expected a MIDI file, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_midi(path: Path) -> mido.MidiFile:
    """Load a MIDI file."""
    validate_midi_path(path)
    return mido.MidiFile(path)


def save_midi(midi: mido.MidiFile, path: Path) -> None:
    """Save a MIDI file, creating its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def clone_track(track: mido.MidiTrack) -> mido.MidiTrack:
    """Make a copy of a track's MIDI messages."""
    cloned = mido.MidiTrack()
    for message in track:
        cloned.append(message.copy())
    return cloned
