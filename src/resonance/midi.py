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


def transcription_to_midi(transcription, path: Path) -> None:
    """Write independently detected transcription parts to a multi-track MIDI."""
    from .transcription.multi_instrument import MultiInstrumentTranscription

    if not isinstance(transcription, MultiInstrumentTranscription):
        raise TypeError("transcription must be a MultiInstrumentTranscription")

    tempo_bpm = transcription.tempo or 120.0
    tempo = mido.bpm2tempo(tempo_bpm)
    midi = mido.MidiFile(ticks_per_beat=480)

    for part in transcription.parts:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=part.name, time=0))

        if part.program is not None and 0 <= part.program <= 127:
            track.append(mido.Message("program_change", program=part.program, time=0))

        events: list[tuple[int, int, mido.Message]] = []
        for note in part.notes:
            start_tick = max(
                0,
                round(mido.second2tick(note.start, midi.ticks_per_beat, tempo)),
            )
            end_tick = max(
                start_tick + 1,
                round(mido.second2tick(note.end, midi.ticks_per_beat, tempo)),
            )
            events.append(
                (
                    start_tick,
                    1,
                    mido.Message(
                        "note_on",
                        note=note.pitch,
                        velocity=note.velocity,
                    ),
                )
            )
            events.append(
                (
                    end_tick,
                    0,
                    mido.Message(
                        "note_off",
                        note=note.pitch,
                        velocity=0,
                    ),
                )
            )

        events.sort(key=lambda event: (event[0], event[1], event[2].note))
        previous_tick = 0
        for tick, _order, message in events:
            message.time = tick - previous_tick
            track.append(message)
            previous_tick = tick

        midi.tracks.append(track)

    if not midi.tracks:
        midi.tracks.append(mido.MidiTrack())

    midi.tracks[0].insert(
        0,
        mido.MetaMessage("set_tempo", tempo=tempo, time=0),
    )
    save_midi(midi, Path(path))
