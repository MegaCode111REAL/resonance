"""Deterministic baseline arranger.

This is intentionally not the final Resonance AI. It provides a complete
target-instrument arrangement pipeline and a reference implementation for
future learned arrangement models.
"""

from __future__ import annotations

from pathlib import Path

import mido

from ..instruments import Instrument, get_instrument
from ..midi import load_midi, save_midi


def _clamp_pitch(pitch: int, instrument: Instrument, part_index: int, part_count: int) -> int:
    """Move a pitch into the instrument's usable range with octave shifts."""
    pitch = int(pitch)
    while pitch < instrument.low_midi:
        pitch += 12
    while pitch > instrument.high_midi:
        pitch -= 12

    if part_count > 1:
        span = instrument.preferred_high_midi - instrument.preferred_low_midi
        if span > 0:
            section = span / part_count
            center = instrument.preferred_low_midi + section * (part_index + 0.5)
            while pitch < center - 18 and pitch + 12 <= instrument.high_midi:
                pitch += 12
            while pitch > center + 18 and pitch - 12 >= instrument.low_midi:
                pitch -= 12

    return max(instrument.low_midi, min(instrument.high_midi, pitch))


def _source_note_messages(midi: mido.MidiFile) -> list[tuple[int, mido.Message]]:
    """Flatten note events while preserving absolute tick positions."""
    events: list[tuple[int, mido.Message]] = []

    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += message.time
            if message.type in {"note_on", "note_off"}:
                if message.type == "note_on" and message.velocity == 0:
                    message = message.copy(type="note_off")
                events.append((absolute, message.copy()))

    return events


def arrange(
    source_path: Path,
    output_path: Path,
    instrument_name: str,
    parts: int,
) -> Path:
    """Create a target-instrument arrangement from source MIDI."""
    if parts < 1:
        raise ValueError("parts must be at least 1")

    source = load_midi(source_path)
    instrument = get_instrument(instrument_name)

    target = mido.MidiFile(type=1, ticks_per_beat=source.ticks_per_beat)

    conductor = mido.MidiTrack()
    conductor.append(
        mido.MetaMessage(
            "track_name",
            name=f"Resonance arrangement: {instrument.name} x{parts}",
            time=0,
        )
    )
    target.tracks.append(conductor)

    tracks = [mido.MidiTrack() for _ in range(parts)]
    for index, track in enumerate(tracks):
        track.append(
            mido.MetaMessage(
                "track_name",
                name=f"{instrument.name.replace('_', ' ').title()} {index + 1}",
                time=0,
            )
        )
        track.append(
            mido.Message(
                "program_change",
                program=instrument.program,
                channel=index % 16,
                time=0,
            )
        )
        target.tracks.append(track)

    events = _source_note_messages(source)
    events.sort(key=lambda event: (event[0], event[1].note, event[1].type))

    active: dict[tuple[int, int], tuple[int, mido.Message]] = {}
    output_events: list[list[tuple[int, mido.Message]]] = [[] for _ in range(parts)]

    for absolute, message in events:
        key = (message.channel, message.note)

        if message.type == "note_on" and message.velocity > 0:
            bucket = (message.note + absolute) % parts
            pitch = _clamp_pitch(message.note, instrument, bucket, parts)
            rewritten = message.copy(
                note=pitch,
                channel=bucket % 16,
                velocity=max(1, min(127, int(message.velocity))),
            )
            active[key] = (bucket, rewritten)
            output_events[bucket].append((absolute, rewritten))

        elif message.type == "note_off":
            started = active.pop(key, None)
            if started is None:
                continue
            bucket, original = started
            rewritten = message.copy(
                note=original.note,
                channel=bucket % 16,
                velocity=0,
            )
            output_events[bucket].append((absolute, rewritten))

    for index, track in enumerate(tracks):
        previous = 0
        for absolute, message in sorted(output_events[index], key=lambda event: event[0]):
            track.append(message.copy(time=max(0, absolute - previous)))
            previous = absolute

    save_midi(target, output_path)
    return output_path
