"""MIDI transposition and automatic key optimization.

The automatic optimizer tests every chromatic transposition and chooses the
one whose resulting pitched notes contain the fewest piano black-key pitches.
This is a simple, deterministic post-arrangement optimization; it does not
change rhythm, velocity, channels, programs, or note durations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from ..instruments import Instrument, get_instrument
from ..midi import load_midi, save_midi

BLACK_KEY_PITCH_CLASSES = frozenset({1, 3, 6, 8, 10})
KEY_NAMES = ("C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B")


@dataclass(frozen=True)
class TransposeResult:
    semitones: int
    key: str
    black_key_notes: int
    total_notes: int


def _note_messages(midi: mido.MidiFile):
    for track_index, track in enumerate(midi.tracks):
        for message_index, message in enumerate(track):
            if message.type == "note_on" and message.velocity > 0:
                yield track_index, message_index, message


def _score_shift(
    pitches: list[int],
    shift: int,
    instrument: Instrument | None = None,
) -> tuple[int, int, int]:
    transposed = [pitch + shift for pitch in pitches]
    out_of_midi = sum(pitch < 0 or pitch > 127 for pitch in transposed)
    out_of_range = 0
    if instrument is not None:
        out_of_range = sum(
            pitch < instrument.low_midi or pitch > instrument.high_midi
            for pitch in transposed
        )
    black_keys = sum(pitch % 12 in BLACK_KEY_PITCH_CLASSES for pitch in transposed)
    return out_of_midi, out_of_range, black_keys


def choose_best_transposition(
    midi: mido.MidiFile,
    instrument_name: str | None = None,
) -> TransposeResult:
    """Return the chromatic shift with the fewest black-key notes.

    MIDI/range violations are rejected before accidental count. Ties are
    resolved by the smallest absolute shift, then the smallest positive
    shift, so an unchanged arrangement is preferred when it ties.
    """
    pitches = [message.note for _, _, message in _note_messages(midi)]
    if not pitches:
        return TransposeResult(0, KEY_NAMES[0], 0, 0)

    instrument = get_instrument(instrument_name) if instrument_name else None
    candidates = []
    for shift in range(-11, 12):
        out_of_midi, out_of_range, black_keys = _score_shift(pitches, shift, instrument)
        candidates.append(
            (
                out_of_midi,
                out_of_range,
                black_keys,
                abs(shift),
                0 if shift == 0 else 1,
                shift,
            )
        )

    _, _, black_keys, _, _, shift = min(candidates)
    return TransposeResult(
        semitones=shift,
        key=KEY_NAMES[shift % 12],
        black_key_notes=black_keys,
        total_notes=len(pitches),
    )


def transpose_midi(
    source_path: Path,
    output_path: Path,
    semitones: int,
) -> Path:
    """Write a MIDI file transposed by a chromatic number of semitones."""
    if not -127 <= semitones <= 127:
        raise ValueError("semitones must be between -127 and 127")

    midi = load_midi(source_path)
    for _, _, message in _note_messages(midi):
        new_pitch = message.note + semitones
        if not 0 <= new_pitch <= 127:
            raise ValueError(
                f"transposition would create MIDI note {new_pitch}; "
                "choose a smaller interval"
            )
        message.note = new_pitch

    save_midi(midi, output_path)
    return output_path


def auto_transpose_midi(
    source_path: Path,
    output_path: Path,
    instrument_name: str | None = None,
) -> TransposeResult:
    """Transpose a MIDI arrangement to the least-accidental chromatic key."""
    midi = load_midi(source_path)
    result = choose_best_transposition(midi, instrument_name)

    for _, _, message in _note_messages(midi):
        message.note += result.semitones

    save_midi(midi, output_path)
    return result
